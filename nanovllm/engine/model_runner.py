import gc
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.spec_decode import Eagle3SpecBackend, DFlashSpecBackend


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.spec_profile = config.spec_profile
        self.spec_debug = config.spec_debug
        self.world_size = config.tensor_parallel_size
        self.rank = rank #process rank
        self.event = event

        if not dist.is_initialized():
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config) # 初始化模型结构
        self.model.model.spec_debug = self.spec_debug
        load_model(self.model, config.model) #load 模型参数
        self.sampler = Sampler()
        # Speculative decoding backend
        self.has_spec = config.draft_model is not None
        if self.has_spec:
            if config.spec_method == "eagle3":
                self.spec_backend = Eagle3SpecBackend(self)
            elif config.spec_method == "dflash":
                self.spec_backend = DFlashSpecBackend(self)
            else:
                raise ValueError(f"unsupported spec_method: {config.spec_method}")
        self.warmup_model() # 预跑一下模型, 预估得到kv的内存空间
        self.allocate_kv_cache() # 里面会实际在GPU上分配内存
        if not self.enforce_eager:
            if self.has_spec:
                self.capture_spec_decode_cudagraph()
            else:
                self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) # 1MB
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            if hasattr(self, 'graphs'):
                del self.graphs
            if hasattr(self, 'graph_pool'):
                del self.graph_pool
            if hasattr(self, 'spec_graphs'):
                del self.spec_graphs
            if hasattr(self, 'spec_graph_pool'):
                del self.spec_graph_pool
            if hasattr(self, 'spec_graph_vars'):
                del self.spec_graph_vars
        if hasattr(self, 'spec_backend'):
            del self.spec_backend
        if hasattr(self, 'draft_kv_cache'):
            del self.draft_kv_cache
        if hasattr(self, 'draft_model'):
            del self.draft_model
        del self.kv_cache, self.model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()  #clears memory,
        torch.cuda.reset_peak_memory_stats() #resets the trackers you use to monitor that memory
        # max_num_batched_tokens 们会将多个请求(Requests)组合成一个 Batch。由于不同的请求长度不同,GPU 处理的压力主要取决于 "这一秒我手里一共有多少个 Token"。max_num_batched_tokens 限制的就是这个总数
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len # 16384, 4096
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs) #min(4, 512)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.scheduled_chunk_size = max_model_len
        # warmup 时临时关闭 spec，走普通路径估算显存
        # 这样 @torch.compile 函数在 @torch.inference_mode() 下编译，避免 autograd 冲突
        saved_has_spec = self.has_spec
        self.has_spec = False
        self.run(seqs, []) #只跑prefill
        self.has_spec = saved_has_spec
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        '''
        每一个 Block(块)并不是只代表一层,而是代表整个模型所有层中对应那几个 Token 的缓存集合
        '''
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        if self.has_spec:
            draft_config = self.config.draft_hf_config
            draft_num_kv_heads = draft_config.num_key_value_heads // self.world_size
            draft_head_dim = getattr(draft_config, "head_dim", draft_config.hidden_size // draft_config.num_attention_heads)
            draft_num_layers = 1 if self.config.spec_method == "eagle3" else draft_config.num_hidden_layers
            draft_block_bytes = 2 * draft_num_layers * self.block_size * draft_num_kv_heads * draft_head_dim * hf_config.torch_dtype.itemsize
            block_bytes += draft_block_bytes
        '''
        total * config.gpu_memory_utilization - used - peak + current
        total * config.gpu_memory_utilization - used - (peak - current)
        current 和 peak都是指torch 申请tensor占用的内存
        used：当前 GPU 已用显存（所有来源：tensor、驱动、其他进程）
        peak - current: 模型运行时 tensor 内存的峰值波动(forward 中的临时激活值等）
        '''
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 2. 把每层的切片绑定到对应 Attention 模块
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"): # 只有attention层才有这个属性
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        # 3. Draft model KV cache
        if self.has_spec:
            draft_config = self.config.draft_hf_config
            draft_num_kv_heads = draft_config.num_key_value_heads // self.world_size
            draft_head_dim = getattr(draft_config, "head_dim", draft_config.hidden_size // draft_config.num_attention_heads)
            draft_num_layers = 1 if self.config.spec_method == "eagle3" else draft_config.num_hidden_layers
            self.draft_kv_cache = torch.empty(2, draft_num_layers, config.num_kvcache_blocks, self.block_size, draft_num_kv_heads, draft_head_dim)
            draft_layer_id = 0
            for module in self.draft_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.draft_kv_cache[0, draft_layer_id]
                    module.v_cache = self.draft_kv_cache[1, draft_layer_id]
                    draft_layer_id += 1
            assert draft_layer_id == draft_num_layers, (draft_layer_id, draft_num_layers)

    #  把所有 seq 的 block_table 填充到相同长度后拼成一个 2D GPU tensor
    # 不同 seq 的 block_table 长度不同(取决于 seq 长度和已分配的 blocks),tensor 需要统一形状,所以用 -1 补齐短的。-1 表示无效 block,attention不会访问到这些位置。
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # chunked prefill: 只处理 [start, end) 这个 chunk
            start = seq.num_computed_tokens
            end = seq.num_computed_tokens + seq.scheduled_chunk_size
            seqlen_q = end - start
            seqlen_k = end  # KV cache 中已有的 + 本轮新算的
            token_ids = seq[start:end]
            assert len(token_ids) == seqlen_q, (
                "prefill token/position length mismatch",
                {
                    "seq_id": seq.seq_id,
                    "start": start,
                    "end": end,
                    "seqlen_q": seqlen_q,
                    "slice_len": len(token_ids),
                    "num_tokens": seq.num_tokens,
                    "token_ids_len": len(seq.token_ids),
                    "num_computed_tokens": seq.num_computed_tokens,
                    "scheduled_chunk_size": seq.scheduled_chunk_size,
                    "status": str(seq.status),
                },
            )
            input_ids.extend(token_ids) # 把不同seq 的token展平, 解决不同seq维度不对齐的问题
            positions.extend(list(range(start, end)))
            # 用于 Flash Attention
            # Q 决定"谁需要算",K 决定"能看到什么"。缓存的 token 不需要重新算,但必须能被看到。
            # seqlen_k = end(到本轮结束的长度)描述的是 KV cache 中有多少有效 token,不是要重新计算多少。
            # 相当于push了一个(cu_seqlens_q.back() + seqlen_q)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            '''
            slot_mapping: 只映射本轮 chunk [start, end) 需要写入 KV cache 的 slot。

            chunked prefill 下 [start, end) 可能落在 block 中间,
            所以需要以 token 粒度精确切片,而非整块跳过/写入。
            例如 block 覆盖 token 0~255,但本轮只需处理 token 200~255,
            就只映射这 56 个 slot。
            '''
            for i in range(seq.num_blocks):
                token_start = i * self.block_size
                token_end = min((i + 1) * self.block_size, end)
                if token_end <= start:
                    continue
                if token_start >= end:
                    break
                block_offset = seq.block_table[i] * self.block_size
                within_start = max(0, start - token_start)
                within_end = token_end - token_start
                slot_mapping.extend(list(range(block_offset + within_start, block_offset + within_end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache or chunked prefill
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables

    def prepare_decode(self, seqs: list[Sequence]):
        # decode 每个 seq 只处理 1 个 token(最后一个),所以每项都只 append 一次
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)        # 本轮要 forward 的 token
            positions.append(len(seq) - 1)           # 该 token 在序列中的位置(0-indexed)
            context_lens.append(len(seq))            # attention 能看到的 KV 条目数(含自身)
            # slot_mapping: 本轮 token 的 KV 写入 KV cache 的物理位置
            # = 最后一个 block 的起始地址 + block 内偏移
            # 例: 长度 260, block_size=256 → block_table[-1]*256 + (260-256) - 1 = block*256 + 3
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        return input_ids, positions, slot_mapping, context_lens, block_tables

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, has_prefill: bool):
        if has_prefill or self.enforce_eager or input_ids.size(0) > 512: # prefill
            #把模型输出的hidden stats 再 经过一层得到特征向量
            return self.model.compute_logits(self.model(input_ids, positions))
        else: #decode
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.decode_slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.decode_context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]) -> list[int]: #从调度到模型真正执行
        has_prefill = len(prefill_seqs) > 0
        has_decode = len(decode_seqs) > 0

        # 判断哪些 prefill seq 在本轮完成(需要采样)
        finishing_prefill_indices = [i for i, seq in enumerate(prefill_seqs)
                                     if seq.num_computed_tokens + seq.scheduled_chunk_size >= seq.num_tokens]
        finishing_prefill_seqs = [prefill_seqs[i] for i in finishing_prefill_indices]

        if has_prefill:
            p_input_ids, p_positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, p_slot_mapping, p_block_tables = self.prepare_prefill(prefill_seqs)

        if has_decode:
            d_input_ids, d_positions, d_slot_mapping, d_context_lens, d_block_tables = self.prepare_decode(decode_seqs)

        # 拼接输入
        if has_prefill:
            assert p_input_ids.shape[0] == p_positions.shape[0], (
                "prefill input/position length mismatch",
                p_input_ids.shape,
                p_positions.shape,
            )
        if has_decode:
            assert d_input_ids.shape[0] == d_positions.shape[0], (
                "decode input/position length mismatch",
                d_input_ids.shape,
                d_positions.shape,
            )

        if has_prefill and has_decode:
            input_ids = torch.cat([p_input_ids, d_input_ids])
            positions_cat = torch.cat([p_positions, d_positions])
        elif has_prefill:
            input_ids = p_input_ids
            positions_cat = p_positions
        else:
            input_ids = d_input_ids
            positions_cat = d_positions

        # 需要采样的 seq
        sample_seqs = finishing_prefill_seqs + decode_seqs
        temperatures = self.prepare_sample(sample_seqs) if self.rank == 0 and sample_seqs else None

        # 设置 context
        if has_prefill:
            set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        p_slot_mapping, block_tables=p_block_tables,
                        num_prefill_tokens=p_input_ids.size(0),
                        num_decode_seqs=len(decode_seqs),
                        finishing_prefill_indices=finishing_prefill_indices,
                        decode_slot_mapping=d_slot_mapping if has_decode else None,
                        decode_context_lens=d_context_lens if has_decode else None,
                        decode_block_tables=d_block_tables if has_decode else None)
        else:
            set_context(False, decode_slot_mapping=d_slot_mapping, decode_context_lens=d_context_lens, block_tables=d_block_tables,
                        num_decode_seqs=len(decode_seqs))

        assert input_ids.shape[0] == positions_cat.shape[0], (
            input_ids.shape, positions_cat.shape
        )

        logits = self.run_model(input_ids, positions_cat, has_prefill)

        if self.rank == 0 and sample_seqs:
            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None

        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_speculative_step(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]) -> dict | None:
        return self.spec_backend.run_step(prefill_seqs, decode_seqs)

    def set_block_manager(self, block_manager: BlockManager):
        """供 engine 在初始化后设置 block_manager 引用"""
        self.block_manager = block_manager

    def reset_spec_profile_metrics(self):
        self.spec_backend.reset_profile_metrics()

    @torch.inference_mode()
    def capture_spec_decode_cudagraph(self):
        self.spec_backend.capture_decode_cudagraph()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512) #bs batch size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 小 bs 用 1/2/4/8 精细覆盖(decode 常见场景),大 bs 按 16 步长递增。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, decode_slot_mapping=slot_mapping[:bs], decode_context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

"""
pin_memory=True (锁页内存)
这是性能优化的第一步。

普通内存(Pageable Memory):默认情况下,CPU 内存是可分页的,操作系统可能会将其交换到磁盘。GPU 无法直接访问这种内存。

锁页内存(Pinned Memory):设置 pin_memory=True 会在 CPU 内存中申请一块固定区域,不会被交换到磁盘。

优势:GPU 可以通过 DMA(直接内存访问) 技术直接从这块内存读取数据,跳过 CPU 的干预,传输速度大幅提升。

.cuda(non_blocking=True) (异步传输)
这是性能优化的第二步。

同步(Default):CPU 发起拷贝指令后,必须等待数据完全传输到 GPU 才能执行下一行代码。

异步(non_blocking=True):CPU 只要发出了"开始传输"的指令,就立即执行后面的代码,不需要等待传输完成。

前提条件:异步传输必须配合 pin_memory=True 才能真正发挥作用。

2. 为什么要这样写？(性能视角)
在 LLM 推理(如 vLLM)中,每一毫秒都至关重要。

流水线并行(Overlapping):当 GPU 还在忙着处理上一组计算时,CPU 已经可以通过 non_blocking=True 把下一组 input_ids 异步塞进显存了。

减少 CPU 负担:使用 DMA 传输时,CPU 不需要参与数据搬运的细节,可以腾出空位去处理更复杂的逻辑(如调度、采样等)。

3. 直观的类比
想象你在往货车(GPU)上搬运货物(Data):

普通写法:你把货物放在路边(普通内存),搬运工得先把它搬到站台(锁页内存),然后再搬上车。你必须站在旁边看着车装满才走(同步)。

这段代码的写法:

你提前把货物放到了专属的快速站台(pin_memory=True)。

搬运工(DMA)直接从站台往车上装。

你跟搬运工说"你开始装吧",然后转头就去忙别的了(non_blocking=True)。
"""
