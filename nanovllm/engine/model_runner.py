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
        # EAGLE3 draft model
        self.has_spec = config.draft_model is not None
        if self.has_spec:
            from nanovllm.models.eagle3 import Eagle3DraftModel
            self.draft_model = Eagle3DraftModel(
                config.draft_hf_config,
                target_hidden_size=hf_config.hidden_size,
                num_fuse_layers=len(config.eagle3_fuse_layers),
            )
            self.draft_model.midlayer.spec_debug = self.spec_debug
            load_model(self.draft_model, config.draft_model)
            # d2t 在 checkpoint 中存储的是 offset（vLLM 约定: target_id = draft_id + d2t[draft_id]）
            # 转换为绝对 target ID，简化后续使用
            self.draft_model.d2t.add_(torch.arange(self.draft_model.draft_vocab_size, device='cuda'))
            # 共享 target model 的 embedding
            self.draft_model.embed_tokens = self.model.model.embed_tokens
            self.eagle3_fuse_layers = config.eagle3_fuse_layers
            self.num_spec_tokens = config.num_spec_tokens
        self.warmup_model() # 预跑一下模型, 预估得到kv的内存空间
        self.allocate_kv_cache() # 里面会实际在GPU上分配内存
        if not self.enforce_eager:
            self.capture_cudagraph()
            if self.has_spec:
                self.capture_spec_decode_cudagraph()
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
            del self.graphs, self.graph_pool
            if hasattr(self, 'spec_graphs'):
                del self.spec_graphs
            if hasattr(self, 'spec_graph_vars'):
                del self.spec_graph_vars
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
            draft_head_dim = draft_config.head_dim
            draft_block_bytes = 2 * self.block_size * draft_num_kv_heads * draft_head_dim * hf_config.torch_dtype.itemsize
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
        # 3. Draft model KV cache(只有 1 层,很小)
        if self.has_spec:
            draft_config = self.config.draft_hf_config
            draft_num_kv_heads = draft_config.num_key_value_heads // self.world_size
            draft_head_dim = draft_config.head_dim
            self.draft_kv_cache = torch.empty(2, 1, config.num_kvcache_blocks, self.block_size, draft_num_kv_heads, draft_head_dim)
            for module in self.draft_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.draft_kv_cache[0, 0]
                    module.v_cache = self.draft_kv_cache[1, 0]

    #  把所有 seq 的 block_table 填充到相同长度后拼成一个 2D tensor
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

    def _fuse_captured_hidden(self, captured, indices=None):
        if indices is None:
            hidden = [captured[l] for l in self.eagle3_fuse_layers]
        else:
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(
                    indices,
                    dtype=torch.int64,
                    device=captured[self.eagle3_fuse_layers[0]].device,
                )
            hidden = [captured[l].index_select(0, indices) for l in self.eagle3_fuse_layers]
        return self.draft_model.fc(torch.cat(hidden, dim=-1))

    def _assign_prev_draft_tokens(self, seqs: list[Sequence], draft_tokens_all: list[torch.Tensor]):
        if not seqs:
            return
        draft_tokens = torch.stack(draft_tokens_all, dim=1).cpu().tolist()
        for seq, tokens in zip(seqs, draft_tokens):
            seq.prev_draft_tokens = tokens

    @torch.inference_mode()
    def _generate_draft_tokens_from_state(self, seqs, current_input, current_hidden,
                                          start_positions, num_steps,
                                          initial_tokens=None, first_slots=None,
                                          rollback_num_blocks=None):
        """从给定 EAGLE 状态继续 serial 生成 draft tokens。"""
        draft_tokens_all = list(initial_tokens) if initial_tokens is not None else []
        if num_steps <= 0:
            if rollback_num_blocks is not None:
                for seq, num_blocks in zip(seqs, rollback_num_blocks):
                    self.block_manager.rollback_blocks(seq, num_blocks)
            return draft_tokens_all

        N = len(seqs)
        if isinstance(start_positions, torch.Tensor):
            start_positions_t = start_positions.to(dtype=torch.int64, device='cuda')
        else:
            start_positions_t = torch.tensor(start_positions, dtype=torch.int64, device='cuda')

        if first_slots is not None:
            if isinstance(first_slots, torch.Tensor):
                first_slots_t = first_slots.to(dtype=torch.int32, device='cuda')
            else:
                first_slots_t = torch.tensor(first_slots, dtype=torch.int32, device='cuda')
        else:
            first_slots_t = None

        new_slots_per_seq = []
        num_new_slots = num_steps - (1 if first_slots_t is not None else 0)
        for seq in seqs:
            new_slots_per_seq.append(
                self.block_manager.append_n_slots(seq, num_new_slots) if num_new_slots > 0 else []
            )
        new_slots_t = (torch.tensor(new_slots_per_seq, dtype=torch.int32, device='cuda')
                       if num_new_slots > 0 else None)
        draft_block_tables = self.prepare_block_tables(seqs)

        for k in range(num_steps):
            draft_positions = start_positions_t + k
            if first_slots_t is not None and k == 0:
                draft_slot_mapping = first_slots_t
            else:
                slot_idx = k if first_slots_t is None else k - 1
                draft_slot_mapping = new_slots_t[:, slot_idx].contiguous()
            draft_context_lens = (draft_positions + 1).to(torch.int32)

            set_context(False, decode_slot_mapping=draft_slot_mapping,
                        decode_context_lens=draft_context_lens,
                        block_tables=draft_block_tables, num_decode_seqs=N)
            draft_logits, current_hidden = self.draft_model(current_input, draft_positions, current_hidden)
            reset_context()

            draft_token = draft_logits.argmax(dim=-1)
            target_token = self.draft_model.d2t[draft_token]

            # DEBUG: 前几步打印 draft logits 质量
            if self.spec_debug and not hasattr(self, '_debug_logits_printed'):
                if not hasattr(self, '_debug_logits_count'):
                    self._debug_logits_count = 0
                self._debug_logits_count += 1
                if self._debug_logits_count <= 3:
                    logits_0 = draft_logits[0]
                    topk_vals, topk_ids = logits_0.topk(5)
                    topk_target = self.draft_model.d2t[topk_ids]
                    print(f"    [DEBUG] draft k={k}: raw_argmax={draft_token[0].item()}, "
                          f"top5_draft={topk_ids.tolist()}, top5_target={topk_target.tolist()}, "
                          f"top5_vals={[f'{v:.2f}' for v in topk_vals.tolist()]}, "
                          f"logits_std={logits_0.std().item():.4f}, hidden_norm={current_hidden[0].norm().item():.2f}")
                if self._debug_logits_count >= 3:
                    self._debug_logits_printed = True

            current_input = target_token
            draft_tokens_all.append(target_token)

        if rollback_num_blocks is not None:
            for seq, num_blocks in zip(seqs, rollback_num_blocks):
                self.block_manager.rollback_blocks(seq, num_blocks)

        return draft_tokens_all

    @torch.inference_mode()
    def _propose_after_prefill(self, finishing_prefill_infos, sync_logits, sync_hidden_out):
        """Prefill 完成后立即生成 EAGLE draft tokens，下一步直接进入 verify。"""
        K = self.num_spec_tokens
        if K <= 0:
            return

        active_infos = []
        for seq, last_idx, sampled_token in finishing_prefill_infos:
            if (not seq.ignore_eos and sampled_token == self.config.eos) or seq.num_completion_tokens + 1 >= seq.max_tokens:
                seq.prev_draft_tokens = []
            else:
                active_infos.append((seq, last_idx))
        if not active_infos:
            return

        seqs = [info[0] for info in active_infos]
        last_indices = torch.tensor([info[1] for info in active_infos], dtype=torch.int64, device='cuda')
        original_lens = [len(seq) for seq in seqs]
        original_num_blocks = [seq.num_blocks for seq in seqs]

        # EAGLE first pass: sync 在 prompt 最后位置使用 sampled token，直接产出第一个 draft token。
        current_hidden = sync_hidden_out[last_indices]
        draft_token = sync_logits[last_indices].argmax(dim=-1)
        first_token = self.draft_model.d2t[draft_token]

        draft_tokens_all = self._generate_draft_tokens_from_state(
            seqs=seqs,
            current_input=first_token,
            current_hidden=current_hidden,
            start_positions=original_lens,
            num_steps=K - 1,
            initial_tokens=[first_token],
            rollback_num_blocks=original_num_blocks,
        )

        self._assign_prev_draft_tokens(seqs, draft_tokens_all)

    def _slot_for_position(self, seq: Sequence, pos: int) -> int:
        block_idx = pos // self.block_size
        block_offset = pos % self.block_size
        return seq.block_table[block_idx] * self.block_size + block_offset

    def _build_spec_target_batch(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]):
        K = self.num_spec_tokens
        rows = []
        seqs = []
        input_ids = []
        positions = []
        slot_mapping = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0

        for prefill_index, seq in enumerate(prefill_seqs):
            start = seq.num_computed_tokens
            end = start + seq.scheduled_chunk_size
            q_len = end - start
            k_len = end
            row_start = len(input_ids)
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            for pos in range(start, end):
                slot_mapping.append(self._slot_for_position(seq, pos))
            row_end = len(input_ids)
            rows.append({
                "kind": "prefill",
                "seq": seq,
                "prefill_index": prefill_index,
                "row_start": row_start,
                "row_end": row_end,
                "start": start,
                "end": end,
                "original_len": len(seq),
                "is_finishing_prefill": end >= seq.num_tokens,
            })
            seqs.append(seq)
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + k_len)
            max_q = max(max_q, q_len)
            max_k = max(max_k, k_len)

        for decode_index, seq in enumerate(decode_seqs):
            if len(seq.prev_draft_tokens) != K:
                raise RuntimeError(
                    f"spec decode seq {seq.seq_id} missing prev_draft_tokens: "
                    f"got {len(seq.prev_draft_tokens)}, expected {K}"
                )
            original_len = len(seq)
            # decode-verify forwards [last_token] + K drafts. The last_token is
            # already in the logical sequence; only K future draft/bonus positions
            # need pre-allocation.
            self.block_manager.append_n_slots(seq, K)
            row_start = len(input_ids)
            input_ids.append(seq.last_token)
            input_ids.extend(seq.prev_draft_tokens)
            positions.extend(range(original_len - 1, original_len + K))
            for pos in range(original_len - 1, original_len + K):
                slot_mapping.append(self._slot_for_position(seq, pos))
            row_end = len(input_ids)
            q_len = K + 1
            k_len = original_len + K
            rows.append({
                "kind": "decode_verify",
                "seq": seq,
                "decode_index": decode_index,
                "row_start": row_start,
                "row_end": row_end,
                "original_len": original_len,
                "num_prev_drafts": K,
            })
            seqs.append(seq)
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + k_len)
            max_q = max(max_q, q_len)
            max_k = max(max_k, k_len)

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs) if cu_k[-1] > cu_q[-1] else None
        return rows, input_ids_t, positions_t, cu_q_t, cu_k_t, max_q, max_k, slot_mapping_t, block_tables

    def _can_run_spec_decode_graph(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence],
                                   block_tables: torch.Tensor | None, max_k: int) -> bool:
        if self.enforce_eager or prefill_seqs or not decode_seqs:
            return False
        if not hasattr(self, 'spec_graphs') or not self.spec_graphs:
            return False
        if block_tables is None:
            return False
        if len(decode_seqs) > self.spec_graph_bs[-1]:
            return False
        if max_k > self.spec_graph_max_seqlen_k:
            return False
        return True

    def _run_spec_decode_graph(self, rows: list[dict], input_ids: torch.Tensor, positions: torch.Tensor,
                               cu_k: torch.Tensor, max_k: int, slot_mapping: torch.Tensor,
                               block_tables: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        bs = len(rows)
        graph_bs = next(x for x in self.spec_graph_bs if x >= bs)
        q_len = self.spec_graph_query_len
        num_tokens = bs * q_len
        graph_tokens = graph_bs * q_len
        graph = self.spec_graphs[graph_bs]
        graph_vars = self.spec_graph_vars

        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"][:graph_tokens].fill_(-1)
        graph_vars["slot_mapping"][:num_tokens] = slot_mapping
        graph_vars["cu_k"][:bs + 1] = cu_k
        if graph_bs > bs:
            cu_k_pad = [0]
            for row in rows:
                cu_k_pad.append(cu_k_pad[-1] + row["original_len"] + self.num_spec_tokens)
            while len(cu_k_pad) <= graph_bs:
                cu_k_pad.append(cu_k_pad[-1] + q_len)
            cu_k_pad_t = torch.tensor(cu_k_pad, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            graph_vars["cu_k"][:graph_bs + 1] = cu_k_pad_t
        graph_vars["block_tables"][:graph_bs].zero_()
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

        graph.replay()
        hidden_out = graph_vars["outputs"][:num_tokens]
        captured = {
            layer: graph_vars["captured"][layer][:num_tokens]
            for layer in self.eagle3_fuse_layers
        }
        return hidden_out, captured

    @torch.inference_mode()
    def run_speculative_step(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]) -> dict | None:
        K = self.num_spec_tokens
        fuse_layers = self.eagle3_fuse_layers
        profile = self.spec_profile

        if profile:
            import time as _time
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()

        rows, input_ids, positions, cu_q, cu_k, max_q, max_k, slot_mapping, block_tables = \
            self._build_spec_target_batch(prefill_seqs, decode_seqs)
        use_spec_graph = self._can_run_spec_decode_graph(prefill_seqs, decode_seqs, block_tables, max_k)

        if profile:
            torch.cuda.synchronize()
            _t1 = _time.perf_counter()

        if use_spec_graph:
            hidden_out, captured = self._run_spec_decode_graph(
                rows, input_ids, positions, cu_k, max_k, slot_mapping, block_tables
            )
        else:
            set_context(True, cu_q, cu_k, max_q, max_k, slot_mapping,
                        block_tables=block_tables,
                        num_prefill_tokens=input_ids.size(0),
                        num_decode_seqs=0,
                        finishing_prefill_indices=[])
            hidden_out, captured = self.model(input_ids, positions, capture_layers=fuse_layers)

        if profile:
            torch.cuda.synchronize()
            _t2 = _time.perf_counter()

        prefill_token_ids = [None] * len(prefill_seqs)
        finishing_rows = [row for row in rows if row["kind"] == "prefill" and row["is_finishing_prefill"]]
        decode_rows = [row for row in rows if row["kind"] == "decode_verify"]

        logit_indices = []
        for row in finishing_rows:
            row["sample_logit_index"] = len(logit_indices)
            logit_indices.append(row["row_end"] - 1)
        for row in decode_rows:
            row["logit_start"] = len(logit_indices)
            logit_indices.extend(range(row["row_start"], row["row_end"]))
            row["logit_end"] = len(logit_indices)

        logits_selected = None
        if logit_indices:
            logit_indices_t = torch.tensor(logit_indices, dtype=torch.int64, device=hidden_out.device)
            logits_selected = self.model.lm_head.forward_all(hidden_out.index_select(0, logit_indices_t))

        if finishing_rows:
            if self.rank == 0:
                sample_logits = torch.stack([logits_selected[row["sample_logit_index"]] for row in finishing_rows])
                temperatures = self.prepare_sample([row["seq"] for row in finishing_rows])
                sampled_tokens = self.sampler(sample_logits, temperatures).tolist()
            else:
                sampled_tokens = [0] * len(finishing_rows)
            for row, token_id in zip(finishing_rows, sampled_tokens):
                row["sampled_token"] = token_id
                if self.rank == 0:
                    prefill_token_ids[row["prefill_index"]] = token_id

        decode_target_token_ids = []
        if self.rank == 0 and decode_rows:
            decode_logits_start = len(finishing_rows)
            decode_target_token_ids = (
                logits_selected[decode_logits_start:]
                .argmax(dim=-1)
                .view(len(decode_rows), K + 1)
                .cpu()
                .tolist()
            )

        decode_accepted_tokens = [[] for _ in decode_seqs]
        for row_idx, row in enumerate(decode_rows):
            seq = row["seq"]
            if self.rank == 0:
                target_token_ids = decode_target_token_ids[row_idx]
                accepted = []
                for j in range(K):
                    target_pred = target_token_ids[j]
                    draft_tok = seq.prev_draft_tokens[j]
                    if target_pred == draft_tok:
                        accepted.append(draft_tok)
                    else:
                        accepted.append(target_pred)
                        break
                else:
                    accepted.append(target_token_ids[K])
            else:
                accepted = [0]
            if self.rank == 0:
                if not seq.ignore_eos and self.config.eos in accepted:
                    accepted = accepted[:accepted.index(self.config.eos) + 1]
                remaining_tokens = seq.max_tokens - seq.num_completion_tokens
                accepted = accepted[:remaining_tokens]
            row["accepted_tokens"] = accepted
            row["num_accepted"] = len(accepted)
            if self.rank == 0:
                decode_accepted_tokens[row["decode_index"]] = accepted

        if profile:
            torch.cuda.synchronize()
            _t3 = _time.perf_counter()

        reset_context()

        sync_input_ids = []
        sync_positions = []
        sync_slot_mapping = []
        sync_fused_indices = []
        sync_cu_q = [0]
        sync_cu_k = [0]
        sync_seqs = []
        sync_rows = []
        max_sync_q = 0
        max_sync_k = 0

        for row in rows:
            seq = row["seq"]
            if row["kind"] == "prefill":
                start = row["start"]
                end = row["end"]
                q_len = end - start
                shifted = list(seq.token_ids[start + 1:end])
                if row["is_finishing_prefill"]:
                    shifted.append(row["sampled_token"])
                else:
                    shifted.append(seq.token_ids[end])
                row_sync_start = len(sync_input_ids)
                sync_input_ids.extend(shifted)
                sync_positions.extend(range(start, end))
                for pos in range(start, end):
                    sync_slot_mapping.append(self._slot_for_position(seq, pos))
                sync_fused_indices.extend(range(row["row_start"], row["row_end"]))
                row_sync_end = len(sync_input_ids)
                sync_rows.append({
                    "kind": "prefill",
                    "seq": seq,
                    "source_row": row,
                    "row_start": row_sync_start,
                    "row_end": row_sync_end,
                })
                sync_seqs.append(seq)
                sync_cu_q.append(sync_cu_q[-1] + q_len)
                sync_cu_k.append(sync_cu_k[-1] + end)
                max_sync_q = max(max_sync_q, q_len)
                max_sync_k = max(max_sync_k, end)
            else:
                accepted = row["accepted_tokens"]
                M = row["num_accepted"]
                original_len = row["original_len"]
                final_len = original_len + M
                final_num_blocks = (final_len + self.block_size - 1) // self.block_size
                row["final_num_blocks"] = final_num_blocks
                self.block_manager.rollback_blocks(seq, final_num_blocks)

                will_finish = (
                    (not seq.ignore_eos and accepted[-1] == self.config.eos)
                    or seq.num_completion_tokens + M >= seq.max_tokens
                )
                if will_finish:
                    seq.prev_draft_tokens = []
                    continue

                row_sync_start = len(sync_input_ids)
                sync_input_ids.extend(accepted)
                sync_positions.extend(range(original_len - 1, original_len - 1 + M))
                for pos in range(original_len - 1, original_len - 1 + M):
                    sync_slot_mapping.append(self._slot_for_position(seq, pos))
                sync_fused_indices.extend(range(row["row_start"], row["row_start"] + M))
                row_sync_end = len(sync_input_ids)
                sync_rows.append({
                    "kind": "decode_verify",
                    "seq": seq,
                    "source_row": row,
                    "row_start": row_sync_start,
                    "row_end": row_sync_end,
                })
                sync_seqs.append(seq)
                sync_cu_q.append(sync_cu_q[-1] + M)
                sync_cu_k.append(sync_cu_k[-1] + original_len - 1 + M)
                max_sync_q = max(max_sync_q, M)
                max_sync_k = max(max_sync_k, original_len - 1 + M)

        sync_fused_hidden = None
        if sync_input_ids:
            use_full_fuse = (
                len(sync_fused_indices) == input_ids.size(0)
                and all(i == idx for idx, i in enumerate(sync_fused_indices))
            )
            sync_fused_hidden = self._fuse_captured_hidden(
                captured,
                None if use_full_fuse else sync_fused_indices,
            )

        if profile:
            torch.cuda.synchronize()
            _t4 = _time.perf_counter()

        if sync_input_ids:
            sync_input_ids_t = torch.tensor(sync_input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            sync_positions_t = torch.tensor(sync_positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            sync_slot_mapping_t = torch.tensor(sync_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            sync_cu_q_t = torch.tensor(sync_cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            sync_cu_k_t = torch.tensor(sync_cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            sync_block_tables = self.prepare_block_tables(sync_seqs) if sync_cu_k[-1] > sync_cu_q[-1] else None

            set_context(True, sync_cu_q_t, sync_cu_k_t, max_sync_q, max_sync_k,
                        sync_slot_mapping_t, block_tables=sync_block_tables,
                        num_prefill_tokens=len(sync_input_ids), num_decode_seqs=0,
                        finishing_prefill_indices=[])
            sync_logits, sync_hidden_out = self.draft_model(sync_input_ids_t, sync_positions_t, sync_fused_hidden)
            reset_context()

            finishing_prefill_infos = []
            active_decode_sync_rows = []
            for sync_row in sync_rows:
                source = sync_row["source_row"]
                if sync_row["kind"] == "prefill" and source["is_finishing_prefill"]:
                    finishing_prefill_infos.append((sync_row["seq"], sync_row["row_end"] - 1, source["sampled_token"]))
                elif sync_row["kind"] == "decode_verify":
                    active_decode_sync_rows.append(sync_row)

            if finishing_prefill_infos:
                self._propose_after_prefill(finishing_prefill_infos, sync_logits, sync_hidden_out)

            if active_decode_sync_rows:
                last_indices = torch.tensor(
                    [sync_row["row_end"] - 1 for sync_row in active_decode_sync_rows],
                    dtype=torch.int64,
                    device='cuda',
                )
                current_hidden = sync_hidden_out[last_indices]
                sync_draft_token = sync_logits[last_indices].argmax(dim=-1)
                first_draft_token = self.draft_model.d2t[sync_draft_token]
                active_decode_seqs = [sync_row["seq"] for sync_row in active_decode_sync_rows]
                source_rows = [sync_row["source_row"] for sync_row in active_decode_sync_rows]
                draft_loop_start = [row["original_len"] + row["num_accepted"] - 1 for row in source_rows]
                first_slots = [self._slot_for_position(seq, pos) for seq, pos in zip(active_decode_seqs, draft_loop_start)]
                rollback_num_blocks = [row["final_num_blocks"] for row in source_rows]

                draft_tokens_all = self._generate_draft_tokens_from_state(
                    seqs=active_decode_seqs,
                    current_input=first_draft_token,
                    current_hidden=current_hidden,
                    start_positions=draft_loop_start,
                    num_steps=K - 1,
                    initial_tokens=[first_draft_token],
                    first_slots=first_slots,
                    rollback_num_blocks=rollback_num_blocks,
                )
                self._assign_prev_draft_tokens(active_decode_seqs, draft_tokens_all)
        else:
            reset_context()

        if profile:
            torch.cuda.synchronize()
            _t5 = _time.perf_counter()

            if not hasattr(self, '_spec_timings'):
                self._spec_timings = {'build_batch': 0, 'target_fwd': 0, 'verify': 0, 'fuse_build_sync': 0, 'draft_all': 0, 'count': 0}
            t = self._spec_timings
            step_build_batch = _t1 - _t0
            step_target_fwd = _t2 - _t1
            step_verify = _t3 - _t2
            step_fuse_build_sync = _t4 - _t3
            step_draft_all = _t5 - _t4
            t['build_batch'] += step_build_batch
            t['target_fwd'] += step_target_fwd
            t['verify'] += step_verify
            t['fuse_build_sync'] += step_fuse_build_sync
            t['draft_all'] += step_draft_all
            t['count'] += 1
            if t['count'] % 10 == 0:
                n = t['count']
                print(f"  [SPEC TIMING] steps={n} mode={'graph' if use_spec_graph else 'eager'} | "
                      f"last build={step_build_batch*1000:.1f}ms target={step_target_fwd*1000:.1f}ms "
                      f"verify={step_verify*1000:.1f}ms fuse+sync={step_fuse_build_sync*1000:.1f}ms "
                      f"draft={step_draft_all*1000:.1f}ms total={(_t5-_t0)*1000:.1f}ms | "
                      f"avg build={t['build_batch']/n*1000:.1f}ms target={t['target_fwd']/n*1000:.1f}ms "
                      f"verify={t['verify']/n*1000:.1f}ms fuse+sync={t['fuse_build_sync']/n*1000:.1f}ms "
                      f"draft={t['draft_all']/n*1000:.1f}ms")

        if self.rank != 0:
            return None
        return {
            "prefill_seqs": prefill_seqs,
            "prefill_token_ids": prefill_token_ids,
            "decode_seqs": decode_seqs,
            "decode_accepted_tokens": decode_accepted_tokens,
        }

    def set_block_manager(self, block_manager: BlockManager):
        """供 engine 在初始化后设置 block_manager 引用"""
        self.block_manager = block_manager

    def reset_spec_profile_metrics(self):
        if hasattr(self, '_spec_timings'):
            del self._spec_timings

    @torch.inference_mode()
    def capture_spec_decode_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        q_len = self.num_spec_tokens + 1
        max_spec_bs = min(
            config.max_num_seqs,
            config.max_num_batched_tokens // q_len,
            512 // q_len,
        )
        if max_spec_bs < 1:
            self.spec_graphs = {}
            self.spec_graph_bs = []
            return

        graph_bs = [1, 2, 4, 8] + list(range(16, max_spec_bs + 1, 16))
        graph_bs.append(max_spec_bs)
        self.spec_graph_bs = sorted({bs for bs in graph_bs if 1 <= bs <= max_spec_bs})
        self.spec_graph_query_len = q_len
        self.spec_graph_max_seqlen_k = config.max_model_len + self.num_spec_tokens
        max_spec_tokens = self.spec_graph_bs[-1] * q_len
        max_num_blocks = (self.spec_graph_max_seqlen_k + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_spec_tokens, dtype=torch.int64, device='cuda')
        positions = torch.arange(max_spec_tokens, dtype=torch.int64, device='cuda')
        slot_mapping = torch.full((max_spec_tokens,), -1, dtype=torch.int32, device='cuda')
        cu_q = torch.arange(self.spec_graph_bs[-1] + 1, dtype=torch.int32, device='cuda') * q_len
        cu_k = torch.arange(self.spec_graph_bs[-1] + 1, dtype=torch.int32, device='cuda') * q_len
        block_tables = torch.zeros(self.spec_graph_bs[-1], max_num_blocks, dtype=torch.int32, device='cuda')
        outputs = torch.zeros(max_spec_tokens, hf_config.hidden_size, dtype=hf_config.torch_dtype, device='cuda')
        captured_outputs = {
            layer: torch.zeros(max_spec_tokens, hf_config.hidden_size, dtype=hf_config.torch_dtype, device='cuda')
            for layer in self.eagle3_fuse_layers
        }

        self.spec_graphs = {}
        for bs in reversed(self.spec_graph_bs):
            num_tokens = bs * q_len

            def forward_graph():
                hidden, captured = self.model(
                    input_ids[:num_tokens],
                    positions[:num_tokens],
                    capture_layers=self.eagle3_fuse_layers,
                )
                outputs[:num_tokens] = hidden
                for layer in self.eagle3_fuse_layers:
                    captured_outputs[layer][:num_tokens] = captured[layer]

            set_context(True, cu_q[:bs + 1], cu_k[:bs + 1], q_len, self.spec_graph_max_seqlen_k,
                        slot_mapping[:num_tokens], block_tables=block_tables[:bs],
                        num_prefill_tokens=num_tokens, num_decode_seqs=0,
                        finishing_prefill_indices=[])
            forward_graph()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                forward_graph()
            self.spec_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.spec_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_q=cu_q,
            cu_k=cu_k,
            block_tables=block_tables,
            outputs=outputs,
            captured=captured_outputs,
        )

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
