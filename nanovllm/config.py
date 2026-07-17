import os
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import AutoConfig


@dataclass
class Config:
    model: str #pa
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512 #batch 里 seq size的最大值
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9 # 现在可以先理解为kv cache 能使用的GPU内存， kv cache是如何更新的
    tensor_parallel_size: int = 1 # tensor 并线计算的时候要用到多少张GPu
    enforce_eager: bool = False # torch的一种运行模式， eager 是指 即时模式， 每次torch 操作cpu都会调用一次gpu kernal，更慢一点 方便调试
    enable_chunked_prefill: bool = True #是否开启 chunked prefill，开启后 prefill/decode 可混合调度
    hf_config: Optional[AutoConfig] = None # AutoConfig 类型
    eos: int = -1 #end of sequence
    kvcache_block_size: int = 256 # Size of blocks for KV cache allocation 假如一个完成的sequence是1024个token，那么需要4个block
    num_kvcache_blocks: int = -1 # -1 根据显存动态计算
    # speculative decoding
    draft_model: Optional[str] = None # draft model 路径
    spec_method: str = "eagle3" # speculative backend: eagle3 / dflash / dspark
    num_spec_tokens: int = 5 # 每步 draft 生成的 token 数 K
    eagle3_fuse_layers: Optional[List[int]] = None # target model 中用于融合的层索引，None 时自动选取
    spec_profile: bool = False # 是否开启 spec 阶段同步计时
    spec_debug: bool = False # 是否开启 EAGLE3 debug 打印

    # __post_init__ 则是在自动生成的 __init__ 执行完毕后，被立即调用的方法。
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 应该就是这个文件的内容 /root/llm/model/Qwen3-0.6B/config.json
        self.hf_config = AutoConfig.from_pretrained(self.model) # 根据指定的模型标识符，从云端（或本地缓存）下载并加载该模型的元数据配置文件
        self.hf_config.save_pretrained('/root/llm/nano-vllm/')
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # speculative draft model config
        if self.draft_model:
            assert self.spec_method in {"eagle3", "dflash", "dspark"}, "spec_method must be 'eagle3', 'dflash', or 'dspark'"
            assert os.path.isdir(self.draft_model)
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)

            normalized_dflash_config = dict(getattr(self.draft_hf_config, "dflash_config", None) or {})
            for key in ("mask_token_id", "target_layer_ids", "layer_ids", "use_aux_hidden_state"):
                value = getattr(self.draft_hf_config, key, None)
                if value is not None:
                    normalized_dflash_config[key] = value
            if normalized_dflash_config:
                self.draft_hf_config.dflash_config = normalized_dflash_config

            if self.spec_method == "eagle3":
                # 自动选取融合层：EAGLE3 paper default (layer 2, middle, third-from-last)
                if self.eagle3_fuse_layers is None:
                    # 优先使用 draft model config 中指定的层索引
                    eagle_config = getattr(self.draft_hf_config, 'eagle_config', None)
                    if eagle_config and 'eagle_aux_hidden_state_layer_ids' in eagle_config:
                        self.eagle3_fuse_layers = eagle_config['eagle_aux_hidden_state_layer_ids']
                    else:
                        n = self.hf_config.num_hidden_layers
                        self.eagle3_fuse_layers = [2, n // 2, n - 3]
            else:
                method_name = self.spec_method.upper()
                assert getattr(self.hf_config, "model_type", None) == "qwen3", f"{method_name} currently only supports Qwen3 target models"
                dflash_config = normalized_dflash_config
                assert dflash_config, f"{method_name} draft model requires dflash_config"
                assert dflash_config.get("mask_token_id") is not None, f"{method_name} requires dflash_config.mask_token_id"
                assert (
                    dflash_config.get("target_layer_ids") is not None
                    or dflash_config.get("layer_ids") is not None
                ), f"{method_name} requires dflash_config.target_layer_ids or layer_ids"
                if self.spec_method == "dspark":
                    assert getattr(self.draft_hf_config, "markov_rank", None) is not None, "DSpark requires markov_rank"
                assert not dflash_config.get("use_swa", False), f"{method_name} SWA is not supported yet"
                layer_types = getattr(self.draft_hf_config, "layer_types", None)
                if layer_types is not None:
                    has_sliding = any(layer_type == "sliding_attention" for layer_type in layer_types)
                    has_full = any(layer_type != "sliding_attention" for layer_type in layer_types)
                    assert not has_sliding, f"{method_name} sliding attention is not supported yet"
                    assert not (has_sliding and has_full), f"{method_name} mixed sliding/full attention is not supported yet"
