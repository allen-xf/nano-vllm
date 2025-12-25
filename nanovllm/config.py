import os
from dataclasses import dataclass
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
    hf_config: AutoConfig | None = None
    eos: int = -1 #end of sequence
    kvcache_block_size: int = 256 # Size of blocks for KV cache allocation 假如一个完成的sequence是1024个token，那么需要4个block
    num_kvcache_blocks: int = -1 # -1 根据显存动态计算

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
