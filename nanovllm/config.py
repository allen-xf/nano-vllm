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
    hf_config: AutoConfig | None = None # AutoConfig 类型
    eos: int = -1 #end of sequence
    kvcache_block_size: int = 256 # Size of blocks for KV cache allocation 假如一个完成的sequence是1024个token，那么需要4个block
    num_kvcache_blocks: int = -1 # -1 根据显存动态计算

    # __post_init__ 则是在自动生成的 __init__ 执行完毕后，被立即调用的方法。
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 应该就是这个文件的内容 /root/llm/model/Qwen3-0.6B/config.json
        self.hf_config = AutoConfig.from_pretrained(self.model) # 根据指定的模型标识符，从云端（或本地缓存）下载并加载该模型的元数据配置文件
        self.hf_config.save_pretrained('/root/llm/nano-vllm/')
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
