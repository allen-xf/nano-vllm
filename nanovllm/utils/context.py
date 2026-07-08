from dataclasses import dataclass
import torch


@dataclass
class Context:
    has_prefill: bool = False
    '''
    1. 为什么要用"累积长度"？
    在普通的深度学习中，如果一个 Batch 里的句子长度不一，我们会用 Padding（补零） 把它们凑成一样长。但这会浪费大量的显存和计算资源去处理那些"零"。
    为了提速，高性能框架会将 Batch 里的所有句子首尾相连地拼成一根长条（即 Flatten 操作）。这时候，我们就需要 cu_seqlens_q 来告诉模型：这根长条里，哪一段属于哪句话。
    而 q 代表 Query（在 Attention 机制中对应输入端）
    example
    cu_seqlens_q tensor([    0,  4096,  8192, 12288, 16384], device='cuda:0', dtype=torch.int32)
    cu_seqlens_k tensor([    0,  4096,  8192, 12288, 16384], device='cuda:0', dtype=torch.int32)
    max_seqlen_q 4096
    max_seqlen_k 4096
    '''
    cu_seqlens_q: torch.Tensor | None = None # prefill
    cu_seqlens_k: torch.Tensor | None = None # prefill
    max_seqlen_q: int = 0 # prefill
    max_seqlen_k: int = 0 # prefill
    p_slot_mapping: torch.Tensor | None = None # prefill
    decode_context_lens: torch.Tensor | None = None # decode
    block_tables: torch.Tensor | None = None # prefill or decode
    # chunked prefill: 混合 batch 支持
    num_prefill_tokens: int = 0
    num_decode_seqs: int = 0
    # chunked prefill: 完成 prefill 的 seq 在 prefill_seqs 中的下标（用于 LMHead 选取 logits）
    finishing_prefill_indices: list | None = None
    # decode 部分的字段（混合 batch 时同时存在）
    decode_slot_mapping: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    causal: bool = True


'''
在代码中，_CONTEXT = Context() 是在所有函数和类的外部定义的。
位置： 它位于模块级别（Module Level）。
效果： 当 Python 加载这个 .py 文件时，_CONTEXT 对象会被创建并存放在内存的一个固定位置
'''
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(has_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, p_slot_mapping=None, decode_context_lens=None, block_tables=None, **kwargs):
    global _CONTEXT
    _CONTEXT = Context(has_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, p_slot_mapping, decode_context_lens, block_tables, **kwargs)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
