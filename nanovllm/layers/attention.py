import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    '''
    idx = tl.program_id(0)    # 当前线程块处理第几个 token
    key_stride = key.stride(0) # = 1024，token 之间的步长
    D = 1024                   # num_heads × head_dim = 8 × 128

    key_offsets = idx * key_stride + tl.arange(0, D)
    #            ↑                   ↑
    #         起始偏移              [0, 1, 2, ..., 1023]

    以具体数字为例

    key 在显存中是连续排列的：

    显存地址:  0    1    2   ...  1023  1024  1025  ...  2047  2048  ...
                |←── token 0 ──→|  |←── token 1 ──→|  |←── token 2 ──→|

    idx=0（第 0 个 token）：
    0 * 1024 + [0, 1, 2, ..., 1023] = [0, 1, 2, ..., 1023]

    idx=1（第 1 个 token）：
    1 * 1024 + [0, 1, 2, ..., 1023] = [1024, 1025, 1026, ..., 2047]

    idx=2（第 2 个 token）：
    2 * 1024 + [0, 1, 2, ..., 1023] = [2048, 2049, 2050, ..., 3071]

    然后 tl.load(key_ptr + key_offsets) 一次性把这 1024 个元素从显存读出来。

    类比 Python

    # 等价于
    flat_key = key.view(N, -1)    # [N, 1024]
    flat_key[idx]                  # 取第 idx 行的 1024 个元素
    '''
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    '''
    slot = 3075    # 从 slot_mapping 查到的物理位置
    D = 1024
    cache_offsets = 3075 * 1024 + [0, 1, 2, ..., 1023]
        = [3148800, 3148801, ..., 3149823]
    然后 tl.store(k_cache_ptr + cache_offsets, key) 把 1024 个元素写到这个位置。
    '''
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

def store_kvcache_simplified(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    '''
    参数：
    - key: 当前步计算的key张量，形状为[N, num_heads, head_dim]
    - value: 当前步计算的value张量， 形状为[N, num_heads, head_dim]
    - k_cache: key缓存， 形状为[max_blocks, num_heads, head_dim]， 最细粒度是slot级别
    - v_cache: key缓存， 形状为[max_blocks, num_heads, head_dim]
    - slot_mapping: 每个token应该存在缓存中的哪个位置，形状为[N]
    '''
    N= key.shape

    # 展平 head和head_dim维度
    flat_key = key.view(N, -1)
    flat_value = value.view(N, -1)

    # 根据 slot_mapping 将数据存入缓存
    # 虽然 for 循环本身在 CPU 上跑，但 k_cache 和 flat_key 都是 CUDA tensor（在 GPU 显存里），所以每次 k_cache[slot] = flat_key[i] 实际上是 CPU 发起一次 GPU kernel 调用来完成数据拷贝。
    for i in range(N):
        slot =slot_mapping[i].item()
        k_cache[slot] = flat_key[i]
        v_cache[slot] = flat_value[i]

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def update_kv_cache(self, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor | None):
        if slot_mapping is None:
            return
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(key, value, self.k_cache, self.v_cache, slot_mapping)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # numel() 返回 tensor 中元素的总数，
        # warmup 阶段 cache 还没分配，跳过写入；正式推理时 cache 已分配，每次 forward 都会把新的 K/V 写进去。
        # 在 model_runner.py:123-128
        num_prefill_tokens = context.num_prefill_tokens
        num_decode_seqs = context.num_decode_seqs

        # 计算 Attention（分 prefill 和 decode 两条路径，chunked prefill 时可能同时存在）
        if context.has_prefill:
            # 第一步：处理 prefill 部分
            p_q, p_k, p_v = q[:num_prefill_tokens], k[:num_prefill_tokens], v[:num_prefill_tokens]

            # 写入 prefill 的 KV cache
            #  warmup 时 KV cache 还没分配，k_cache.numel() == 0，所以跳过写入。warmup结束后会accolcate kv cache
            if k_cache.numel() and v_cache.numel():
                store_kvcache(p_k, p_v, k_cache, v_cache, context.p_slot_mapping)

            if context.block_tables is not None:    # prefix cache or chunked prefill
                p_k, p_v = k_cache, v_cache
            p_o = flash_attn_varlen_func(p_q, p_k, p_v,
                                         max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                         max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                         softmax_scale=self.scale, causal=context.causal, block_table=context.block_tables)

            if num_decode_seqs > 0:
                # decode 部分
                d_q = q[num_prefill_tokens:]
                d_k = k[num_prefill_tokens:]
                d_v = v[num_prefill_tokens:]

                # 写入 decode 的 KV cache
                if k_cache.numel() and v_cache.numel():
                    store_kvcache(d_k, d_v, k_cache, v_cache, context.decode_slot_mapping)

                d_o = flash_attn_with_kvcache(d_q.unsqueeze(1), k_cache, v_cache,
                                              cache_seqlens=context.decode_context_lens, block_table=context.decode_block_tables,
                                              softmax_scale=self.scale, causal=context.causal).squeeze(1)
                o = torch.cat([p_o, d_o], dim=0)
            else:
                o = p_o
        else:
            # 纯 decode
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.decode_slot_mapping)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.decode_context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=context.causal)
        return o
