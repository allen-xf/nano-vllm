# DFlash 实现文档

## 1. 背景和目标

当前 nano-vLLM 已有 EAGLE3 speculative decoding，但实现和 EAGLE3 强绑定：`ModelRunner` 同时负责 target verify、draft sync、serial proposal、accept/reject、CUDA graph。DFlash 的 proposal 机制和 EAGLE3 不同，不能直接在现有 `_generate_draft_tokens_from_state()` 上打补丁。

本文参考 vLLM 的 Qwen3 DFlash 实现，说明 DFlash 和 EAGLE3 的差异、nano-vLLM 缺失的基础能力，以及后续落地顺序。

本阶段实现范围只覆盖 **Qwen3 DFlash**：

- target model 只考虑 nano-vLLM 当前已有的 `Qwen3ForCausalLM`。
- draft model 只实现 Qwen3 DFlash 结构，例如 `Qwen3DFlashForCausalLM` / `Qwen3DFlashModel`。
- 不做 Laguna、Gemma、MiMo 等其它 DFlash 架构适配。
- 不为了兼容其它架构抽象过度复杂的 model registry；先把 Qwen3 路径跑正确。

主要参考文件：

- `/Users/xiao-jy/repos/vllm/vllm/model_executor/models/qwen3_dflash.py`
- `/Users/xiao-jy/repos/vllm/vllm/v1/spec_decode/dflash.py`
- `/Users/xiao-jy/repos/vllm/vllm/v1/spec_decode/utils.py`
- `/Users/xiao-jy/repos/vllm/vllm/v1/spec_decode/llm_base_proposer.py`

## 2. EAGLE3 和 DFlash 的核心区别

### 2.1 EAGLE3：serial draft rollout

当前 nano-vLLM EAGLE3 路径是：

```text
target forward 捕获多层 hidden states
        ↓
fc 融合 hidden states
        ↓
draft model sync 到 accepted context
        ↓
从 sync logits 采样第一个 draft token
        ↓
serial proposal: d0 -> d1 -> ... -> dK-1
        ↓
把 K 个 draft token 保存到 seq.prev_draft_tokens
```

特点：

- draft model 的输入是 `input_ids + fused_hidden`。
- proposal 是自回归串行生成，K 个草稿 token 需要 K 次左右的 draft 状态推进。
- `seq.prev_draft_tokens` 保存下一轮 target verify 要验证的 K 个 target-vocab token。
- target verify row 是 `[last_token] + prev_draft_tokens`，目标模型一次 forward 验证 K 个 draft token，并给出 bonus/correction token。

### 2.2 DFlash：context KV precompute + query-only parallel proposal

vLLM DFlash 路径是：

```text
target forward 产出/捕获 hidden states
        ↓
select / fuse target hidden states
        ↓
把 fused hidden states 投影成 draft model 每一层的 context K/V
        ↓
直接写入 draft KV cache
        ↓
构造 query: [next_token] + [mask_token] * K
        ↓
draft model query-only forward
        ↓
只从后 K 个 mask query rows 采样 draft tokens
```

特点：

- target hidden states 不再作为每一步 draft 的 recurrent hidden 输入，而是变成 draft attention 的 context K/V。
- proposal 是 parallel drafting，一次 query forward 生成 K 个草稿 token。
- query token 形状是 `[bonus_token, mask, mask, ...]`。
- DFlash 可能需要 non-causal query attention；nano-vLLM 当前 attention 基本写死 `causal=True`。
- DFlash 使用 `mask_token_id`，部分 checkpoint 还会带单独的 `mask_embedding.pt`。

### 2.3 Qwen3 DFlash 模型结构

Qwen3-only DFlash 可以拆成 target model、hidden fusion、draft context-KV precompute、query-only draft forward 四个部分：

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Target Qwen3ForCausalLM                                               │
│                                                                      │
│ input_ids / positions                                                │
│      │                                                               │
│      ▼                                                               │
│ Qwen3Model layers                                                     │
│      │                                                               │
│      ├─ capture target hidden states from target_layer_ids            │
│      │     h_l0, h_l1, ..., h_ln                                      │
│      │                                                               │
│      └─ target logits for verify / correction                         │
└──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Hidden fusion                                                         │
│                                                                      │
│ concat(h_l0, h_l1, ..., h_ln) -> fc -> context_states                 │
└──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Qwen3DFlashModel: precompute context K/V                              │
│                                                                      │
│ context_states                                                        │
│   -> hidden_norm                                                      │
│   -> per-layer KV projection                                          │
│   -> K norm + RoPE                                                    │
│   -> write layer_i.k_cache / layer_i.v_cache                          │
└──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Qwen3DFlashModel: query-only forward                                  │
│                                                                      │
│ query input_ids = [next_token, mask, mask, ..., mask]                 │
│      │                                                               │
│      ▼                                                               │
│ embed_tokens / optional mask_embedding                                │
│      │                                                               │
│      ▼                                                               │
│ DFlash decoder layers attend to pre-inserted context K/V              │
│      │                                                               │
│      ▼                                                               │
│ hidden states for query rows                                          │
│      │                                                               │
│      ▼                                                               │
│ lm_head / d2t -> sample only mask rows -> K draft tokens              │
└──────────────────────────────────────────────────────────────────────┘
```

每个 DFlash decoder layer 仍然是 Qwen3 风格的 decoder block，但它处理的是 query tokens，context 已经提前写入 KV cache：

```text
query hidden
   │
   ▼
input_layernorm
   │
   ▼
DFlashQwen3Attention
   ├─ 当前 query tokens: qkv_proj -> Q/K/V
   ├─ query Q/K: q_norm / k_norm + RoPE
   └─ attention over pre-inserted context K/V + query K/V
   │
   ▼
post_attention_layernorm
   │
   ▼
MLP
   │
   ▼
next hidden
```

注意：DFlash 的“模型结构”不是把 target hidden states 当作普通 token 送进 draft layers，而是把它们先变成 draft layers 的历史 KV cache。真正进入 draft forward 的 input_ids 只有 query：`[next_token] + K 个 mask_token`。

## 3. vLLM DFlash 数据流

### 3.1 配置 attention 模式

vLLM 在 `qwen3_dflash.py` 的 `_resolve_layer_attention()` 中解析 DFlash 每层 attention 行为。nano-vLLM 第一版只支持 Qwen3，因此只需要按 Qwen3 DFlash config 处理，不需要照搬 vLLM 为多模型生态准备的所有分支：

- `dflash_config.causal` 可以全局指定 causal / non-causal。
- `dflash_config.use_swa` 可以开启 sliding window attention。
- `dflash_config.swa_window_size` 或顶层 `sliding_window` 提供窗口大小。
- 混合 sliding/full attention 当前未完整支持。

DFlash 默认 full attention layer 是 non-causal，SWA layer 默认 causal。Qwen3-only 版本不要假设一定有 SWA：如果目标 checkpoint 的 `dflash_config.use_swa` 或 `layer_types` 指明 sliding attention，再进入 SWA 分支；否则按 full-attention DFlash 实现。

### 3.2 DFlash draft attention

vLLM 的 `DFlashQwen3Attention` 假设 context K/V 已经提前写入 KV cache。它的 forward 只处理 query tokens：

```text
query embeddings -> qkv_proj -> q/k/v -> q_norm/k_norm -> RoPE -> attention(query over cached context+query KV)
```

这里的 context K/V 不来自当前 forward 的 `input_ids`，而是由 `precompute_and_store_context_kv()` 预先写入。

### 3.3 context K/V 预计算

vLLM `DFlashQwen3Model.precompute_and_store_context_kv()` 做以下事情：

```text
context_states
  -> hidden_norm
  -> fused KV projection for all draft layers
  -> reshape to [2, num_layers, num_context, num_kv_heads, head_dim]
  -> grouped K RMSNorm
  -> RoPE on K
  -> per-layer KV cache update
```

核心点：

- 一次大 GEMM 同时算所有 draft layers 的 K/V。
- K norm 和 RoPE 也是按 layer-major batch 执行。
- 最终逐层调用 attention backend 的 KV cache update，把 context K/V 写入每层 draft KV cache。

### 3.4 每层 KV inject 的含义

DFlash 的 per-layer KV inject 是“流程相同、数值不同”：

```text
for each draft layer i:
    context_states
      -> layer_i 的 KV projection 得到 K_i / V_i
      -> layer_i 的 k_norm 处理 K_i
      -> 对 K_i 做 RoPE
      -> 写入 layer_i 自己的 KV cache
```

也就是说，每一层都执行同样的注入流程，但不会把同一份 K/V 复制到所有层。不同层有不同的 `qkv_proj` 权重、`k_norm` 权重和 KV cache，因此最终写入的是不同的 `K_i / V_i`：

```text
context_states
   │
   ├─ layer0.kv_proj -> K_0 / V_0 -> layer0.k_cache / layer0.v_cache
   │
   ├─ layer1.kv_proj -> K_1 / V_1 -> layer1.k_cache / layer1.v_cache
   │
   ├─ layer2.kv_proj -> K_2 / V_2 -> layer2.k_cache / layer2.v_cache
   │
   └─ ...
```

对 Qwen3 full-attention DFlash 来说，通常每层使用相同的 `context_positions` 和 `context_slot_mapping`；区别在于每层投影权重不同、K norm 权重不同、写入的目标 cache 不同。vLLM 为了减少 Python loop 和小 GEMM，会把所有层的 KV projection weight 拼起来，一次算出：

```text
all_k: [num_layers, num_context, num_kv_heads, head_dim]
all_v: [num_layers, num_context, num_kv_heads, head_dim]
```

然后再逐层写入：

```text
all_k[i], all_v[i] -> draft_layer_i KV cache
```

nano-vLLM 第一版可以先不用 fused projection，直接按层循环实现，确认 correctness 后再优化成 vLLM 的 fused 版本。

### 3.5 DFlash proposer 输入构造

vLLM `DFlashProposer.set_inputs_first_pass()` 构造两类输入：

1. context：target tokens / positions / hidden states，用来预计算 context K/V。
2. query：`[next_token] + [mask_token] * K`，用来并行生成 K 个 draft tokens。

`copy_and_expand_dflash_inputs_kernel()` 对每个 request 做：

```text
1. copy context positions
2. compute query positions
3. write query input_ids = [next_token, mask, mask, ...]
4. compute context slot_mapping and query slot_mapping
5. record token_indices_to_sample for mask rows
```

第一版 nano-vLLM 不需要直接写 Triton kernel，可以先用 Python 构造这些张量，确认 token/position/slot 对齐正确后再优化。

## 4. DFlash 所需配置和权重

### 4.1 必需或重要配置

DFlash draft model config 需要读取：

- `dflash_config.mask_token_id`：mask query 使用的 token id。
- `dflash_config.target_layer_ids`：从 target model 捕获哪些层的 hidden states。
- `dflash_config.use_aux_hidden_state`：是否使用 target hidden fusion，默认 true。
- `dflash_config.causal`：DFlash query attention 是否 causal。
- `dflash_config.use_swa`：是否启用 sliding window attention。
- `dflash_config.swa_window_size`：SWA 窗口大小。
- `dflash_config.attention_sink_bias`：部分 SWA checkpoint 需要。
- `draft_vocab_size`：draft vocab 大小；缺省可等于 target vocab。
- `target_hidden_size`：当 target hidden size 和 draft hidden size 不同时使用。

### 4.2 权重加载注意点

vLLM DFlash 支持：

- `q_proj/k_proj/v_proj` 打包到 `qkv_proj`。
- `gate_proj/up_proj` 打包到 `gate_up_proj`。
- `d2t` 映射到 `draft_id_to_target_id`。
- 如果 checkpoint 不包含 embed tokens，可以跳过 draft embedding 加载。
- 如果存在 `mask_embedding.pt`，用它替换 mask token embedding。
- DFlash 不应该包含 `mask_hidden`，vLLM 会 assert 拒绝。

nano-vLLM 当前 loader 能处理 packed modules 和 buffer，但 DFlash 还需要补充 `mask_embedding.pt`、`d2t`、可选 embedding/lm_head 等特殊逻辑。

## 5. nano-vLLM 当前缺失的基础能力

### 5.1 DFlash draft model

当前只有 `nanovllm/models/eagle3.py`，它是 EAGLE3 风格：

- `fc` 融合 target hidden states。
- 单层 modified decoder。
- forward 输入 `input_ids, positions, fused_hidden`。
- serial proposal 使用上一步 hidden state。

DFlash 需要新的 Qwen3-only 模型类，例如：

- `nanovllm/models/qwen3_dflash.py`

第一版只实现 Qwen3 DFlash；其它架构不要放进同一个模型类里兼容。

核心 API：

```python
class DFlashQwen3ForCausalLM(nn.Module):
    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
```

### 5.2 直接写 KV cache 的 API

nano-vLLM 当前 attention 只在 normal forward 中调用 `store_kvcache()`。DFlash 需要独立的 context K/V 写入能力：

```python
def update_kv_cache(
    self,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None: ...
```

第一版可以复用已有 `store_kvcache()` kernel，先按 layer 循环写入；后续再做 vLLM 那种 fused KV projection / grouped norm / fused RoPE 优化。

### 5.3 non-causal attention

当前 `nanovllm/layers/attention.py` 中 `flash_attn_varlen_func()` 和 `flash_attn_with_kvcache()` 基本都使用 `causal=True`。

DFlash checkpoint 如果要求 non-causal，需要把 causality 做成 context 或 attention 层参数：

```python
set_context(..., causal=dflash_causal)
```

然后 attention backend 使用：

```python
causal=context.causal
```

第一阶段可以先只支持 causal DFlash 或显式报错，等确认 checkpoint 配置后再支持 non-causal。

### 5.4 DFlash input packing

DFlash proposal 需要同时构造：

- context positions
- context slot mapping
- query input ids：`[next_token] + [mask_token] * K`
- query positions
- query slot mapping
- sample indices：只采样后 K 个 mask rows

EAGLE3 的 proposal slots 是 serial draft forward 使用；DFlash 的 query slots 是一次性 `K + 1` 行 query forward 使用。这部分应该放在 `DFlashSpecBackend`，不要污染 `ModelRunner`。

## 6. 建议的 backend 抽象

先把现有 EAGLE3 拆成 backend，后续 DFlash 作为第二个 backend。

```python
class Eagle3SpecBackend:
    def run_step(self, prefill_seqs, decode_seqs): ...
    def capture_decode_cudagraph(self): ...
    def reset_profile_metrics(self): ...
```

后续 DFlash：

```python
class DFlashSpecBackend:
    def run_step(self, prefill_seqs, decode_seqs): ...
    def build_query_batch(...): ...
    def precompute_context_kv(...): ...
    def propose(...): ...
```

公共部分应该保留在 target verify / accept-reject 层：

```text
build target verify batch
run target forward
select logits
verify / accept
```

backend-specific 部分：

```text
EAGLE3: accepted context draft sync + serial proposal
DFlash: accepted context KV precompute + query-only parallel proposal
```

## 7. DFlash 落地顺序

建议分阶段做：

1. **EAGLE3 backend refactor**
   - 把当前 EAGLE3 逻辑从 `ModelRunner` 抽到 `Eagle3SpecBackend`。
   - 保持 `seq.prev_draft_tokens`、`K + 1` target verify、rollback 语义不变。

2. **Config seam**
   - 增加 `spec_method`，当前只支持 `eagle3`。
   - 后续支持 `dflash` 时读取 `dflash_config`。
   - 第一版 `dflash` 只允许 Qwen3 target/draft config；其它架构直接显式报错。

3. **Attention / KV primitive**
   - 给 attention 增加直接写 KV cache 的方法。
   - 根据 checkpoint 需求决定是否实现 non-causal。

4. **DFlash model eager path**
   - 新增 `Qwen3DFlash` draft model。
   - 先用 Python loop 实现 per-layer context K/V precompute 和写入。
   - 暂不做 fused GEMM/Triton input packing。

5. **DFlash pure decode**
   - 先只支持纯 decode request。
   - 复用 target verify / accept-reject。
   - proposal 阶段构造 `[next_token] + mask * K`。

6. **finishing prefill**
   - prefill 完成后立即用 sampled token 作为 DFlash bonus token，生成下一轮 draft tokens。

7. **mixed chunked prefill + decode**
   - 处理 prefill rows 和 decode rows 混合时的 context hidden、slot mapping、rollback。

8. **性能优化**
   - CUDA graph for fixed query shape。
   - GPU/Triton input preparation。
   - fused context K/V projection。

## 8. 第一版运行时实现方案

本轮先实现 **Qwen3-only / eager-only / correctness-first** 的 DFlash 路径。目标是让 `spec_method="dflash"` 能跑通基本 prefill/decode，并尽量复用现有 EAGLE3 refactor 后的 target verify / accept-reject 合约。

### 8.1 需要新增或修改的文件

```text
nanovllm/models/qwen3_dflash.py              # Qwen3 DFlash draft model
nanovllm/engine/spec_decode/dflash.py        # DFlashSpecBackend
nanovllm/layers/attention.py                 # 直接写 KV cache 的 update_kv_cache API
nanovllm/engine/model_runner.py              # backend dispatch + multi-layer draft KV cache
nanovllm/config.py                           # spec_method=dflash 配置校验
nanovllm/engine/spec_decode/__init__.py      # 导出 DFlashSpecBackend
```

### 8.2 配置约束

`Config.__post_init__()` 中允许：

```python
spec_method in {"eagle3", "dflash"}
```

`eagle3` 分支保持现有行为不变。`dflash` 分支第一版只接受 Qwen3 target / Qwen3 DFlash draft，并读取：

- `dflash_config.mask_token_id`
- `dflash_config.target_layer_ids` 或 `dflash_config.layer_ids`
- `dflash_config.use_aux_hidden_state`，默认 true
- `draft_vocab_size`，缺省等于 `vocab_size`

第一版建议显式拒绝以下 checkpoint，而不是静默跑错：

- `dflash_config.use_swa=True`
- mixed `layer_types`

`dflash_config.causal` 通过 `set_context(..., causal=...)` 传给 attention backend；full-attention DFlash 可以按 checkpoint 配置跑 causal 或 non-causal。

### 8.3 KV cache 和 attention primitive

DFlash draft model 有多层 decoder，因此不能继续假设 draft KV cache 只有 1 层。`ModelRunner.allocate_kv_cache()` 需要按 backend 计算 draft layer 数：

```python
if spec_method == "eagle3":
    draft_num_layers = 1
elif spec_method == "dflash":
    draft_num_layers = draft_hf_config.num_hidden_layers
```

然后分配：

```python
draft_kv_cache: [2, draft_num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

并按 draft attention module 遍历顺序绑定每层 cache。

`Attention` 增加直接写 KV cache 的 API：

```python
def update_kv_cache(self, key, value, slot_mapping):
    store_kvcache(key, value, self.k_cache, self.v_cache, slot_mapping)
```

DFlash 的 `precompute_and_store_context_kv()` 用它把 target hidden 投影后的 context K/V 写进每个 draft layer 的 KV cache。

### 8.4 `Qwen3DFlashForCausalLM` 第一版结构

新增 `nanovllm/models/qwen3_dflash.py`，只实现 Qwen3 路径：

```python
class DFlashQwen3ForCausalLM(nn.Module):
    def combine_hidden_states(self, hidden_states): ...
    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping): ...
    def forward(self, input_ids, positions): ...
    def compute_logits(self, hidden_states): ...
```

内部结构：

- `DFlashQwen3Attention`：参考 `Qwen3Attention`，但始终有 per-head `q_norm/k_norm`。
- `DFlashQwen3DecoderLayer`：复用 Qwen3 decoder 的 residual / norm / MLP 流程。
- `DFlashQwen3Model`：包含 `embed_tokens`、`layers`、`hidden_norm`、`norm`、可选 `fc`。
- `DFlashQwen3ForCausalLM`：包含 `lm_head` 和可选 `d2t` draft-to-target vocab 映射。

`precompute_and_store_context_kv()` 第一版用 Python loop：

```text
normed_context = hidden_norm(context_states)
for each draft layer i:
    qkv = layer.self_attn.qkv_proj(normed_context)
    discard Q, keep K/V
    K -> k_norm -> RoPE(context_positions)
    layer.self_attn.attn.update_kv_cache(K, V, context_slot_mapping)
```

暂不做 vLLM 的 fused all-layer projection。

### 8.5 `DFlashSpecBackend` 数据流

新增 `nanovllm/engine/spec_decode/dflash.py`，对外保持和 EAGLE3 backend 相同的接口：

```python
class DFlashSpecBackend:
    def run_step(self, prefill_seqs, decode_seqs): ...
    def capture_decode_cudagraph(self): ...
    def reset_profile_metrics(self): ...
```

`run_step()` 分成四段：

1. **target verify batch**
   - prefill rows：当前 prefill chunk。
   - decode rows：继续使用 `[last_token] + prev_draft_tokens`。
   - target verify 成本仍然是 `K + 1`。

2. **target forward + accept/reject**
   - `Qwen3ForCausalLM.forward(..., capture_layers=target_layer_ids)`。
   - finishing prefill 从最后一行采样。
   - decode rows 用 greedy verify：draft 命中则接受，miss 则追加 target correction，全命中则追加 bonus token。

3. **context K/V precompute**
   - prefill chunk 的 captured hidden 全部写入 draft KV cache，保证 chunked prefill 能累积 draft context。
   - decode row 只把这次 target forward 中真实存在 hidden state 的 token 写入 context K/V。

4. **query-only proposal**
   - 对每个需要继续生成的 seq 构造：`[bonus_token] + [mask_token_id] * K`。
   - 只采样后 K 个 mask rows。
   - 采样结果写回 `seq.prev_draft_tokens`，保持 scheduler 合约不变。
   - query rows 使用 `append_n_slots(seq, K + 1, start_pos=...)` 临时占位，proposal 后用 `rollback_blocks()` 回滚。

### 8.6 rejection 的关键正确性规则

decode verify 中如果第 j 个 draft 被拒绝，target correction token 是 target logits 采样/argmax 出来的，但这轮 target forward **没有计算 correction token 自己的 hidden state**。

因此第一版 DFlash 不能把 correction token 当成 context K/V 写入 draft cache。正确做法是：

```text
accepted/matched prefix 的 target hidden -> precompute context K/V
correction token -> 下一次 DFlash query 的 bonus_token
query = [correction_token, mask, mask, ...]
```

如果 K 个 draft 全部命中，则 bonus token 同样作为下一次 DFlash query 的第一个 token。

### 8.7 第一版限制

- DFlash 只支持 Qwen3。
- DFlash 只跑 eager path。
- 暂不支持 SWA / mixed sliding-full attention。
- full-attention DFlash 支持按 `dflash_config.causal` 切换 causal / non-causal。
- 暂不实现 CUDA graph。
- 暂不实现 Triton input packing。
- 暂不实现 fused all-layer KV projection。
- 暂不把 `seq.prev_draft_tokens` 替换成通用 `spec_state`。

## 9. 验证矩阵

### 9.1 EAGLE3 refactor 后必须通过

- 非 spec 生成行为不变。
- EAGLE3 greedy 输出和 refactor 前一致。
- chunked prefill + EAGLE3 可正常运行。
- decode-only CUDA graph 路径仍可用。
- `seq.prev_draft_tokens` 生命周期不变。

### 9.2 DFlash 后续验证

- greedy spec 输出和 baseline 一致。
- pure decode DFlash 正确。
- finishing prefill 后能立即产生下一轮 draft。
- mixed chunked prefill + decode 正确。
- `d2t` draft vocab 映射正确。
- `mask_embedding.pt` 存在/不存在两种情况都正确。
- causal / non-causal checkpoint 行为明确：支持或显式报错。

## 10. 第一版不做的事情

- 不直接移植 vLLM Triton input packing kernel。
- 不一开始实现 fused all-layer KV projection。
- 不在 EAGLE3 的 serial proposal 中硬塞 DFlash 逻辑。
- 不提前删除 `seq.prev_draft_tokens`。
- 不把 DFlash 和 EAGLE3 模型内部强行统一。
