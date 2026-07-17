# DSpark 实现设计文档

## 1. 背景与目标

本文基于以下三类材料给出 **nano-vllm 落地 DSpark 的实现设计**：

- 论文：`/Users/xiao-jy/Documents/thesis/DSpark/DSpark.pdf`
- vLLM 模型实现：`/Users/xiao-jy/repos/vllm/vllm/model_executor/models/qwen3_dspark.py`
- vLLM runtime 实现：`/Users/xiao-jy/repos/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`

同时结合 nano-vllm 当前已有能力：

- chunked prefill
- EAGLE3 speculative decoding
- DFlash speculative backend
- 多层 draft KV cache 与 query-only DFlash 模型

目标不是“抽象出一个泛化 speculative 框架”，而是 **先在 nano-vllm 中把 Qwen3 DSpark 路径设计正确**，并明确：

1. 什么可以直接复用当前 DFlash/EAGLE3 基础设施；
2. 什么是 DSpark 独有的增量；
3. 什么是论文提出但 vLLM 开源实现尚未接入的能力；
4. 建议按什么顺序落地，既能尽快跑通，又不把后续 paper-complete 版本堵死。

---

## 2. 先给结论：DSpark 在工程上等于什么

### 2.1 算法本质

DSpark 不是一个完全独立于 DFlash 的新体系，而是：

- **并行 backbone**：继续沿用 DFlash 的 “context-KV precompute + query-only forward” 路径；
- **顺序依赖注入**：在并行 backbone 产出的每个 draft 位置 base logits 上，再叠加一个轻量的 sequential head（论文默认 Markov head）；
- **可选的 confidence-scheduled verification**：根据每个 draft 位置的 prefix survival probability，动态裁剪 target verify 长度。

换句话说：

```text
DSpark = DFlash-style parallel drafting
       + lightweight sequential head
       + (optional) confidence-scheduled verification
```

### 2.2 论文版 DSpark 与 vLLM 开源版 DSpark 的差异

这是本次设计里最关键的观察。

**论文中的 DSpark** 包含两部分：

1. **Semi-autoregressive generation**（Section 3.1, p.5-6）
   - 并行 backbone
   - Markov/RNN sequential head
2. **Confidence-scheduled verification**（Section 3.2, p.6-8）
   - confidence head
   - calibration（STS）
   - hardware-aware prefix scheduler

**vLLM 开源实现当前只实现了第 1 部分**，即：

- `qwen3_dspark.py` 实现了 `DSparkMarkovHead`
- `dspark/speculator.py` 实现了 sequential Markov sampling
- 但 `qwen3_dspark.py` 在 load weights 时显式跳过了 `confidence_head` 权重
- runtime 也没有动态裁剪 verify prefix 的逻辑

因此，若目标是：

- **尽快与 vLLM 当前可运行路径对齐**：实现 fixed-length DSpark 即可；
- **完整对齐论文**：还需要额外实现 confidence head + calibration + scheduler。

### 2.3 对 nano-vllm 的直接启示

对于 nano-vllm，最合理的路线是分两阶段：

- **Phase 1：实现 vLLM-parity DSpark**
  - Qwen3-only
  - 复用 DFlash backbone
  - 加入 Markov head 与 sequential sampling
  - verify 长度仍固定为 `K`
- **Phase 2：实现 paper-complete DSpark**
  - confidence head
  - prefix survival / calibration
  - dynamic verification scheduler

这样收益最大：

- Phase 1 改动小，能快速复用现有 DFlash backend；
- Phase 2 再把 scheduler 改造成论文所说的“verify smarter, not longer”。

---

## 3. 论文、vLLM、nano-vllm 三者的能力对照

| 能力 | 论文 | vLLM 开源实现 | nano-vllm 当前 | 建议 |
|---|---|---|---|---|
| DFlash-style 并行 backbone | 有 | 有 | 有 | 直接复用 |
| Markov head | 有 | 有 | 无 | Phase 1 新增 |
| RNN head | 有（备选） | 无 inference 支持 | 无 | 暂不做 |
| confidence head | 有 | 权重被跳过 | 无 | Phase 2 新增 |
| STS calibration | 有 | 无 | 无 | Phase 2 新增 |
| hardware-aware prefix scheduler | 有 | 无 | 无 | Phase 2 新增 |
| anchor-as-first query layout | 有（论文默认） | 有 | 无 | Phase 1 支持 |
| bonus-anchor 兼容模式 | 非论文默认 | 有（`dspark_bonus_anchor=True`） | 无 | Phase 1 支持 |
| non-causal draft attention | 需要 | 强制启用 | nano-vllm 已有 `causal` seam | DSpark 默认 false |

---

## 4. DSpark 为何值得单独实现

论文给出的几个工程上重要的结论，直接影响实现策略：

### 4.1 DSpark 的核心收益来自“少量顺序建模”

论文 Figure 2（p.12）指出：

- DFlash 在 draft block 第 1 个位置通常很强；
- 但后续位置会因独立并行生成而快速 suffix decay；
- DSpark 用轻量顺序头缓解这一问题，使条件接受率在整个 block 上更稳定。

### 4.2 默认应优先实现 Markov head，而不是 RNN head

Figure 3 / Figure 4（p.13-14）给出两个非常强的工程结论：

- 浅层 DSpark 已经可以打过更深的 DFlash；
- RNN head 相比 Markov head 只有边际提升，但实现更复杂、部署更差。

因此在 nano-vllm 中：

- **第一版只做 Markov head 是正确策略**；
- 不建议一开始就做 RNN head。

### 4.3 confidence scheduler 是“系统优化”，不是“模型结构优化”

论文 Section 3.2 / 4.3.3 / 5.2 强调：

- confidence head 的目的不是提高 draft token 本身质量；
- 而是用来避免 target model 去验证那些大概率会被拒绝的 trailing suffix；
- 尤其在高并发下，这部分浪费会显著吃掉批容量。

这说明在 nano-vllm 中：

- Markov head 和 confidence scheduler 应该解耦设计；
- 不要把 scheduler 逻辑塞进 draft model 内部；
- scheduler 决策应留在 engine/scheduler/backend 层。

---

## 5. nano-vllm 当前基础，哪些可复用

### 5.1 最重要的结论：DSpark 应该构建在现有 DFlash backend 之上

nano-vllm 当前的 `nanovllm/engine/spec_decode/dflash.py` 已经具备 DSpark 最核心的基础设施：

1. target hidden 捕获
2. 将 target hidden 融合成 draft hidden
3. precompute context KV 并直接写入 draft KV cache
4. query-only draft forward
5. 完成 prefill / decode verify 后，为下一轮生成 `seq.prev_draft_tokens`

而 DSpark 相比 DFlash，真正变化的是：

- query layout 可以从 DFlash 的 `1 + K` 变成 `K`；
- 采样不再是对所有 query rows 独立 argmax，而是 left-to-right sequential Markov sampling；
- （Phase 2）verify 长度不再固定是 `K`。

所以最佳实现方式不是“从 EAGLE3 重新做一套”，而是：

```text
DFlashSpecBackend 负责：
- target verify batch
- context KV precompute
- query-only forward
- rollback / block lifecycle

DSparkSpecBackend 在其上新增：
- DSpark query packing mode
- Markov sequential sampling
- (optional) confidence outputs + dynamic verify length
```

### 5.2 当前 repo 中可以直接复用的模块

#### 1）draft KV cache 与 query-only forward 基础设施

- `nanovllm/models/qwen3_dflash.py`
- `nanovllm/layers/attention.py`
- `nanovllm/utils/context.py`

这些已经支持：

- 直接 `update_kv_cache()` 写 KV
- 用 context 控制 causal / non-causal
- query-only draft forward

#### 2）spec verify / accept-reject 合约

- `nanovllm/engine/spec_decode/eagle3.py`
- `nanovllm/engine/spec_decode/dflash.py`
- `nanovllm/engine/scheduler.py`

当前公共合约是：

- `seq.prev_draft_tokens` 保存下一轮要验证的 draft tokens
- target verify row 形如 `[last_token] + prev_draft_tokens`
- verify 后返回 accepted prefix + correction / bonus token

DSpark Phase 1 可以沿用这套合约。

#### 3）chunked prefill / mixed batch 支持

当前 DFlash backend 已经显式处理：

- partial prefill：只累积 context KV，不 proposal
- finishing prefill：sample 第一个真实 token 后立即 proposal
- mixed prefill + decode

DSpark 应直接复用这一点，而不应重做一套独立逻辑。

---

## 6. 实现范围建议

## 6.1 Phase 1：固定 verify 长度的 DSpark（推荐先做）

范围：

- 只支持 `spec_method="dspark"`
- 只支持 Qwen3 target + Qwen3 DSpark draft
- 复用 DFlash backbone
- 实现 Markov head
- 实现 sequential Markov sampling
- verify 长度固定为 `K`
- 暂不接 confidence scheduler

收益：

- 与 vLLM 当前开源 DSpark 路径基本对齐；
- 代码量可控；
- 不改 scheduler 主干语义。

## 6.2 Phase 2：论文完整版本 DSpark

新增：

- confidence head
- calibration（STS）
- hardware-aware prefix scheduler
- 动态 verify 长度

注意：这一阶段不应与 Phase 1 混在一起做，否则会让第一版 correctness 难以收敛。

## 6.3 暂不做的内容

第一轮不建议做：

- RNN head
- 训练代码
- pipeline parallel 支持
- SWA / mixed sliding-full attention
- 概率型 rejection-sampling 全量接入
- 生产版 Section 5.2 的两步滞后异步 scheduler

---

## 7. Phase 1 的模型设计

## 7.1 新文件

新增：

```text
nanovllm/models/qwen3_dspark.py
```

其定位应当非常明确：

- 不是新的 target model；
- 不是独立于 DFlash 的新 backbone；
- 而是 **在 `DFlashQwen3ForCausalLM` 之上增加 DSpark head**。

## 7.2 继承关系建议

```python
class DSparkMarkovHead(nn.Module):
    ...

class Qwen3DSparkModel(DFlashQwen3Model):
    ...

class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    ...
```

原因：

- `precompute_and_store_context_kv()` 完全复用 DFlash；
- `combine_hidden_states()` 完全复用 DFlash；
- query-only forward 完全复用 DFlash；
- 只是在 logits 头之后多了一段 sequential bias 注入。

## 7.3 `DSparkMarkovHead` 结构

与 vLLM 保持一致，使用低秩转移偏置：

```text
markov_w1: [target_vocab_size, markov_rank]
markov_w2: [markov_rank, draft_vocab_size]
```

语义：

- 输入：上一个已采样 token `x_{k-1}`
- 输出：当前位置 `k` 的 transition bias `B_k(x_{<k}, ·)`

即：

```text
embed(prev_token) -> markov_embed [r]
markov_embed -> bias over draft vocab [V_draft]
```

对应论文公式（Eq. 5, p.6）：

```text
B(x_{k-1}, ·) = W1[x_{k-1}] W2
```

## 7.4 `Qwen3DSparkForCausalLM` 应新增的方法

建议暴露以下 API：

```python
class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor: ...
    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor: ...
    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor: ...
```

说明：

- `compute_draft_logits()`：返回 **draft vocab 空间** 的 logits，不做 d2t scatter；
- `markov_bias()`：也输出 draft vocab 空间 bias；
- `map_draft_to_target()`：若 draft vocab 与 target vocab 不同，将最终采样 id 映射回 target vocab。

这样 runtime 可以按以下顺序做：

```text
hidden_states -> base draft logits
prev sampled token -> markov bias
base logits + bias -> sample in draft space
sampled draft id -> map to target vocab id
```

## 7.5 权重加载

需要兼容以下情况：

1. `embed_tokens` 缺失，复用 target embed
2. `lm_head` 缺失，复用 target lm_head
3. `d2t` 映射存在时加载；不存在且 vocab 不同则报错
4. `t2d` 忽略
5. `confidence_head` 先不接入 Phase 1；若 checkpoint 中存在，允许跳过

需要注意：

- 如果目标是对齐 vLLM 开源当前行为，Phase 1 可以和 vLLM 一样跳过 `confidence_head`
- 但代码结构应预留之后接入它的位置

---

## 8. Phase 1 的 backend 设计

## 8.1 新文件

新增：

```text
nanovllm/engine/spec_decode/dspark.py
```

推荐继承 `DFlashSpecBackend`：

```python
class DSparkSpecBackend(DFlashSpecBackend):
    ...
```

原因是 DSpark 与 DFlash 最大的共性都在 backend 里，而不在 model 里。

## 8.2 两种 query layout 都要支持

这是实现里最容易踩坑的地方。

### 模式 A：论文默认 / native DSpark

论文 Section 3.1（p.5）采用：

```text
K 个 query token = [anchor] + [mask] * (K - 1)
```

含义：

- anchor 本身就是第一个 prediction position
- 总 query 长度是 `K`
- 每个 query row 都要 sample
- 第 `i` 个 query row 预测的是下一个 token

### 模式 B：bonus-anchor 兼容模式

vLLM 同时兼容另一种 checkpoint 格式：

```text
K + 1 个 query token = [anchor] + [mask] * K
```

含义：

- anchor 只是一个 bonus / fill-in token
- 真正 sample 的是后面 K 个 mask rows
- 这更接近 DFlash 的布局

### 推荐做法

在 config 中引入布尔开关：

```python
dspark_bonus_anchor: bool = False
```

并在 backend 初始化时统一为：

```python
self.sample_from_anchor = not dspark_bonus_anchor
self.num_query_per_req = K if self.sample_from_anchor else K + 1
```

这样：

- 论文默认路径走 `sample_from_anchor=True`
- 如需兼容 vLLM speculators-format checkpoint，可设 `dspark_bonus_anchor=True`

## 8.3 Phase 1 复用 DFlash 的部分

以下逻辑建议 **完全复用 DFlash**：

1. `_run_spec_target_forward()`
2. `_combine_captured_hidden()`
3. `_append_context_range()`
4. `_precompute_context_kv()`
5. target verify 的 finishing-prefill / decode-greedy accept-reject
6. block rollback

换句话说，DSpark backend 中真正要改的是：

- `_build_spec_draft_sync_batch()` 的 proposal 信息打包
- `_run_spec_draft_sync_and_propose()` 的 query 构造与 sampling

## 8.4 `proposal_infos` 的语义

沿用 DFlash 的设计即可：

```python
@dataclass
class DSparkProposalInfo:
    seq: Sequence
    bonus_token: int
    query_start_position: int
    rollback_num_blocks: int
```

注意这里虽然字段名叫 `bonus_token`，但在 native DSpark 下它本质上是 **anchor token**。

来源：

- finishing prefill：target 本轮 sample 出的第一个真实 token
- decode verify：accepted prefix 的最后一个 token，或 correction token

关键原则：

- **这个 token 作为下一轮 DSpark query 的 anchor 输入**
- **不是 target context KV 的一部分**
- 因为当前 target forward 并没有计算它自己的 hidden state

这点和 DFlash 完全一致。

## 8.5 query 构造

### native DSpark

对每个 request 构造：

```text
input_ids   = [anchor] + [mask] * (K - 1)
positions   = [P, P+1, ..., P+K-1]
num_query   = K
sample_rows = all K rows
sample_pos  = [P+1, P+2, ..., P+K]
```

其中：

- `P = query_start_position`
- anchor token 被放在位置 `P`
- 第 0 个 query row 预测位置 `P+1`

### bonus-anchor 兼容模式

对每个 request 构造：

```text
input_ids   = [anchor] + [mask] * K
positions   = [P, P+1, ..., P+K]
num_query   = K + 1
sample_rows = 后 K 个 rows
sample_pos  = [P+1, P+2, ..., P+K]
```

### slot 分配

- native DSpark：`append_n_slots(seq, K, start_pos=P)`
- bonus-anchor：`append_n_slots(seq, K + 1, start_pos=P)`

proposal 完成后统一 `rollback_blocks()` 回滚到 `rollback_num_blocks`。

## 8.6 non-causal attention

DSpark 的 draft attention 应默认使用 **non-causal**。

原因：

- 论文和 vLLM 都将 parallel backbone 视为 block 内双向建模；
- DSpark 的顺序性来自 Markov head，而不是 backbone attention mask；
- vLLM `load_dspark_model()` 会显式设置 `use_non_causal=True`。

因此在 nano-vllm 中建议：

- Phase 1 直接把 DSpark backend 的 `causal` 固定为 `False`
- 不沿用 DFlash config 中的 `dflash_causal`

即：

```python
set_context(..., causal=False)
```

---

## 9. Phase 1 的 sequential Markov sampling 设计

## 9.1 总体流程

query-only forward 之后，拿到所有 query rows 的 hidden states：

```text
hidden_states
  -> select sample rows
  -> base draft logits [B, K, V_draft]
  -> left-to-right sampling with Markov bias
```

即：

```text
prev = anchor_token
for i in range(K):
    markov_embed = W1[prev]
    bias = W2(markov_embed)
    logits_i = base_logits[:, i, :] + bias
    sampled_i = sample(logits_i)
    prev = sampled_i
```

## 9.2 为什么必须逐位置循环

因为论文 Section 3.1（p.6）定义的是：

```text
p_k(v | x0, x<k) ∝ exp(U_k(v) + B_k(x0, x<k, v))
```

其中 `B_k` 依赖于 **此前已经采样出的 token**。

所以：

- backbone forward 可以并行；
- 但最终采样循环必须是 left-to-right 的；
- 这是 DSpark 与 DFlash 的本质差异。

## 9.3 greedy / probabilistic 的建议

nano-vllm 当前 speculative 路径基本是 greedy verify，因此 Phase 1 建议：

- 先实现 **greedy DSpark**
- 即 `sampled_i = argmax(logits_i)`
- 若 draft vocab != target vocab，则先在 draft space 取 argmax，再 `map_draft_to_target()`

后续若要接 probabilistic rejection-sampling，再加入：

- draft logits 缓存
- d2t scatter buffer
- Gumbel sampling / probability ratio test

但这不应阻塞第一版。

## 9.4 与 DFlash 的关系

一个非常好的 sanity check 是：

- 如果把 Markov bias 置零，DSpark 应退化为“并行 backbone + 独立位置 argmax”
- 即行为接近 DFlash（仅 query layout 不同时存在轻微差异）

这是后续测试的重要基线。

---

## 10. Phase 1 的 target verify 设计

## 10.1 verify 合约保持不变

在 Phase 1 中，target verify 仍保持当前合约：

```text
target verify row = [last_token] + prev_draft_tokens
```

其中：

- `len(prev_draft_tokens) = K`
- target forward row 长度仍是 `K + 1`
- scheduler 的 decode 成本仍是 `K + 1`

也就是说：

- DSpark proposal 端的 query 长度可能是 `K` 或 `K+1`
- 但 target verify 长度仍是 `K+1`

这是完全合理的，因为 verify 需要：

1. 验证 K 个 draft token
2. 在全命中时额外给出一个 bonus token

## 10.2 finishing prefill 处理

与 DFlash 保持一致：

- partial prefill：只预写 context KV，不 proposal
- finishing prefill：
  1. target 采样第一个真实 token
  2. 若未结束，则以该 token 作为 anchor
  3. 立即生成 `seq.prev_draft_tokens`

## 10.3 correction token 的正确处理

若 decode verify 在第 `j` 个 draft 位置拒绝：

- target 只计算到了 verify row 中已有 token 的 hidden states
- correction token 是 logits 产物，不存在自身 hidden state

因此：

- correction token **不能**被写入 context KV
- 它只能作为下一轮 query 的 anchor token

这和当前 DFlash backend 的正确性原则完全一致。

---

## 11. Phase 1 需要改动的文件

## 11.1 新增文件

```text
nanovllm/models/qwen3_dspark.py
nanovllm/engine/spec_decode/dspark.py
```

## 11.2 修改文件

### `nanovllm/config.py`

增加：

```python
spec_method in {"eagle3", "dflash", "dspark"}
```

`dspark` 分支建议校验：

- target model_type 必须是 `qwen3`
- draft config 必须包含：
  - `mask_token_id`
  - `target_layer_ids` 或等价字段
  - `markov_rank`
- 若 `draft_vocab_size != vocab_size`，则必须存在 d2t
- 暂不支持 SWA / mixed layer types

可选字段：

- `dspark_bonus_anchor`
- `draft_vocab_size`
- `enable_confidence_head`
- `confidence_head_with_markov`

### `nanovllm/engine/spec_decode/__init__.py`

导出：

```python
from nanovllm.engine.spec_decode.dspark import DSparkSpecBackend
```

### `nanovllm/engine/model_runner.py`

增加 dispatch：

```python
elif config.spec_method == "dspark":
    self.spec_backend = DSparkSpecBackend(self)
```

KV cache 分配基本可复用当前逻辑：

- 现在代码已经是 `eagle3` 用 1 层，其他 spec backend 用 draft `num_hidden_layers`
- `dspark` 会自然走多层 draft KV cache

但仍建议显式注释说明：

- `dspark` 和 `dflash` 一样使用 multi-layer draft KV cache

### `nanovllm/engine/scheduler.py`

Phase 1 不需要修改 decode 调度语义：

- `decode_target_cost(seq)` 仍返回 `K + 1`
- `seq.prev_draft_tokens` 生命周期保持不变

---

## 12. Phase 1 的实现顺序（推荐）

### 步骤 1：先把模型加上

先新增 `nanovllm/models/qwen3_dspark.py`：

- 继承 DFlash 模型
- Markov head 能 load weights
- eager forward 正常
- `compute_draft_logits()` / `markov_bias()` / `map_draft_to_target()` 可用

### 步骤 2：只支持 pure decode 的 DSpark backend

先做最简单闭环：

- decode only
- fixed `K`
- greedy sampling
- native DSpark query layout

验证：

- `prev_draft_tokens` 能生成
- target verify 能正常 accept/reject
- rollback 正确

### 步骤 3：补 finishing prefill

让 finishing prefill 在 sample 第一个真实 token 后，立即 proposal 下一轮 draft。

### 步骤 4：补 mixed chunked prefill + decode

复用 DFlash 当前逻辑，把 DSpark proposal 接进来。

### 步骤 5：补 bonus-anchor 兼容模式

如果要兼容 vLLM 某些现成 checkpoint，再支持：

```python
dspark_bonus_anchor = True
```

### 步骤 6：再考虑 CUDA graph

vLLM 已证明 DSpark 的并行 backbone forward + sequential sampling 仍可被 full graph capture。

但 nano-vllm 第一版不建议一开始就上图，因为：

- query layout 有两种 shape
- sequential loop 更容易在第一版先以 eager 验证 correctness

---

## 13. Phase 2：confidence-scheduled verification 设计

这一部分是论文真正把 DSpark 与“只是带 Markov head 的 DFlash”区分开的地方。

## 13.1 先明确一个事实：vLLM 开源版当前没有这一层

vLLM 当前：

- loader 跳过 `confidence_head`
- runtime 不消费 confidence score
- scheduler 没有动态 verify length

因此 nano-vllm 若要做这一阶段，本质上是 **在 vLLM 当前开源实现之上再补论文逻辑**。

## 13.2 新增模型头

论文 Eq. 7（p.7）定义：

```text
c_k = σ(w^T [h_k; W1[x_{k-1}]])
```

因此建议在 `Qwen3DSparkForCausalLM` 中再加一个轻量 head：

```python
class DSparkConfidenceHead(nn.Module):
    ...
```

输入：

- backbone hidden state `h_k`
- 可选 markov embedding `W1[x_{k-1}]`

输出：

- 每个 draft 位置的 conditional acceptance estimate `c_k ∈ (0, 1)`

建议 API：

```python
def compute_confidence(
    self,
    hidden_states: torch.Tensor,
    prev_tokens: torch.Tensor,
) -> torch.Tensor: ...
```

输出 shape：

```text
[B, K]
```

## 13.3 Sequence 状态如何存

Phase 2 开始，仅靠 `seq.prev_draft_tokens` 不够了。

最小增量建议是给 `Sequence` 增加 DSpark 专用字段：

```python
self.prev_draft_confidences: list[float] = []
self.prev_draft_survival: list[float] = []
self.verify_prefix_len: int = 0
```

不建议此时就全仓重构成抽象 `spec_state`，原因：

- 当前 repo 只有 EAGLE3 / DFlash / DSpark 三种 backend
- 只有 DSpark 真正需要额外 per-draft metadata
- 直接加 2~3 个字段更符合当前代码风格

## 13.4 prefix survival 的计算

按论文 Section 3.2.2（p.8）：

```text
a_{r,j} = Π_{i<=j} c_{r,i}
```

这里：

- `c_{r,i}`：第 `r` 个请求第 `i` 个 draft 位置的 conditional survival
- `a_{r,j}`：该请求 prefix 长度至少到 `j` 的 survival probability

proposal 阶段结束后即可计算：

```python
survival = torch.cumprod(confidences, dim=-1)
```

然后写回 `seq.prev_draft_survival`。

## 13.5 调度器目标函数

论文 Algorithm 1（p.8）优化的是：

```text
θ = τ * SPS(B)
```

其中：

- `B = R + Σ l_r`：本轮送到 target 的总 verify token 数（每个请求至少 1 个 last_token）
- `τ = R + Σ Σ_{j<=l_r} a_{r,j}`：期望接受 token 数
- `SPS(B)`：target model 在 batch size `B` 下的吞吐曲线

因此调度器要做的不是“每个请求独立选阈值”，而是：

- 在整个 active batch 上联合决定每个请求的 verify prefix length `l_r`
- 目标是让系统总吞吐最大，而不是单请求 acceptance 最大

## 13.6 nano-vllm 中的调度实现建议

### Step A：离线 profile `SPS(B)`

在 engine 初始化或 warmup 阶段做一张表：

```text
B -> target forward SPS(B)
```

可以只 profile 常见 batch token 数，例如：

```text
1..max_num_batched_tokens
```

也可以粗粒度采样后插值。

### Step B：为每个 decode seq 计算 candidate prefixes

对于每个 seq：

```text
(a_1, a_2, ..., a_K)
```

### Step C：全局 greedy admission

按论文 Algorithm 1：

1. 构造所有 candidate `(request, prefix_pos)`
2. 按 survival probability 从高到低排序
3. 逐步扩展验证前缀，更新：
   - 当前各请求的 `l_r`
   - 总 batch size `B`
   - 期望 accepted `τ`
   - 吞吐 `θ = τ * SPS(B)`
4. 记录最优解

### Step D：早停规则必须保留

论文 Appendix A（p.32-33）给出重要结论：

- **不能**做“看完整个未来后再回头选全局最优 prefix”
- 否则 admission 决策会依赖尚未允许依赖的信息，破坏 lossless property

因此实现时必须保留论文的 **early-stopping / causal admission** 思想，不能做 retrospective global search。

## 13.7 Phase 2 对 scheduler / backend 的具体改动

### `Scheduler.decode_target_cost()`

从固定：

```python
return K + 1
```

改为：

```python
return seq.verify_prefix_len + 1
```

其中 `verify_prefix_len ∈ [0, K]`。

### target verify batch

decode row 改为：

```text
[last_token] + prev_draft_tokens[:L]
```

其中 `L = seq.verify_prefix_len`。

### verify 结果

- 命中时最多接受 `L` 个 draft token
- 若前 `L` 个全命中，则额外产出一个 bonus token
- 未送去 verify 的 trailing draft tokens 直接丢弃

### proposal 阶段

proposal 仍然可以继续产生固定 `K` 个 draft token。

也就是说：

- **draft length 固定**
- **verify length 动态**

这是论文的核心思路。

## 13.8 calibration（STS）

论文指出 raw confidence 往往过于自信（Figure 6, p.15），因此：

- 若没有 calibration 数据，建议不要默认开启 dynamic scheduler
- Phase 2 需要支持从 checkpoint / sidecar metadata 读入 per-position temperature
- 若没有 temperature，则可：
  - 仅输出 raw confidence 作为 debug 指标
  - 或退回 fixed-length verify

---

## 14. 进一步的工程化：Section 5.2 异步 scheduler 是否要做

论文 Section 5.2 讲的是 **生产系统版** adaptation：

- jagged SPS 曲线
- CUDA graph / ZOS 约束
- 用“两步前的历史预测”近似下一步容量

对 nano-vllm 而言，这不应成为第一轮阻塞项。

建议分层理解：

### Phase 2A：先做论文 Algorithm 1 的同步版

适用场景：

- 当前 nano-vllm 的同步 step loop
- correctness / algorithm 验证优先
- 不追求完全 production-grade 的硬件调度

### Phase 2B：若后续追求线上吞吐，再做异步版

需要额外能力：

- scheduler 与 backend 更强的状态解耦
- lagged confidence prediction
- 非平滑 SPS 曲线下的 admission 近似
- CUDA graph shape 管理

换句话说：

- **同步版 prefix scheduler 就足以体现论文主要思想**
- **异步版是后续系统优化项，不应阻塞 DSpark 首版实现**

---

## 15. 推荐的文件级改造清单

## Phase 1 必改

```text
nanovllm/config.py
nanovllm/engine/model_runner.py
nanovllm/engine/spec_decode/__init__.py
nanovllm/models/qwen3_dspark.py      # new
nanovllm/engine/spec_decode/dspark.py # new
```

## Phase 2 必改

```text
nanovllm/engine/scheduler.py
nanovllm/engine/sequence.py
nanovllm/models/qwen3_dspark.py
nanovllm/engine/spec_decode/dspark.py
```

## 可能需要但不应前置的优化项

```text
nanovllm/layers/attention.py         # 若后续做更深的 kernel 优化
nanovllm/engine/spec_decode/dspark.py # CUDA graph / packing 优化
```

---

## 16. 正确性约束（实现时必须守住）

1. **只有真正算过 hidden state 的 token 才能写入 context KV**
   - correction / bonus token 不应伪造 hidden state

2. **proposal 的临时 query slots 必须 rollback**
   - 不得污染真实 sequence block_table

3. **native DSpark 与 bonus-anchor 两种 query layout 都必须最终产出 K 个 `prev_draft_tokens`**
   - 否则 scheduler / verify 合约会乱

4. **DSpark draft attention 默认 non-causal**
   - 顺序性来自 Markov head，而不是 attention mask

5. **dynamic scheduler 不能做 retrospective global search**
   - 必须保留论文 Appendix A 的因果性约束

6. **如果 draft vocab != target vocab，必须保证 d2t 存在**
   - 否则禁止运行

7. **没有 calibration 时不要静默启用 confidence scheduling**
   - 否则 throughput decision 可能系统性偏移

---

## 17. 测试与验证矩阵

## 17.1 Phase 1 必测

### 模型层

- `Qwen3DSparkForCausalLM` 能正确 load weights
- `compute_draft_logits()` 输出 shape 正确
- `markov_bias()` 输出 shape 正确
- `d2t` 映射正确

### backend 层

- pure decode / native DSpark query layout 正确
- pure decode / bonus-anchor query layout 正确
- finishing prefill 能立即 proposal
- mixed chunked prefill + decode 正确
- rollback 后 block_table 长度正确

### 行为层

- `prev_draft_tokens` 长度始终等于 `K`
- 将 Markov bias 人为置零时，行为退化接近 DFlash
- EOS / max_tokens 边界正确

## 17.2 Phase 2 必测

- confidence 输出 shape 与数值范围正确
- `cumprod(confidence)` 的 prefix survival 单调非增
- scheduler 选出的 `verify_prefix_len` 在 `[0, K]`
- target verify row 长度随 `verify_prefix_len` 动态变化
- early-stopping 逻辑不破坏 acceptance correctness
- 未校准 / 已校准两种模式行为明确

## 17.3 对齐验证

如果手头有与 vLLM 一致的 checkpoint，建议做以下对齐：

- 相同 prompt
- 相同 `K`
- 相同 query layout (`dspark_bonus_anchor`)
- greedy 模式

比较：

- proposal tokens
- verify accepted length
- 最终生成结果

---

## 18. 推荐落地顺序（最终版）

### Milestone 1：vLLM-parity DSpark

- `spec_method="dspark"`
- Qwen3 DSpark model
- Markov head
- native query layout
- greedy sampling
- fixed-length verify

### Milestone 2：checkpoint compatibility

- `dspark_bonus_anchor`
- shared embed/lm_head
- d2t mapping

### Milestone 3：mixed prefill/decode 完整跑通

- chunked prefill
- finishing prefill
- rollback / block correctness

### Milestone 4：性能优化

- CUDA graph
- 更高效的 query packing
- 若必要，再考虑 fused kernel 优化

### Milestone 5：paper-complete scheduler

- confidence head
- calibration
- dynamic verify prefix
- `SPS(B)` profile + greedy admission

### Milestone 6：更强系统优化（可选）

- 论文 Section 5.2 异步近似 scheduler
- 更强生产级 throughput 优化

---

## 19. 最终建议

如果现在的目标是“尽快把 DSpark 做出来并能跑”，我建议你按下面这条线推进：

```text
先做 Phase 1：
Qwen3 DSpark = 现有 DFlash backbone + Markov sequential head
verify 仍固定 K

跑通后再做 Phase 2：
confidence head + calibrated prefix scheduler
```

原因很简单：

1. 这条路线与 vLLM 当前开源实现最贴近；
2. 现有 nano-vllm DFlash 代码已经提供了 80% 的基础设施；
3. Markov head 的实现增量小，但收益很大；
4. confidence scheduler 是系统层功能，拆开做更稳。

如果后续要真正对齐论文的“DSpark 完整形态”，重点就不再是 model，而是：

- confidence state 如何存
- `SPS(B)` 如何建模
- scheduler 如何在保证 causality 的前提下动态裁剪 verify prefix

这三点会决定最终 throughput 上限。
