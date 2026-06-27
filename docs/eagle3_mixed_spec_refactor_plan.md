# nano-vLLM EAGLE3 Unified Spec Refactor Plan

目标：开启 EAGLE3 后，engine 外层只区分 **spec 开 / spec 关**。不再在 `LLMEngine.step()` 里区分 prefill-only、decode-only、mixed 走不同 model-runner 路径。

> 注意：外层不区分 prefill/decode；但 model runner 内部仍然需要 row metadata 标记每一行是什么语义，因为 prefill 和 decode-verify 的 input 构造、logits 解释和 draft sync 规则不同。这个区别只作为统一 batch 内部的 metadata，不再变成多个 target forward 或多个外部分支。

## 1. 背景

当前 `LLMEngine.step()` 在开启 EAGLE3 且同一轮同时有 `prefill_seqs` 和 `decode_seqs` 时，会拆成两次 model-runner 调用：

```python
accepted_tokens = self.model_runner.call("run_speculative", decode_seqs)
token_ids = self.model_runner.call("run", prefill_seqs, [])
```

这会让同一个 scheduler step 做两次 target-model forward。vLLM v1 的结构是：scheduler 产生统一 batch，worker 做一次 target forward，然后用 target outputs 做 sampling / rejection，并驱动 drafter 生成下一轮 draft tokens；额外工作只发生在 drafter 上。

本次改动目标：只要 `has_spec=True`，就走统一的 spec step：

- 一个 model-runner 入口处理本轮所有 scheduled seqs。
- 一次 target forward 覆盖本轮所有需要 target 计算的 rows。
- target forward 后统一做 sampling / accept-reject / draft KV sync / 生成下一轮 draft tokens。
- scheduler 统一 postprocess spec result。

## 2. 外层执行路径

`LLMEngine.step()` 最终应该变成这种结构：

```python
prefill_seqs, decode_seqs = self.scheduler.schedule()

if self.has_spec:
    result = self.model_runner.call(
        "run_speculative_step",
        prefill_seqs,
        decode_seqs,
    )
    self.scheduler.postprocess_speculative_step(result)
else:
    token_ids = self.model_runner.call("run", prefill_seqs, decode_seqs)
    self.scheduler.postprocess(prefill_seqs, decode_seqs, token_ids)
```

也就是外层只分两种：

| 模式 | 路径 |
| --- | --- |
| `has_spec=False` | 普通 `run()` |
| `has_spec=True` | 统一 `run_speculative_step()` |

不再有：

```python
if has_spec and prefill_seqs and decode_seqs:
    ...
elif has_spec and decode_seqs:
    ...
elif has_spec and prefill_seqs:
    ...
```

## 3. 统一 spec step 的内部流程

`run_speculative_step(prefill_seqs, decode_seqs)` 内部统一处理所有情况：

```text
build one target batch
  rows = scheduled prefill chunks
       + decode-verify rows

one target forward with capture_layers
  hidden_out, captured = target(...)
  logits = lm_head.forward_all(hidden_out)

interpret logits by row metadata
  partial prefill   → no user-visible sampled token
  finishing prefill → sampled token + 生成 prev_draft_tokens
  decode-verify     → accept/reject + bonus

draft KV sync / 生成下一轮 drafts by row metadata
  partial prefill   → draft KV sync only
  finishing prefill → draft KV sync + 生成 prev_draft_tokens
  decode-verify     → accepted-only draft KV sync + 生成 prev_draft_tokens

return structured result
  prefill updates
  decode accepted tokens
  token counts
```

一句话总结：

```text
interpret logits by row metadata
= target forward 后，根据每个 row 的类型决定 logits 是 sample、discard，还是做 accept/reject。

draft KV sync / 生成下一轮 drafts by row metadata
= target 结果确定后，根据每个 row 的类型决定 draft model 是只同步 KV，还是生成下一轮 prev_draft_tokens。
```

这样即使本轮只有 prefill、只有 decode、或者 prefill/decode 混合，都是同一个入口。

### 3.1 当前 spec 实现流程

当前实现可以按 **scheduler → target verify → logits 解释 → draft sync/proposal → scheduler postprocess** 五段理解。

#### 3.1.1 Scheduler 阶段

scheduler 仍然返回：

```python
prefill_seqs, decode_seqs = self.scheduler.schedule()
```

开启 spec 后，decode 的 token budget 不是 1，而是：

```python
decode_target_cost = K + 1
```

其中 `K = config.num_spec_tokens`。进入 decode 的 seq 必须已经有完整的：

```python
seq.prev_draft_tokens  # 长度为 K
```

chunked prefill 模式下调度顺序是：

```text
1. running 队列优先：decode 优先，也可能继续未完成的 chunked prefill。
2. 如果还有 token budget，再从 waiting 队列拉新 prefill。
3. spec decode 按 K + 1 个 target tokens 计入 max_num_batched_tokens。
```

#### 3.1.2 `LLMEngine.step()` 外层分支

外层只区分 spec 开 / 关：

```python
if self.has_spec:
    result = self.model_runner.call("run_speculative_step", prefill_seqs, decode_seqs)
    self.scheduler.postprocess_speculative_step(result)
else:
    token_ids = self.model_runner.call("run", prefill_seqs, decode_seqs)
    self.scheduler.postprocess(prefill_seqs, decode_seqs, token_ids)
```

spec 模式下不再把 mixed step 拆成 `run_speculative()` + `run()` 两次 target forward。

#### 3.1.3 Target batch 构造

`_build_spec_target_batch(prefill_seqs, decode_seqs)` 把 prefill rows 和 decode-verify rows 拼成一个 flat target batch。

prefill row：

```text
input_ids = seq[start:end]
positions = start ... end-1
q_len = end - start
k_len = end
```

其中：

```python
start = seq.num_computed_tokens
end = start + seq.scheduled_chunk_size
```

prefill row 会记录：

```python
is_finishing_prefill = end >= seq.num_tokens
```

用于区分 partial prefill 和 finishing prefill。

decode-verify row：

```text
input_ids = [last_token] + prev_draft_tokens
positions = original_len - 1 ... original_len + K - 1
q_len = K + 1
k_len = original_len + K
```

构造 decode row 前会先预留 K 个 future slots：

```python
self.block_manager.append_n_slots(seq, K)
```

注意只预留 K 个，不是 K + 1 个，因为 `last_token` 已经在 logical sequence 里。

#### 3.1.4 Target forward

target forward 统一返回：

```python
hidden_out, captured = self._run_spec_target_forward(...)
```

其中 `captured` 是 EAGLE3 fuse layers 的 target hidden states，后续给 draft model sync/proposal 使用。

pure decode-only spec verify 会优先尝试专门的 CUDA graph：

```text
条件：
  prefill_seqs 为空
  decode_seqs 非空
  not enforce_eager
  graph bucket 覆盖当前 decode batch size
  max_k 不超过 graph capture 上限
```

graph query length 固定为：

```python
q_len = K + 1
```

mixed prefill+decode 或 prefill-only spec step 继续走 eager varlen target forward。

#### 3.1.5 Target logits 解释

`_select_spec_logits()` 不会对整个 `hidden_out` 做 vocab projection，而是只选择需要 logits 的位置：

```text
finishing prefill: 只取 row 最后一行 logits，用于 sample 第一个真实生成 token。
decode_verify:    取 K + 1 行 logits，用于验证 K 个 draft tokens + bonus token。
partial prefill:  不取 logits，只保留 hidden/captured 给 draft sync。
```

这样避免大 prefill batch 下对所有 prompt tokens 做完整 vocab projection。

finishing prefill：

```python
sampled_token = sampler(last_prompt_logits)
```

这个 token 是 target 真实生成 token，稍后由 scheduler append 到 `seq.token_ids`。

decode verify 当前是 greedy accept/reject：

```text
for j in 0..K-1:
  if target_pred[j] == prev_draft_tokens[j]:
      accept draft[j]
  else:
      accept target_pred[j]
      break
else:
  accept bonus target_pred[K]
```

然后根据 EOS / max_tokens 截断 accepted tokens。

#### 3.1.6 Draft sync batch 构造

`_build_spec_draft_sync_batch()` 根据 row metadata 构造 draft model 的 sync batch。

partial prefill：

```text
用 prompt 中已知的 next token 做 shifted input。
只同步 draft KV，不 sample，不 proposal，不写 prev_draft_tokens。
```

finishing prefill：

```text
shifted input 的最后一个 token 使用 target sampled_token。
先同步 draft KV，再进入 proposal，生成下一轮 prev_draft_tokens。
```

decode verify：

```text
先根据 accepted tokens 计算 final_num_blocks。
rollback 到 accepted 后应该保留的 block 数。
如果 seq 还没结束，只用 accepted tokens 做 draft sync。
rejected draft tail 不进入 draft sync。
```

对应逻辑是：

```python
final_len = original_len + M
final_num_blocks = (final_len + block_size - 1) // block_size
self.block_manager.rollback_blocks(seq, final_num_blocks)
```

如果本轮已经 EOS / max_tokens，则清空：

```python
seq.prev_draft_tokens = []
```

并跳过 proposal。

#### 3.1.7 Draft sync forward

`_run_spec_draft_sync_and_propose()` 先把 sync batch 跑一次 draft model：

```python
sync_logits, sync_hidden_out = self.draft_model(
    sync_input_ids_t,
    sync_positions_t,
    sync_fused_hidden,
)
```

这一步的作用是：

```text
1. 把 draft KV cache 同步到 target 已确认的位置。
2. 得到每个 sync row 最后一行 logits / hidden，作为下一轮 proposal 的起点。
```

sync 是一个 batch；proposal 也整理成一个 batch，但 partial prefill 不参与 proposal。

#### 3.1.8 Proposal 生成下一轮 `prev_draft_tokens`

进入 proposal 的只有两类：

```text
finishing prefill：target 已 sample 出第一个真实生成 token。
decode_verify：accept/reject 后 seq 还没结束。
```

当前 proposal metadata 是：

```python
@dataclass
class SpecProposalInfo:
    seq: Sequence
    last_sync_index: int
    start_position: int
    rollback_num_blocks: int
```

`_run_spec_proposal_batch()` 先从 sync 最后一行 logits 得到第一个 draft token：

```python
first_draft_token = self.draft_model.d2t[sync_logits[last_indices].argmax(dim=-1)]
```

如果 `K == 1`，这个 token 就是完整的 `prev_draft_tokens`，不需要额外 draft forward，也不需要 proposal slots。

如果 `K > 1`，EAGLE3 继续 serial 生成剩余 `K - 1` 个 draft tokens：

```text
d0 -> d1 -> ... -> dK-1
```

proposal 开始时一次性按显式 `start_position` 预留 slots：

```python
draft_slots = [
    self.block_manager.append_n_slots(info.seq, K - 1, start_pos=info.start_position)
    for info in proposal_infos
]
```

`_generate_draft_tokens_from_state()` 只消费这些 `draft_slots`，不再自己隐式 append slots。proposal 完成后写：

```python
seq.prev_draft_tokens = [d0, d1, ..., dK-1]
```

然后 rollback proposal 阶段临时扩出来的 blocks。注意 rollback 不清空 draft KV tensor 内容，只是让 `seq.block_table` 不再指向这些未验证 draft slots。

#### 3.1.9 Scheduler postprocess

`run_speculative_step()` 返回后，scheduler 才真正修改 `seq.token_ids`。

prefill：

```python
seq.num_computed_tokens += seq.scheduled_chunk_size
if not seq.is_prefill:
    seq.append_token(sampled_token)
```

其中 `sampled_token` 是 target 模型采样出的第一个真实生成 token，不是 draft token。

decode：

```python
seq.append_tokens(accepted_tokens)
```

只有 target verify 接受后的 tokens 才会 append 到 `seq.token_ids`。EAGLE3 proposal 产生的 `prev_draft_tokens` 在下一轮 verify 前都只是候选 token，不对用户可见，也不计入 completion tokens。

#### 3.1.10 当前 spec 状态不变量

当前实现保持这些状态约束：

```text
partial prefill:
  draft KV sync only
  不写 prev_draft_tokens

finishing prefill:
  target sampled_token 会被 scheduler append
  EAGLE3 draft tokens 只写入 prev_draft_tokens

decode verify:
  verify 前临时 append K 个 target verify slots
  verify 后 rollback 到 accepted blocks
  scheduler postprocess 再 append accepted tokens

proposal:
  K == 1 不需要额外 draft slots
  K > 1 预留 K - 1 个 draft proposal slots
  proposal 结束 rollback 临时 blocks
  长期只保存 prev_draft_tokens
```

## 4. 内部 row metadata

统一入口不代表所有 row 语义一样。model runner 内部需要记录 row metadata，用来切分 flat target outputs。

建议字段：

- `kind`: `prefill` / `decode_verify`
- `seq`
- `row_start`, `row_end`
- `original_len`
- `is_finishing_prefill`
- `start`, `end` for prefill chunks
- `num_prev_drafts` for verify rows

约束：spec 模式下，进入 decode 阶段的 active seq 必须已有完整 `prev_draft_tokens`。正常情况下 finishing prefill 会在同一个 spec step 里生成这些 drafts，后续 decode row 直接成为 `decode_verify`。如果 decode seq 没有完整 drafts，应视为状态错误并尽早暴露，而不是静默普通 decode。

这些 metadata 只存在于 `run_speculative_step()` 内部，用于：

- 构造 input ids / positions / slot mapping。
- 解释 target logits。
- 选择 captured hidden slices。
- 构造 draft sync input。
- 决定是否写 `seq.prev_draft_tokens`。

### 4.1 Row 坐标系说明

`run_speculative_step()` 里的 row metadata 同时保存两套坐标：

- `row_start` / `row_end`：本轮 flat target batch 里的切片范围。
- `start` / `end`：当前 `seq` 自己的逻辑 token 位置范围。

简单记：

```text
row_start / row_end：给 target output、logits、captured hidden 切片用。
start / end：给 seq 逻辑位置、draft sync、positions、slot mapping 用。
```

例如 prefill row：

```python
start = seq.num_computed_tokens
end = start + seq.scheduled_chunk_size
row_start = len(input_ids)
input_ids.extend(seq[start:end])
row_end = len(input_ids)
```

其中：

- `start/end` 表示 `seq[start:end]`，是 sequence 坐标。
- `row_start/row_end` 表示 `input_ids[row_start:row_end]`，是 flat batch 坐标。

后续 draft sync 需要 `start/end`：

```python
shifted = list(seq.token_ids[start + 1:end])
sync_positions.extend(range(start, end))
for pos in range(start, end):
    sync_slot_mapping.append(self._slot_for_position(seq, pos))
```

而 target logits / captured hidden 解释需要 `row_start/row_end`：

```python
logit_indices.append(row["row_end"] - 1)
sync_fused_indices.extend(range(row["row_start"], row["row_end"]))
```

## 5. Target batch row 设计

### 5.1 Prefill row

```python
start = seq.num_computed_tokens
end = start + seq.scheduled_chunk_size
input_ids = seq[start:end]
positions = range(start, end)
q_len = end - start
k_len = end
is_finishing_prefill = end >= seq.num_tokens
```

slot mapping 复用当前 `prepare_prefill()` 的 token-granular 逻辑。

### 5.2 Decode-verify row

用于已有完整 `prev_draft_tokens` 的 decode seq。

```python
L = len(seq)
input_ids = [seq.last_token] + seq.prev_draft_tokens
positions = range(L - 1, L + K)
q_len = K + 1
k_len = L + K
```

构造 slot mapping 前要确保 block table 覆盖 `L ... L+K-1`。

## 6. Target forward

统一 spec step 用一个 varlen-style target batch：

```python
set_context(
    has_prefill=True,
    cu_seqlens_q=...,
    cu_seqlens_k=...,
    max_seqlen_q=...,
    max_seqlen_k=...,
    p_slot_mapping=...,
    block_tables=...,
    num_prefill_tokens=total_target_tokens,
    num_decode_seqs=0,
)

hidden_out, captured = self.model(
    input_ids,
    positions,
    capture_layers=self.eagle3_fuse_layers,
)
logits = self.model.lm_head.forward_all(hidden_out)
```

这里不再用外层 prefill/decode 分支决定 target forward 次数。所有 scheduled target rows 都在同一次 forward 里完成。

## 7. Output 解释逻辑

| Row kind | Target logits 用法 | 产物 |
| --- | --- | --- |
| `partial prefill`  | 不产生 user-visible sample | 只用于 draft KV sync |
| `finishing prefill`  | sample row 最后一个位置 | sampled token + 后续 draft tokens metadata |
| `decode_verify` | 前 K 个位置验证 drafts，第 K+1 个位置是 bonus | accepted drafts / correction / bonus |

当前保持 greedy verification，不在本次 refactor 里实现 stochastic rejection sampling。

## 8. Draft KV sync / 生成下一轮 drafts 规则

### 8.1 Partial prefill

- draft shifted input 使用 prompt 中已知的下一个 token。
- 跑 draft model 填充 draft KV。
- 不设置 `seq.prev_draft_tokens`。
- 不生成下一轮 `prev_draft_tokens`。

### 8.2 Finishing prefill

- draft shifted input 的最后一个 token 使用 target sampled token。
- 跑 draft sync。
- 用 sync 最后位置的 logits/hidden 产生第一个 draft token。
- 继续调用 `_generate_draft_tokens_from_state()` 生成剩余 `K-1` 个 drafts。
- 写入 `seq.prev_draft_tokens = [d0, ..., dK-1]`。
- 如果 sampled token 已 EOS 或达到 max_tokens，清空 `prev_draft_tokens` 并跳过生成下一轮 drafts。
- 生成 draft tokens 用到的临时 blocks 必须 rollback。

### 8.3 Decode-verify

- 先从 target logits 做 accept/reject。
- rollback 到 accepted 后的 final logical length。
- 只用 accepted tokens 构造 draft sync shifted input。
- 不要写入 rejected draft tail。
- sync 后继续生成下一轮 drafts，覆盖 `seq.prev_draft_tokens`。

## 9. Scheduler 和 BlockManager 改动

### 9.1 BlockManager

新增：

- `num_extra_blocks_for_append(seq, n)`
- `can_append_n(seq, n)`

用途：spec verify / 生成下一轮 draft tokens 会临时覆盖多个 future positions，不能只用当前的一 token `can_append()` 判断。

### 9.2 Prefix cache full-hit 边界问题

问题复现：benchmark warmup 如果使用正式请求里的同一个 prompt，正式请求会命中 prefix cache。对于 `prompt_len=512`、`block_size=256` 的请求，warmup 后两个完整 block 都在 cache 中。如果 scheduler 先计算：

```python
chunk_size = min(seq.num_uncomputed_tokens, remaining_budget)
```

然后再 `allocate(seq)` 并设置：

```python
seq.num_computed_tokens = seq.num_cached_tokens
```

就会出现：

```text
allocate 前: num_computed_tokens=0,   num_uncomputed_tokens=512, chunk_size=512
allocate 后: num_computed_tokens=512, num_uncomputed_tokens=0,   chunk_size 仍是旧值 512
prepare_prefill: start=512, end=1024
```

结果 `positions` 长度是 512，但 `seq[512:1024]` 为空，最终触发 `input_ids.shape[0] != positions.shape[0]`。

修复原则不是“把 full-hit 的 cached block 复用下来，然后只重算最后一个 prompt token”。KV cache / prefix cache 的复用粒度是 block，已经 hash 命中的完整 block 应视为只读；如果在共享 cached block 里重写最后一个 token，会破坏 prefix-cache block immutable 的不变量。

正确做法是：**prefix cache 最多只命中到最后一个 token 之前对应的完整 block；换句话说，在 block 粒度上少匹配一个 block**。

对于 `prompt_len=512`、`block_size=256`：

```text
block0: token 0~255    可以复用 cached block
block1: token 256~511  不能复用，因为它包含 prompt 的最后一个 token
```

所以实际 cache hit blocks 应该是：

```text
max_cache_hit_blocks = floor((num_tokens - 1) / block_size)
                     = max(seq.num_blocks - 1, 0)
```

对于不同长度：

```text
num_tokens=512, block_size=256 -> max_cache_hit_blocks=1
num_tokens=513, block_size=256 -> max_cache_hit_blocks=2
num_tokens=300, block_size=256 -> max_cache_hit_blocks=1
```

这样保证 prefill 至少会从“包含最后一个 prompt token 的 block”开始重新计算，并用最后一个 prompt token 的 logits 采样第一个生成 token。

nano-vLLM 对应修复点可以写成 block 数判断：

```python
max_cache_hit_blocks = max(seq.num_blocks - 1, 0)
can_hit_cache = is_full_block and i < max_cache_hit_blocks
```

当前代码也可以用等价的 token 边界表达：

```python
max_cache_hit_tokens = seq.num_tokens - 1
block_end = (i + 1) * self.block_size
can_hit_cache = is_full_block and block_end <= max_cache_hit_tokens
```

两者语义相同：不是 token 级别重写 cached block，而是在 cache lookup 阶段就少命中最后一个可能包含采样所需 logits 的完整 block。

另一个必要修复是 scheduler 的 waiting prefill 路径必须先 `allocate(seq)` / 更新 `seq.num_computed_tokens`，再根据新的 `seq.num_uncomputed_tokens` 计算 `chunk_size`，否则即使少命中一个 block，也可能继续使用 allocate 前的旧 chunk size。

### 9.3 Spec decode slot 预分配边界问题

问题复现：finishing prefill 后，scheduler postprocess 会把 sampled token append 到 `seq.token_ids`，但这个 sampled token 的 KV 还没有被 target/draft forward 写入 cache。下一轮 spec decode-verify 会 forward：

```text
positions = [len(seq)-1] + K 个 draft positions
```

对于 `prompt_len=512`、`block_size=256`：

```text
prefill 后 block_table 覆盖 token 0~511
append sampled token 后 len(seq)=513，last_token position=512
position 512 属于 block_idx=2，但 block_table 仍然只有 2 个 block
```

旧的 `append_n_slots()` 只在 `temp_len % block_size == 0` 时分配新 block：

```python
if temp_len % self.block_size == 0:
    allocate_new_block()
block_idx = temp_len // self.block_size
slot = seq.block_table[block_idx] * self.block_size + block_offset
```

当 `temp_len=513` 时，`513 % 256 == 1`，不会分配新 block，但 `block_idx=2`，访问 `seq.block_table[2]` 会越界。

修复：`append_n_slots()` 不能假设“非 block 边界时对应 block 一定存在”，应按 `block_idx >= len(seq.block_table)` 判断是否需要分配：

```python
block_idx = temp_len // self.block_size
if block_idx >= len(seq.block_table):
    allocate_new_block()
```

另外 decode-verify 的 target row 是 `[last_token] + K drafts`。`last_token` 已经在 logical sequence 里，真正需要预分配的是 K 个 future draft/bonus positions，因此 `_build_spec_target_batch()` 里应调用：

```python
self.block_manager.append_n_slots(seq, K)
```

而不是 `K + 1`，避免多预留一个不用的 future slot/block。

### 9.4 Spec target logits OOM 问题

问题复现：统一 spec step 里如果直接对整个 target batch 调用：

```python
logits_all = self.model.lm_head.forward_all(hidden_out)
```

在大 prefill batch 下会把所有 prefill token 都投到 vocab。例如 `max_num_batched_tokens=16384`、Qwen3-4B vocab 约 151k，bf16 logits 大小约为：

```text
16384 * 151936 * 2 bytes ≈ 4.6 GiB
```

这会在 `lm_head.forward_all()` 处 OOM。普通 non-spec 路径没有这个问题，因为 `ParallelLMHead.forward()` 在 prefill 时只选择 finishing prefill 的 last-token hidden 去算 logits。

修复原则：target forward 可以覆盖所有 rows，但 vocab projection 只对真正需要 logits 的位置执行：

- finishing prefill：只需要 row 最后一个位置的 logits，用于采样第一个生成 token。
- decode-verify：需要该 verify row 的 `K+1` 个 logits，用于验证 K 个 draft token 和 bonus token。
- partial prefill：不需要 target logits，只需要 hidden/captured states 做 draft KV sync。

实现方式：先收集 `logit_indices`，再只对 selected hidden 调用 lm head：

```python
logit_indices = [finishing_prefill_last_positions] + [decode_verify_row_ranges]
selected_hidden = hidden_out.index_select(0, logit_indices_t)
logits_selected = self.model.lm_head.forward_all(selected_hidden)
```

后续通过 row metadata 里的 selected-logit offset 解释采样和 accept/reject 结果。这样避免为大段 prefill token 生成完整 vocab logits。

### 9.5 Spec decode max_tokens / EOS clamp

Spec decode-verify 一轮可能推进多个 token。如果当前 seq 只剩少量输出预算，但 accept/reject 结果包含更多 token，直接 append 会导致实际输出超过 `SamplingParams.max_tokens`。例如 benchmark 中 `target_output_toks=8192`，但 spec 实际输出到 8309。

修复：在 model runner 生成每个 decode row 的 `accepted` 后、写入 `decode_accepted_tokens` 前先截断：

```python
if not seq.ignore_eos and eos in accepted:
    accepted = accepted[:accepted.index(eos) + 1]
remaining = seq.max_tokens - seq.num_completion_tokens
accepted = accepted[:remaining]
```

这样 scheduler postprocess 只 append 合法 token 数，metrics 和 correctness 对齐 baseline。

### 9.6 Scheduler token budget

开启 spec 后，scheduler 仍然可以返回 `prefill_seqs, decode_seqs`，但 budget 计算要按真实 target token 数。

新增 helper：

```python
def decode_target_cost(seq):
    if not has_spec:
        return 1
    assert len(seq.prev_draft_tokens) == num_spec_tokens
    return num_spec_tokens + 1
```

在 `_schedule_chunked()` 和后续可能的 non-chunked spec 调度里，decode 行按真实 target token 数计入 `max_num_batched_tokens`。

### 9.7 CUDA graph / spec 性能问题

当前 benchmark 里 spec step 数明显减少，但端到端耗时仍可能比 baseline 慢。一个主要原因是当前 unified spec 路径没有复用 baseline decode 的 CUDA graph。

nano-vLLM 当前 baseline decode 路径：

```python
if has_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
else:
    graph.replay()
```

也就是说普通 pure decode 在 batch size 不大时会进入 captured CUDA graph replay。

但当前 `run_speculative_step()` 直接执行：

```python
hidden_out, captured = self.model(
    input_ids,
    positions,
    capture_layers=self.eagle3_fuse_layers,
)
```

它没有经过 `run_model()`，所以 target verify 不会进入普通 decode CUDA graph。注意：这里不是说 spec verify 在语义上是 prefill；它语义上仍然是 decode verify。只是 nano-vLLM 当前实现为了复用 varlen attention metadata，通过：

```python
set_context(
    has_prefill=True,
    num_prefill_tokens=input_ids.size(0),
    num_decode_seqs=0,
    ...
)
```

把这组 `K+1` query 交给 `has_prefill=True` 分支里的 `flash_attn_varlen_func` 执行。即使本轮只有 decode-verify row，也不会进入现有的一 token decode graph 路径。

vLLM v1 的思路更细：spec decode 的 verify row `[last_token] + K drafts` 被视为一种 **uniform decode**，其 query length 是：

```python
uniform_decode_query_len = 1 + num_speculative_tokens
```

然后由 `CudagraphDispatcher` 根据：

- `num_tokens`
- `uniform_decode`
- batch descriptor
- CUDA graph mode (`FULL` / `PIECEWISE` / `FULL_DECODE_ONLY` / `NONE`)

选择是否走 CUDA graph。也就是说 vLLM 明确把 spec verify 当作 uniform decode，而不是 ordinary prefill；`K+1` query len 的 spec decode batch 可以使用专门的 cudagraph dispatch 路径。vLLM 还会给 EAGLE drafter/proposer 单独初始化 cudagraph dispatcher。

短期处理：

1. `run_speculative_step()` 里当前用于 profiling 的多次 `torch.cuda.synchronize()` 会强制 CPU 等 GPU，benchmark 时应删除或放到 debug flag 下。
2. 加 counters/logging 确认：
   - baseline pure decode graph replay 次数。
   - spec target eager 次数。
   - draft sync / draft loop eager 次数。
3. 先实现 **pure spec decode-verify** 的 CUDA graph：
   - 固定每 seq query len = `K + 1`。
   - batch token 数按 `(K + 1) * num_decode_seqs` padding 到 capture size。
   - 单独 capture 带 `capture_layers` 的 target forward。
4. 后续再考虑：
   - mixed prefill+decode 的 piecewise/full graph。
   - draft sync 的 graph。
   - draft token loop 的 one-token decode graph。

注意：不能直接复用当前 `run_model()` 的 decode graph。它只支持每个 seq 一个 token 的 ordinary decode；spec verify 每个 seq 是 `K+1` 个 query token，attention metadata、slot mapping、positions、logits slicing 都不同，需要单独的 graph shape 和 context 构造。

### 9.8 当前 hot-path cleanup 修改说明

这轮优化基于 K sweep 的结果：`num_spec_tokens=3` 基本和 baseline 持平，但没有稳定超过 baseline。说明 step 数已经下降，瓶颈更多在每个 spec step 内部的额外开销。本轮只做低风险 hot-path cleanup，不改变 spec 解码语义、不改变 accept/reject 规则，也不引入 CUDA graph。

#### 9.8.1 只 fuse draft sync 需要的 captured hidden rows

原逻辑会先对 target forward 捕获到的所有 hidden rows 做 EAGLE3 `fc` 融合：

```python
fused_hidden_all = self._fuse_captured_hidden(captured)
sync_fused_hidden = fused_hidden_all[sync_fused_indices]
```

但 decode-verify row 是 `[last_token] + K drafts`，后续 draft sync 只需要被接受的 token rows。未接受的 draft tail 会 rollback，不需要进入 EAGLE3 sync。

因此改成 `_fuse_captured_hidden(captured, indices=None)`，当 `sync_fused_indices` 不是全量 rows 时，先 index 再做 `fc`：

```python
hidden = [captured[l].index_select(0, indices) for l in self.eagle3_fuse_layers]
return self.draft_model.fc(torch.cat(hidden, dim=-1))
```

正确性依据：`fc` 是逐 row 线性层，`fc(all_rows)[indices]` 与 `fc(all_rows[indices])` 等价。收益主要来自减少未接受 rows 的无效融合计算，K 越大或接受率越低越明显。

#### 9.8.2 批量回传 draft tokens，避免逐 token `.item()`

原逻辑为每个 seq、每个 draft token 单独 `.item()`：

```python
for i, seq in enumerate(seqs):
    seq.prev_draft_tokens = [draft_tokens_all[k][i].item() for k in range(K)]
```

`.item()` 会把 CUDA scalar 同步回 CPU，容易形成大量小同步点。本轮新增 `_assign_prev_draft_tokens()`：

```python
draft_tokens = torch.stack(draft_tokens_all, dim=1).cpu().tolist()
for seq, tokens in zip(seqs, draft_tokens):
    seq.prev_draft_tokens = tokens
```

这样把 `N * K` 次 scalar sync 合并成一次批量 GPU→CPU 拷贝，token 值不变。

#### 9.8.3 draft loop 复用 `draft_block_tables`

`_generate_draft_tokens_from_state()` 进入 draft loop 前已经通过 `append_n_slots()` 预分配了后续 draft token 需要的 slots。loop 内只更新 KV 内容和 context length，不再改变 `seq.block_table`。

因此 `draft_block_tables = self.prepare_block_tables(seqs)` 从每个 draft step 内部移到 loop 外：

```python
draft_block_tables = self.prepare_block_tables(seqs)
for k in range(num_steps):
    ...
    set_context(..., block_tables=draft_block_tables, ...)
```

每步仍然重新计算 `draft_context_lens = draft_positions + 1`，所以 attention 可见长度保持正确。收益来自减少 Python list padding、tensor 构造和小 H2D 拷贝。

#### 9.8.4 decode verify logits 整批 argmax / CPU copy

原逻辑对每个 decode row 单独执行：

```python
row_logits = logits_selected[row["logit_start"]:row["logit_end"]]
target_token_ids = row_logits.argmax(dim=-1).cpu().tolist()
```

本轮改成对所有 decode rows 一次性处理：

```python
decode_target_token_ids = (
    logits_selected[decode_logits_start:]
    .argmax(dim=-1)
    .view(len(decode_rows), K + 1)
    .cpu()
    .tolist()
)
```

之后 Python 侧仍按原 accept/reject 规则逐 row 解释结果。正确性依据是 `logits_selected` 中 decode rows 本来就是按 row 顺序连续排列，每个 row 固定 `K + 1` 个 logits。收益来自减少 per-row `argmax + cpu + tolist` 的碎片化同步。

#### 9.8.5 sync fused hidden 延后到知道 accepted rows 后再构造

原先在 accept/reject 之前就构造全部 `fused_hidden_all`。现在先完成 accept/reject、rollback 和 `sync_fused_indices` 收集，再只为即将进入 draft sync 的 rows 构造 `sync_fused_hidden`。

如果 `sync_fused_indices` 恰好覆盖全部 rows，则仍走 full fuse，避免额外 index；否则走 selective fuse：

```python
use_full_fuse = (
    len(sync_fused_indices) == input_ids.size(0)
    and all(i == idx for idx, i in enumerate(sync_fused_indices))
)
sync_fused_hidden = self._fuse_captured_hidden(
    captured,
    None if use_full_fuse else sync_fused_indices,
)
```

这属于纯性能改动，不改变 draft sync 的输入语义。

#### 9.8.6 本轮优化边界

本轮没有处理最大的结构性问题：baseline pure decode 仍可走 CUDA graph replay，而 spec target verify 语义上虽然是 decode，但实现上仍是 eager，并通过 `has_prefill=True` 复用 varlen attention 分支。因此这些 cleanup 只能降低 spec step 内部固定开销，不能替代后续的 pure spec decode-verify CUDA graph。

下一步应使用 `BEST_K=3 --spec-profile` 看分解：如果 `target_fwd` 占大头，继续实现固定 `K+1` query len 的 spec verify CUDA graph；如果 `draft_all` 占大头，再优先优化 draft sync / draft loop。

### 9.9 Pure spec decode-verify CUDA graph 计划

这轮 `K=3 --spec-profile` 的稳态结果显示：

```text
last total  ≈ 48.5ms
last target ≈ 41.8ms
last draft  ≈ 4.5ms
verify      ≈ 1.6ms
fuse+sync   ≈ 0.3ms
build       ≈ 0.2ms
```

因此当前瓶颈已经明确是 **target verify forward**，不是 draft loop。`target_fwd` 约占单步耗时 86%，继续优化 `.item()`、draft loop、fuse/build 对整体收益有限。下一阶段应优先让 pure decode-only spec verify 进入 CUDA graph / uniform decode 路径。

#### 9.9.1 vLLM 参考点

可参考 `/Users/xiao-jy/repos/vllm` 中 vLLM v1 的设计：

- `vllm/v1/cudagraph_dispatcher.py`
  - `uniform_decode_query_len = 1 + num_speculative_tokens`
  - spec verify row 语义上是 decode verify；vLLM 将它视为固定 query len 的 uniform decode，而不是 prefill。
- `vllm/v1/worker/gpu/cudagraph_utils.py`
  - `get_uniform_token_count(...)`
  - `CudaGraphManager` 根据 `decode_query_len` 初始化 graph candidates。
  - FULL graph 只用于 uniform decode 形状。
- `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`
  - speculator 为 prefill / decode drafter 分别初始化 cudagraph manager。
  - target verification、rejection sampling、draft proposal 是拆开的阶段，不强行塞进一个大 graph。
- `vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py`
  - drafter graph 的 capture/dispatch 是独立管理的。
- `vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py`
  - 固定 `num_query_per_req` 的 speculative query graph 是最接近 nano-vLLM pure spec verify graph 的参考形态。

对 nano-vLLM 最重要的映射是：

```text
vLLM uniform_decode_query_len = 1 + num_speculative_tokens
nano-VLLM spec verify query len = K + 1
```

也就是每个 decode seq 的 target verify 输入固定为：

```text
[last_token] + K draft tokens
```

#### 9.9.2 第一阶段范围

第一阶段只做 **pure spec decode-verify target graph**：

```text
适用条件：
  prefill_seqs 为空
  decode_seqs 非空
  每个 decode seq 都有完整 K 个 prev_draft_tokens
  not enforce_eager
  K 固定为 config.num_spec_tokens

graph 内容：
  target model forward with capture_layers
  输出 hidden_out 和 captured hidden states

不放进第一阶段 graph 的内容：
  lm_head / logits_selected
  accept/reject
  draft KV sync
  next draft proposal
```

理由：profile 显示 `verify≈1.6ms`、`draft≈4.5ms`，最大瓶颈是 target model forward。先 graph target forward 可以降低风险，并且和 vLLM 的阶段拆分一致：target graph、rejection、speculator proposal 分开处理。

#### 9.9.3 fallback 规则

保留当前 eager unified spec path 作为 fallback。以下情况不走 spec decode graph：

- 本轮有 `prefill_seqs`，包括 mixed prefill+decode。
- `self.enforce_eager=True`。
- `len(decode_seqs)` 超出 graph capture bucket。
- `input_ids.size(0) = len(decode_seqs) * (K + 1)` 超过 graph 支持上限。
- 后续引入 variable-K 或非 uniform query len。

第一阶段不处理 mixed graph。mixed step 语义上仍包含 decode verify，但实现上继续走当前 `set_context(has_prefill=True, ...)` 的 eager varlen attention 路径。

#### 9.9.4 graph key 和 bucket

新增 spec graph bucket，按 decode request 数量而不是 token 数管理：

```python
spec_graph_bs = [1, 2, 4, 8] + list(range(16, max_spec_bs + 1, 16))
spec_query_len = self.num_spec_tokens + 1
spec_num_tokens = graph_bs * spec_query_len
```

`max_spec_bs` 建议第一阶段保守设置为：

```python
max_spec_bs = min(
    config.max_num_seqs,
    config.max_num_batched_tokens // spec_query_len,
    512 // spec_query_len,
)
```

原因：当前普通 decode graph 已经在 `input_ids.size(0) > 512` 时 fallback eager。spec graph 第一阶段也先保持总 query tokens 不超过 512，避免一次性扩大 capture 风险。

#### 9.9.5 capture-time 静态 buffers

为 spec decode graph 单独维护 buffers，不能复用普通 one-token decode graph：

```python
self.spec_graph_vars = {
    "input_ids":      torch.zeros(max_spec_tokens, dtype=torch.int64),
    "positions":      torch.zeros(max_spec_tokens, dtype=torch.int64),
    "slot_mapping":   torch.zeros(max_spec_tokens, dtype=torch.int32),
    "cu_q":           torch.zeros(max_spec_bs + 1, dtype=torch.int32),
    "cu_k":           torch.zeros(max_spec_bs + 1, dtype=torch.int32),
    "block_tables":   torch.zeros(max_spec_bs, max_num_blocks, dtype=torch.int32),
    "outputs":        torch.zeros(max_spec_tokens, hidden_size),
    "captured":       {layer: torch.zeros(max_spec_tokens, hidden_size) for layer in eagle3_fuse_layers},
}
```

其中：

- `input_ids / positions / slot_mapping` 长度是 `graph_bs * (K + 1)`。
- `cu_q` 是固定 uniform pattern：`[0, K+1, 2(K+1), ...]`。
- `cu_k` 运行时更新，因为每个 seq 当前 context length 不同。
- `block_tables` 运行时更新。
- `outputs` 保存 target final hidden。
- `captured` 保存 EAGLE3 需要 fuse 的 target 中间层 hidden。

#### 9.9.6 capture 逻辑

新增类似 `capture_cudagraph()` 的函数，例如：

```python
def capture_spec_decode_cudagraph(self):
    ...
```

每个 `graph_bs` capture 一张 graph：

```python
num_tokens = graph_bs * (K + 1)

set_context(
    has_prefill=True,
    cu_seqlens_q=cu_q[:graph_bs + 1],
    cu_seqlens_k=cu_k[:graph_bs + 1],
    max_seqlen_q=K + 1,
    max_seqlen_k=config.max_model_len + K,
    p_slot_mapping=slot_mapping[:num_tokens],
    block_tables=block_tables[:graph_bs],
    num_prefill_tokens=num_tokens,
    num_decode_seqs=0,
    finishing_prefill_indices=[],
)

hidden, captured = self.model(
    input_ids[:num_tokens],
    positions[:num_tokens],
    capture_layers=self.eagle3_fuse_layers,
)
outputs[:num_tokens] = hidden
for layer in self.eagle3_fuse_layers:
    captured_buffers[layer][:num_tokens] = captured[layer]
```

虽然当前 graph 里仍通过 `has_prefill=True` 复用 varlen attention 实现路径，但这只是 nano-vLLM 当前的实现选择，不改变 spec verify 的 decode 语义。graph replay 可以先消除 Python/kernel launch 开销。后续如果要进一步接近 vLLM，可再把这条路径改成真正的 uniform decode attention backend。

#### 9.9.7 replay 逻辑

在 `run_speculative_step()` target forward 前判断是否可走 spec graph：

```python
use_spec_graph = (
    not prefill_seqs
    and decode_seqs
    and not self.enforce_eager
    and len(decode_seqs) <= self.max_spec_graph_bs
)
```

如果可以：

1. 仍复用 `_build_spec_target_batch([], decode_seqs)` 构造 rows 和运行时 metadata。
2. 选择 `graph_bs = next(x for x in self.spec_graph_bs if x >= len(decode_seqs))`。
3. 把实际 tensors 拷入 spec graph static buffers：

   ```python
   graph_vars["input_ids"][:num_tokens_actual] = input_ids
   graph_vars["positions"][:num_tokens_actual] = positions
   graph_vars["slot_mapping"][:num_tokens_actual] = slot_mapping
   graph_vars["cu_k"][:num_decode_seqs + 1] = cu_k
   graph_vars["block_tables"][:num_decode_seqs, :block_tables.size(1)] = block_tables
   ```

4. padded dummy rows 保持 capture-time 的安全默认值，输出会被忽略。
5. `graph.replay()`。
6. 返回：

   ```python
   hidden_out = graph_vars["outputs"][:num_tokens_actual]
   captured = {
       layer: graph_vars["captured"][layer][:num_tokens_actual]
       for layer in self.eagle3_fuse_layers
   }
   ```

后续 `logits_selected`、accept/reject、draft sync、next draft proposal 继续沿用现有逻辑。

#### 9.9.8 注意事项

1. **不能复用普通 decode graph**

   普通 decode graph 是每个 seq 一个 query token：

   ```text
   q_len = 1
   ```

   spec verify 是：

   ```text
   q_len = K + 1
   ```

   attention metadata、slot mapping、positions、target logits slicing 都不同，必须单独 capture。

2. **capture_layers 要一起进入 graph**

   EAGLE3 需要 target 中间层 hidden 做 fuse，所以 spec graph 不只返回 final hidden，还要保存 `eagle3_fuse_layers` 对应 captured tensors。

3. **第一阶段先 graph target forward，不 graph accept/reject**

   当前 `verify≈1.6ms`，不是主瓶颈。把 accept/reject 放进 graph 会增加复杂度，但收益有限。

4. **第一阶段先 graph target，不 graph draft**

   当前 `draft≈4.5ms`。即使 draft 全优化掉，也只能把 `48.5ms` 降到约 `44ms`，仍不够。target 是必须先解决的瓶颈。

5. **padded rows 必须安全**

   replay 时 graph 可能按 bucket 计算额外 dummy rows。dummy rows 的 block table、cu_k、slot mapping 要保持合法，不能访问非法 block；输出全部丢弃。

#### 9.9.9 验证计划

1. 语法检查：

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
   import ast
   from pathlib import Path
   for f in [
       'nanovllm/engine/model_runner.py',
       'benchmarks/bench_eagle3.py',
   ]:
       ast.parse(Path(f).read_text())
       print('OK', f)
   PY
   ```

2. correctness：

   ```bash
   python benchmarks/bench_eagle3.py \
     --model /root/repos/models/Qwen/Qwen3-4B \
     --draft-model /root/repos/models/AngelSlim/Qwen3-4B_eagle3 \
     --num-prompts 8 \
     --prompt-len 512 \
     --max-tokens 128 \
     --num-spec-tokens 3
   ```

   期望：

   ```text
   mismatched prompts: 0/8
   actual output toks == target output toks
   ```

3. profile：

   ```bash
   python benchmarks/bench_eagle3.py \
     --model /root/repos/models/Qwen/Qwen3-4B \
     --draft-model /root/repos/models/AngelSlim/Qwen3-4B_eagle3 \
     --num-prompts 8 \
     --prompt-len 512 \
     --max-tokens 128 \
     --num-spec-tokens 3 \
     --spec-profile
   ```

   重点观察：

   ```text
   last target=...
   avg target=...
   total=...
   ```

   当前 baseline 是 `last target≈41.8ms`，目标是显著下降。

4. fallback 验证：

   - 加 `--enforce-eager` 时不使用 graph。
   - mixed prefill+decode 仍走 eager unified spec path。
   - graph bucket 不覆盖的 batch size 正常 fallback。

#### 9.9.10 后续阶段

如果第一阶段 target graph 后仍不够快，再按 profile 继续：

1. 把 spec verify 从 `has_prefill=True + flash_attn_varlen_func` 改成真正的 uniform decode attention path。
2. 对 draft sync / one-token draft loop 做单独 FULL graph，参考 vLLM speculator graph 分离设计。
3. 将 accept/reject GPU 化，减少 `cpu().tolist()`，但这应在 target graph 后再做。
4. 最后再考虑 mixed prefill+decode 的 piecewise graph。

#### 9.9.11 首轮实现结果

已按第一阶段方案实现 pure decode-only spec verify target CUDA graph，并保持 mixed/prefill fallback eager。

实现要点：

- 新增 `ModelRunner.capture_spec_decode_cudagraph()`。
- graph query len 固定为：

  ```python
  q_len = self.num_spec_tokens + 1
  ```

- 只在 pure decode-only 条件下 replay：

  ```text
  prefill_seqs 为空
  decode_seqs 非空
  not enforce_eager
  batch size 在 captured graph bucket 内
  ```

- graph 只覆盖 target model forward + `capture_layers` 输出保存。
- `lm_head`、accept/reject、draft KV sync、next draft proposal 仍保持 eager 阶段处理。
- `--spec-profile` 增加 `mode=graph/eager`，用于确认当前 spec target forward 是否走 graph。

测试结果，`num_prompts=8, prompt_len=512, max_tokens=128, K=3`：

```text
Baseline (spec off):
  elapsed:             2.83s
  target output tok/s: 361.34
  engine steps:        128

Speculative decoding (spec on):
  elapsed:             1.77s
  target output tok/s: 578.20
  engine steps:        48
```

收益：

```text
elapsed speedup:     +37.5%
target tok/s change: +60.0%
```

`--spec-profile` 稳态结果：

```text
mode=graph
last target: 约 19~20ms
last total:  约 26~27ms
```

对比 graph 前：

```text
last target: 约 41.8ms -> 约 19~20ms
last total:  约 48.5ms -> 约 26~27ms
```

这说明前面的 profiling 判断正确：spec decoding 原本 step 数已经下降，但 target verify 虽然语义上是 decode，实现上仍走 eager + `has_prefill=True` 的 varlen attention 路径，导致单步过重。接入 pure spec decode graph 后，spec 的 step 数优势转化成了端到端速度收益，target verify 单步耗时下降超过一半。

correctness 仍需保持：

```text
mismatched prompts: 0/8
actual output toks == target output toks
```

后续优化方向：

1. 如果 profile 显示 `target` 仍占比较大，再考虑把 spec verify 从 `has_prefill=True + flash_attn_varlen_func` 改成更接近 vLLM uniform decode 的 attention path。
2. 如果 `draft_all` 成为主要瓶颈，再参考 vLLM speculator 的分离 graph 设计，为 draft sync / one-token draft loop 加 FULL graph。
3. mixed prefill+decode 暂时继续 eager，后续再考虑 piecewise graph。

### 9.10 Spec KV / block lifecycle 说明

当前 EAGLE3 spec 路径里，target KV 和 draft KV 是两块不同的 cache tensor，但共用同一套 `seq.block_table` / `slot_mapping` 作为逻辑位置映射。

```text
target model KV -> self.kv_cache
draft model KV  -> self.draft_kv_cache

共享的是：block_id / slot index / seq.block_table
不共享的是：KV 内容本身
```

原因：KV cache 保存的是各自模型 attention 计算得到的 K/V hidden states，依赖模型参数。target model 和 draft model 参数、层数和计算图不同，所以同一个 token position 上的 target KV 与 draft KV 不能互相复用。

当前代码中 target KV 和 draft KV 分别分配：

```python
self.kv_cache = torch.empty(
    2,
    hf_config.num_hidden_layers,
    config.num_kvcache_blocks,
    self.block_size,
    num_kv_heads,
    head_dim,
)

self.draft_kv_cache = torch.empty(
    2,
    1,
    config.num_kvcache_blocks,
    self.block_size,
    draft_num_kv_heads,
    draft_head_dim,
)
```

对当前 Qwen3-4B EAGLE3 checkpoint 来说，target/draft 的 per-layer KV shape 基本一致：

```text
num_key_value_heads = 8
head_dim = 128
per-layer KV shape = [num_blocks, block_size, 8, 128]
```

因此理论上可以把 draft KV 作为 target KV tensor 后面额外一层来组织，例如 `[2, target_layers + 1, ...]`。但这只是一种内存 layout 合并，不是语义共享，也不能省掉 target verify / draft proposal / append / rollback 逻辑。当前分开分配更通用：不要求 draft config 和 target config 的 KV shape 永远一致，语义也更清楚。

#### 9.10.1 `append_n_slots()` 与 `rollback_blocks()` 的关系

Spec 路径里 `append_n_slots()` 和 `rollback_blocks()` 基本是一对“临时扩容 / 裁回目标长度”的操作，但不是严格 undo。`rollback_blocks(seq, target_num_blocks)` 不知道刚才 append 了多少，只是把 `seq.block_table` 裁到当前阶段最终应该保留的 block 数。

`append_n_slots()` 会预分配 future token slots：

```python
self.block_manager.append_n_slots(seq, n)
```

它不会修改 `seq.num_tokens` / `seq.token_ids` / `len(seq)`，但如果 future positions 跨到新 block，会修改 `seq.block_table`：

```python
if block_idx >= len(seq.block_table):
    block_id = self.free_block_ids[0]
    self._allocate_block(block_id)
    seq.block_table.append(block_id)
```

因此更准确的 invariant 不是“`seq.block_table` 永远只包含 committed blocks”，而是：

```text
seq.block_table 允许在 spec verify / draft proposal 内部临时扩展；
但阶段结束后会 rollback 掉未验证、未接受的 speculative tail。
```

#### 9.10.2 Decode verify 前后的 block 生命周期

Decode verify row 输入是：

```text
[last_token] + K 个 prev_draft_tokens
```

其中 `last_token` 已经在 logical sequence 里，K 个 draft future positions 需要提前有 slot，所以 `_build_spec_target_batch()` 中会：

```python
self.block_manager.append_n_slots(seq, K)
```

target verify 后得到本轮接受的 tokens：

```python
accepted = row["accepted_tokens"]
M = row["num_accepted"]
```

然后根据 accepted 后的真实长度计算要保留的 block 数：

```python
final_len = original_len + M
final_num_blocks = (final_len + self.block_size - 1) // self.block_size
row["final_num_blocks"] = final_num_blocks
self.block_manager.rollback_blocks(seq, final_num_blocks)
```

这一步不是一定回到 `original_len`，而是回到 `original_len + M` 对应的 block 边界。也就是说：

```text
verify 前：临时预留 K 个 draft slots
verify 后：只保留 target 已接受的 M 个 token 需要的 blocks
```

如果 `M < K`，未接受 draft tail 对应的 extra blocks 会被释放；如果 accepted tokens 仍落在已有 block 内，rollback 可能不会释放任何 block。

#### 9.10.3 Draft sync 与下一轮 proposal 的 block 生命周期

Verify 后如果 seq 还没结束，会用 accepted tokens 做 draft sync：

```python
sync_input_ids.extend(accepted)
sync_positions.extend(range(original_len - 1, original_len - 1 + M))
for pos in range(original_len - 1, original_len - 1 + M):
    sync_slot_mapping.append(self._slot_for_position(seq, pos))
```

这里写的是 draft model 的 KV cache，但 slot 仍由同一套 `seq.block_table` 映射出来。`accepted` 本身已经是 target verify 得到的 next-token 序列，所以 decode 分支不需要像 prefill 那样再手动 `shifted = token_ids[start + 1:end]`。

sync 是一个 batch；proposal 也会整理成一个 batch，但只有下面两类 row 会进入 proposal：

```text
finishing prefill：prompt 已经算完，target sample 出第一个真实生成 token。
decode_verify：本轮 accept/reject 后 seq 还没结束。
```

partial prefill 只做 draft KV sync，不生成 `prev_draft_tokens`，也不会加入 `proposal_infos`。

当前 proposal metadata 用 `SpecProposalInfo` 表示：

```python
@dataclass
class SpecProposalInfo:
    seq: Sequence
    last_sync_index: int
    start_position: int
    rollback_num_blocks: int
```

其中 `start_position` 是下一轮 draft proposal 要写入 draft KV 的起点：

```text
finishing prefill: len(seq)
decode_verify:     original_len + num_accepted - 1
```

`_run_spec_proposal_batch()` 先从 sync 最后一行 logits 得到第一个 draft token：

```python
draft_token = sync_logits[last_indices].argmax(dim=-1)
first_draft_token = self.draft_model.d2t[draft_token]
```

如果 `K > 1`，后续 serial proposal 还要 forward `K - 1` 次，所以会一次性按每个 seq 的显式 `start_position` 预留这些 draft slots：

```python
draft_slots = [
    self.block_manager.append_n_slots(info.seq, K - 1, start_pos=info.start_position)
    for info in proposal_infos
]
```

也就是说，旧的 `first_slot + helper 内部 append 剩余 slots` 已经被替换为：

```text
proposal 开始时统一 ensure [start_position, start_position + K - 1) 的 slots
```

`_generate_draft_tokens_from_state()` 现在只消费调用方传进来的 `draft_slots`。当 `num_steps > 0` 时，如果没有传 `draft_slots`，应视为调用错误，而不是在 helper 内部隐式 append。

proposal 结束后：

```python
if rollback_num_blocks is not None:
    for seq, num_blocks in zip(seqs, rollback_num_blocks):
        self.block_manager.rollback_blocks(seq, num_blocks)
```

这里的 rollback 清理的是 `seq.block_table` 对 proposal 阶段临时 draft slots 的引用，不是把 `self.draft_kv_cache` 里的数据清零。旧 KV 数据可以留在 GPU memory 中；只要没有 slot mapping 再指向它，后续会被新 token 覆盖。

proposal 的长期产物是：

```python
seq.prev_draft_tokens = [draft_0, ..., draft_{K-1}]
```

而不是 proposal 阶段临时扩出来的 blocks。

#### 9.10.4 为什么 proposal rollback 后下一轮还要 append

从 block reservation 角度看，proposal 结束时 rollback、下一轮 verify 前再 `append_n_slots(seq, K)` 确实有冗余：

```text
proposal 阶段：append 临时 draft slots
proposal 结束：rollback 到 accepted 边界
下一轮 verify：再 append K 个 verify slots
```

这样做的主要收益是状态简单：`seq.block_table` 不长期保留未验证 draft tail，scheduler / block budget / preempt / reject / prefix-cache 逻辑都只需要面对“真实 accepted 状态 + 当前阶段临时扩展”。

另外中间并不是完全无变化：`run_speculative_step()` 返回后，scheduler 会在 postprocess 中提交 accepted tokens：

```python
seq.append_tokens(accepted_tokens)
```

所以下一轮 `append_n_slots(seq, K)` 是从新的 `seq.num_tokens = original_len + M` 开始预留 future slots，而不是从旧长度开始。

如果未来想消除这部分冗余，可以考虑保留 speculative reserved slots 到下一轮 verify 复用。但那需要额外 metadata 区分：

```text
committed_len
reserved_spec_len
哪些 slots 是 proposal 阶段写过的 draft KV
哪些 slots 可以被下一轮 target verify 复用
reject 后如何裁剪 speculative tail
preempt/deallocate 时如何释放
```

当前实现选择 correctness-first：只长期保存 `prev_draft_tokens`，不长期保存未验证 draft blocks。

### 9.11 DFlash 后续架构计划

如果后续实现 DFlash，不建议在现有 EAGLE3 `_generate_draft_tokens_from_state()` 里直接打补丁。DFlash 和 EAGLE3 的 proposal 语义不同，应该把 accept/reject 之后的 draft proposal 做成可插拔 speculator。

建议边界：

```text
target verify / accept-reject 是公共阶段
生成下一轮 prev_draft_tokens 是 speculator backend 阶段
```

也就是：

```text
EAGLE3 backend:
  accepted context draft sync
  first draft from sync logits
  serial proposal: d0 -> d1 -> ... -> dK-1

DFlash backend:
  accepted context precompute KV
  query = [bonus_token] + K 个 parallel_drafting_token
  一次 query forward 并行产生 K 个 draft tokens
```

#### 9.11.1 Speculator 抽象

建议引入：

```python
class Speculator:
    def propose(self, proposal_inputs):
        raise NotImplementedError
```

然后拆成：

```text
Eagle3Speculator
DFlashSpeculator
```

`ModelRunner.run_speculative_step()` 最终只保留公共流程：

```text
build target batch
run target forward
select logits
sample / verify
build proposal inputs
self.speculator.propose(...)
return scheduler result
```

当前 `ModelRunner` 里的这些 EAGLE3 细节后续可以逐步移动到 `Eagle3Speculator`：

```text
_build_spec_draft_sync_batch
_run_spec_draft_sync_and_propose
_run_spec_proposal_batch
_generate_draft_tokens_from_state
```

#### 9.11.2 Config backend

不要继续只用 `draft_model is not None` 隐式表示 EAGLE3。建议增加明确 backend：

```python
spec_method: str = "eagle3"  # "eagle3" / "dflash"
```

或：

```python
draft_backend: str = "eagle3"
```

初始化时：

```python
if config.draft_backend == "eagle3":
    self.speculator = Eagle3Speculator(...)
elif config.draft_backend == "dflash":
    self.speculator = DFlashSpeculator(...)
```

#### 9.11.3 DFlash proposal 数据流

DFlash proposal 不再使用：

```text
first_draft_token + current_hidden + serial loop
```

而是需要：

```text
accepted hidden states
accepted positions
accepted slot mapping
bonus token / last sampled token
num_sampled / num_rejected
block_table
```

DFlash backend 的核心流程：

```text
1. 用 accepted rows 的 hidden states 预计算 context KV。
2. 为每个 seq 构造 query：
   [bonus_token] + [parallel_drafting_token] * K
3. query positions：
   final_len, final_len + 1, ..., final_len + K
4. 预留这些 query positions 的 slots。
5. 跑 DFlash query forward。
6. 只从后 K 个 mask/query rows 采样 draft tokens。
7. 写入 seq.prev_draft_tokens。
8. rollback proposal 阶段临时 blocks。
```

注意 DFlash 和 EAGLE3 的 slot 数不同：

```text
EAGLE3 proposal forward 次数: K - 1
DFlash query forward 行数:   K + 1
```

所以 DFlash proposal 应预留：

```python
query_slots = self.block_manager.append_n_slots(
    seq,
    K + 1,
    start_pos=proposal_start,
)
```

#### 9.11.4 Attention causal / non-causal

vLLM DFlash 支持从 draft config 读取 causal / non-causal。nano-vLLM 当前 attention 调用基本写死 `causal=True`。如果 DFlash checkpoint 需要 non-causal query attention，需要把 `causal` 放进 context：

```python
set_context(..., causal=config.dflash_causal)
```

然后 attention backend 使用：

```python
causal=context.causal
```

第一阶段可以保守只支持 causal DFlash，确认 checkpoint 需求后再扩展 non-causal。

#### 9.11.5 落地顺序

建议分阶段做：

```text
Phase 1: 抽象 Speculator，但 Eagle3 行为不变。
Phase 2: 增加 DFlashDraftModel wrapper 和 backend config。
Phase 3: 实现 DFlash eager proposal，只先覆盖 pure decode。
Phase 4: 支持 finishing prefill -> DFlash proposal。
Phase 5: mixed prefill+decode 继续 fallback eager，确认正确性。
Phase 6: 为 DFlash fixed query forward 加 CUDA graph。
Phase 7: 再把 DFlash input preparation GPU/Triton 化。
```

第一版不要直接上 Triton input-preparation kernel。先用 Python 构造 DFlash inputs，验证 token 对齐、slot mapping、rollback 和 `prev_draft_tokens` 正确，再做 GPU 化。

## 10. Unified spec postprocess

新增 `scheduler.postprocess_speculative_step(result)`，用于处理 `run_speculative_step()` 的结构化结果。

### Prefill updates

- `seq.num_computed_tokens += seq.scheduled_chunk_size`
- 如果 prefill 完成，append sampled token。
- 检查 EOS / max_tokens，必要时 finish + deallocate。

### Decode updates

- append accepted token list。
- 检查 EOS / max_tokens，必要时 finish + deallocate。

重要：spec postprocess 不主动清空 `prev_draft_tokens`；该字段由 model runner 的 draft KV sync / 生成下一轮 drafts 逻辑负责设置或清空。

## 11. 实施顺序

1. 加 `BlockManager.can_append_n()` / `num_extra_blocks_for_append()`。
2. 让 scheduler 的 spec decode cost 变成真实 target token 数。
3. 在 `model_runner.py` 加 row metadata 和 target batch builder。
4. 实现 `run_speculative_step()`：统一入口，一次 target forward + `forward_all()`。
5. 实现 row-based output interpretation。
6. 实现 unified draft KV sync / 生成下一轮 drafts，重点保留 finishing prefill 的 predicted drafts 和 metadata 处理。
7. 实现 `scheduler.postprocess_speculative_step()`。
8. 接入 `LLMEngine.step()`：外层只分 `has_spec` / 非 `has_spec`。
9. 跑 correctness / chunked prefill / mixed spec tests。

## 12. 验证计划

### 现有测试

```bash
python tests/verify_spec_correctness.py \
  --model models/Qwen3-4B \
  --draft-model models/Qwen3-4B_eagle3 \
  --max-tokens 64

python tests/verify_chunked_prefill.py
```

### 新增 unified spec 测试

- 开启 `draft_model` 和 `enable_chunked_prefill=True`。
- 分别覆盖：
  - prefill-only step
  - decode-only step
  - prefill + decode mixed step
- 设置较小 `max_num_batched_tokens`，强制出现 mixed step。
- prompt 长短不一，让短 prompt decode 时长 prompt 仍在 chunking。
- `temperature=0`，对比 spec 与 non-spec token-for-token 完全一致。

### Metadata 检查

- partial prefill 不设置 `prev_draft_tokens`。
- finishing prefill 设置恰好 K 个 `prev_draft_tokens`，除非 EOS/max_tokens。
- verify row rejected tail 不进入 draft sync。
- active seq 每轮结束后 `prev_draft_tokens` 长度保持 K。

### 资源检查

- target token count 不超过 `max_num_batched_tokens`。
- spec temporary blocks 正常 rollback，不出现 block_table 持续增长。
- finished seq 正常 deallocate。

## 13. 预期结果

```text
Before:
  has_spec + decode      → run_speculative()
  has_spec + prefill     → run()
  has_spec + mixed       → run_speculative() + run()

After:
  has_spec               → run_speculative_step()
  not has_spec           → run()
```

这样 engine 外层不再区分 prefill/decode/mixed，只区分 spec 开或关；内部通过 row metadata 一次性完成 target forward、draft KV sync / 生成下一轮 drafts 和 postprocess。

## 14. Spec decoding 的并发收益规律

总结规律：

- 低并发 / 延迟敏感场景（在线 serving，1-8 并发）：spec decoding 收益大，常见提升约 +45%~85%。
- 高并发 / 吞吐场景（离线 batch 推理，64+ 并发）：GPU 已经饱和，spec decoding 通常无明显收益，甚至可能略慢。

核心原因：**spec decoding 主要减少的是串行 decode 轮数，而不是显著减少总计算量**。它在 GPU 未被充分利用时收益明显；当 GPU 已经被 batching 打满时，额外开销会抵消收益。

### 14.1 低并发为什么收益大

在线 serving、1-8 并发时，decode 通常是每次只生成一个 token：

```text
target model forward
sample one token
target model forward
sample one token
...
```

这类场景下 GPU 往往没有吃满：

- batch 小，矩阵乘规模小；
- kernel launch / 调度开销占比高；
- 每步只产一个 token，串行依赖强；
- latency 主要来自 decode step 数量。

Spec decoding 让 draft model 先猜多个 token，再让 target model 一次验证：

```text
draft model proposes K tokens
target model verifies K tokens in one forward
accepted tokens advance multiple decode steps
```

因此它把接近 `N` 次 target forward 压缩成约 `N / accepted_tokens_per_step` 次 target forward，再额外付出 draft model 开销。低并发时 target model 还没把 GPU 跑满，draft 开销相对便宜，所以端到端延迟可以明显下降。

### 14.2 高并发为什么收益小

高并发、batch 推理时，每个 decode step 已经包含大量序列：

```text
batch = 64 / 128 / more
```

这时普通 decoding 已经通过 batching 把 GPU 吃满：

- SM occupancy 高；
- memory bandwidth 接近上限；
- matmul 规模足够大；
- 每次 target forward 的单位 token 成本已经较低。

Spec decoding 虽然减少了一些 target decode 轮数，但也引入额外开销：

- draft model forward；
- target verification 的额外 token 维度；
- accept/reject 逻辑；
- KV cache 写入、rollback 和同步；
- scheduler / metadata 处理复杂度；
- 接受率不够高时还会浪费 draft token。

所以高并发时经常变成：

```text
省下了一些 target decode 轮数
但增加了 draft + verify + 调度开销
GPU 仍然满载，tokens/s 不明显提升
```

### 14.3 结论

Spec decoding 解决的是低并发 decode 的串行延迟问题；高并发下 batching 已经把串行开销摊薄并把 GPU 打满，所以 spec decoding 的额外开销容易抵消收益。

```text
低并发：瓶颈 = decode 轮数 / latency
spec decoding 有用

高并发：瓶颈 = GPU 吞吐 / 显存带宽
spec decoding 不一定有用
```
