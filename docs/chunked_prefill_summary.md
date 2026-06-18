# Chunked Prefill 实现总结

> 对应 commit: `8c82a99` — feat: implement chunked prefill with mixed batch support

## 核心设计思想

长 prompt 不再一次算完，而是切成小块（chunk），每步和 decode token 混在一起处理，防止长 prefill 阻塞 decode 请求，提高 GPU 利用率。调度策略变为 decode 优先（保证已有请求的延迟），剩余算力填充 prefill chunk。

## 各模块改动

### 1. `config.py` — 配置开关
- 新增 `enable_chunked_prefill` 配置项（默认 `True`）
- 去掉了 `max_num_batched_tokens >= max_model_len` 的断言（chunked 模式下不再需要一次装下整个 prompt）

### 2. `sequence.py` — 序列状态扩展
- 新增 `num_computed_tokens`（已计算的 token 数）和 `scheduled_chunk_size`（本轮调度的 chunk 大小）
- 新增属性 `is_prefill`（是否还在 prefill 阶段）和 `num_uncomputed_tokens`
- `append_token` 时同步更新 `num_computed_tokens`

### 3. `scheduler.py` — 核心调度逻辑重写
- **非 chunked 模式** (`_schedule_non_chunked`)：保持 prefill 优先、prefill/decode 互斥的原有逻辑
- **Chunked 模式** (`_schedule_chunked`)：
  - **Decode 优先**：先调度 running 队列中的 decode seq（保证低延迟）
  - **Prefill 切片**：用剩余 budget 给 prefill seq 分配 chunk，一个 seq 可以跨多步完成
  - **混合 batch**：prefill 和 decode 可以在同一 step 中执行
- `schedule()` 返回值从 `(seqs, is_prefill)` 变为 `(prefill_seqs, decode_seqs)`
- `postprocess` 适配：只有完成 prefill 的 seq 才追加生成的 token
- 新增可选的 metrics 收集（统计 mixed/pure 步数、GPU 利用率）

### 4. `model_runner.py` — 输入准备与执行
- `prepare_prefill` 改为按 `[num_computed_tokens, num_computed_tokens + chunk_size)` 切片准备输入，而非整个 prompt
- slot_mapping 的计算也相应调整，只映射本轮 chunk 对应的 slot
- `run()` 方法重写：
  - 分别调用 `prepare_prefill` 和 `prepare_decode`
  - **拼接** prefill 和 decode 的 input_ids/positions
  - 识别本轮**完成 prefill** 的 seq（`finishing_prefill_indices`），只对它们和 decode seq 做采样
  - 通过 `set_context` 传入 decode 专属字段（`decode_slot_mapping`, `decode_context_lens`, `decode_block_tables`）

### 5. `attention.py` — 注意力层适配混合 batch
- 按 `num_prefill_tokens` 将 Q/K/V **切分**为 prefill 和 decode 两部分
- Prefill 部分：用 `flash_attn_varlen_func`（变长 attention）
- Decode 部分：用 `flash_attn_with_kvcache`（KV cache attention）
- 分别写入各自的 KV cache slot，最后 `torch.cat` 拼接输出

### 6. `embed_head.py` — LMHead 选取 logits
- 不再取所有 prefill seq 的 last token，而是只取**完成 prefill** 的 seq 的 last token（通过 `finishing_prefill_indices`）
- 加上所有 decode seq 的 token，拼成需要采样的 logits

### 7. `context.py` — Context 扩展字段
- 新增 `num_prefill_seqs`, `num_prefill_tokens`, `num_decode_seqs`
- 新增 `finishing_prefill_indices`
- 新增 decode 专属字段：`decode_slot_mapping`, `decode_context_lens`, `decode_block_tables`
- `set_context` 支持 `**kwargs` 传入扩展字段

### 8. `llm_engine.py` — 引擎层适配
- `step()` 返回 `(outputs, num_prefill_tokens, num_decode_tokens)` 替代原来用正负号区分的方式
- 吞吐量统计同时支持 prefill 和 decode（混合 batch 时两者可以同时非零）
- `exit()` 增加防御性检查和 `gc.collect()`
