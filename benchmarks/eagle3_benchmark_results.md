# EAGLE3 Speculative Decoding 性能对比

## 测试环境
- GPU: NVIDIA A10 (24GB)
- Model: Qwen3-4B
- Draft Model: AngelSlim/Qwen3-4B_eagle3
- num_spec_tokens: 5
- prompt_len: 512, max_tokens: 128, num_prompts: 16

---

## nano-vllm 结果

### Baseline (spec off)
| 指标 | 值 |
|------|-----|
| elapsed | 3.35s |
| requests/s | 4.78 |
| output tok/s | 612.06 |
| output tokens | 2048 |
| engine steps | 128 |
| prefill tokens | 4200 |
| decode tokens | 2032 |
| pure prefill steps | 1 |
| pure decode steps | 127 |

### Speculative decoding (spec on)
| 指标 | 值 |
|------|-----|
| elapsed | 2.20s |
| requests/s | 7.28 |
| output tok/s | 932.01 |
| output tokens | 2048 |
| engine steps | 43 |
| prefill tokens | 4200 |
| decode tokens | 2032 |
| pure prefill steps | 1 |
| pure decode steps | 42 |

### 对比
| 指标 | 值 |
|------|-----|
| elapsed speedup | +34.3% |
| tok/s improvement | +52.3% |
| steps reduction | 128 → 43 (66% fewer) |
| avg acceptance length | 2048/42 ≈ **3.0 tokens/step** |

---

## 分析

### Acceptance Length 估算
- Baseline: 127 个 decode step，batch=16，每步每 seq 产 1 token → 127×16 = 2032 tokens
- Spec: 42 个 decode step，batch=16，同样产出 2032 tokens
- Spec 平均每步每 seq: 2032 / 42 / 16 ≈ **3.0 tokens/seq/step**
- 即 avg acceptance length ≈ 3.0（K=5 时，平均接受 2 个 draft + 1 个 bonus/correction）
- 步数减少: 127 → 42（减少 67%）

### 性能瓶颈
1. Draft model accuracy: acceptance rate ~60% (3/5 tokens accepted on avg)
2. 理论最大加速 (K=5): 如果全部 accept 则 6x per step
3. 实际达到 3.0x，说明接受率约 50-60%

---

## vLLM 对比 (TODO)
- vLLM 0.23.0 + Qwen3-4B + Eagle3 的 acceptance rate 待补充
- 预期 vLLM 的 acceptance rate 应显著更高（正确实现下应达 70-80%）
