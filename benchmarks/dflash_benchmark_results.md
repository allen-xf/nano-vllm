# DFlash Speculative Decoding 性能对比

## 测试环境
- GPU: NVIDIA A10 (24GB)
- Model: Qwen3-4B (target)
- Draft Model: z-lab/Qwen3-4B-DFlash-b16 (block diffusion drafter)
- spec_method: dflash
- num_spec_tokens (K): 5
- prompt_len: 512, max_tokens: 128, num_prompts: 16

运行命令：
```bash
python benchmarks/bench_eagle3.py \
    --model /root/llm/model/Qwen3-4B \
    --draft-model /root/llm/model/Qwen3-4B-DFlash-b16 \
    --spec-method dflash \
    --num-prompts 16 --prompt-len 512 --max-tokens 128
```

---

## DFlash 是什么

DFlash 是一种基于 **block diffusion** 的投机解码 drafter：
- 与 EAGLE3 的自回归串行 draft（K 次 forward）不同，DFlash 用 `[bonus] + mask_token * K` 一次并行 forward 出 K 个 draft token
- draft 开销更低、draft 质量更高，适合大 batch
- 本模型为 drafter 组件，须搭配 target 模型 `Qwen/Qwen3-4B` 使用

---

## 结果演进（三轮）

同一命令、同一硬件下的三轮结果，完整还原了从 bug 到优化的过程：

| 阶段 | elapsed | engine steps | target tok/s | speedup | 说明 |
|------|---------|--------------|--------------|---------|------|
| ① lm_head bug（原始） | 7.20s | 128 | 284 | **-144%** | draft 全被拒，反而更慢 |
| ② 修复 lm_head（eager） | 1.63s | 23 | 1258 | +44.4% | draft 正常，接受率恢复 |
| ③ + CUDA graph | **1.33s** | 23 | **1544** | **+54.6%** | 消除 kernel launch 开销 |

---

## 最终结果（阶段 ③：修复 + CUDA graph）

### Baseline (spec off)
| 指标 | 值 |
|------|-----|
| elapsed | 2.92s |
| requests/s | 5.47 |
| output tok/s | 700.59 |
| output tokens | 2048 |
| engine steps | 128 |
| prefill tokens | 4200 |
| decode tokens | 2032 |
| pure prefill steps | 1 |
| pure decode steps | 127 |

### Speculative decoding (spec on)
| 指标 | 值 |
|------|-----|
| elapsed | 1.33s |
| requests/s | 12.06 |
| output tok/s | 1543.87 |
| output tokens | 2048 |
| engine steps | 23 |
| prefill tokens | 4200 |
| decode tokens | 2032 |
| pure prefill steps | 1 |
| pure decode steps | 22 |

### 对比
| 指标 | 值 |
|------|-----|
| elapsed speedup | +54.6% |
| tok/s improvement | +120.4% |
| steps reduction | 128 → 23 (82% fewer) |
| avg acceptance length | 2032 / 22 / 16 ≈ **5.77 tokens/seq/step** |
| 正确性 | mismatched 0/16 (greedy 无损) |

---

## 根因分析：lm_head 未共享导致零加速

### 现象
最初运行 DFlash，`engine steps = 128` 与 baseline 完全一致，投机完全没生效，反而因 draft 开销 elapsed 慢 144%。但输出正确（0/16 mismatch）。

### 根因
- DFlash checkpoint（`model.safetensors`）只包含 `fc / hidden_norm / norm / layers.0~4.*`，**不含 `lm_head` 和 `embed_tokens`**
- draft config `tie_word_embeddings: true`，意味着 lm_head 应共享 target 模型的 lm_head
- 但 `DFlashSpecBackend.__init__` 原先只共享了 `embed_tokens`，**漏掉了 lm_head**
- `DFlashQwen3ForCausalLM.__init__` 里 `self.lm_head = ParallelLMHead(...)` 是随机初始化，`load_dflash_model` 因 checkpoint 无 lm_head 也不加载
- 结果：draft 用**随机 lm_head** → `argmax` 输出垃圾 token → verify 阶段几乎全部拒绝 → acceptance≈1
- verify 用的是 target lm_head，所以最终输出仍正确，掩盖了 bug

### 修复
在 `nanovllm/engine/spec_decode/dflash.py` 的 backend 初始化中，共享 embed_tokens 之后追加共享 target lm_head：

```python
if not load_info.includes_lm_head:
    target_lm_head = self.model.lm_head
    draft_lm_head = self.draft_model.lm_head
    if target_lm_head.weight.shape != draft_lm_head.weight.shape:
        raise RuntimeError("DFlash checkpoint omitted lm_head but target/draft lm_head are incompatible")
    self.draft_model.lm_head = target_lm_head
```

target/draft 的 hidden_size(2560)、vocab(151936) 完全一致，直接共享即可。

---

## CUDA graph 优化（阶段 ③）

复用 EAGLE3 的 target-verify CUDA graph 逻辑：
- `_run_spec_target_forward` 在 `use_spec_graph=True` 时走 `_run_spec_decode_graph`
- `capture_decode_cudagraph` 直接 `super().capture_decode_cudagraph()`
- `eagle3_fuse_layers = target_layer_ids`，复用 EAGLE3 的层捕获路径

**收益**：engine steps 保持 23 不变（graph 不改变接受率），仅消除 kernel launch 开销，elapsed 1.63s → 1.33s，tok/s 1258 → 1544（+23%）。decode 阶段 batch 小（16 seq × 6 token/step），每步小 kernel launch 开销占比高，graph replay 收益明显。

---

## 与 EAGLE3 对比（同环境同参数）

| 指标 | EAGLE3 | DFlash（阶段③） |
|------|--------|-----------------|
| engine steps | 43 | **23** |
| avg acceptance length | ≈ 3.0 | **≈ 5.77** |
| elapsed speedup | +34.3% | **+54.6%** |
| tok/s improvement | +52.3% | **+120.4%** |

DFlash 的 block diffusion 并行 drafting 质量显著高于 EAGLE3 的自回归 draft：K=5 时接近理论上限（5 draft 几乎全中 + 1 bonus ≈ 6），与官方 README "比 EAGLE-3 快约 2.5x" 的方向一致。
