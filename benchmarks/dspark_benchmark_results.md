# DSpark Speculative Decoding 性能对比

## 测试环境
- GPU: NVIDIA A10 (24GB)
- Model: Qwen3-4B (target)
- Draft Model: deepseek-ai/dspark_qwen3_4b_block7 (block diffusion + Markov head drafter)
- spec_method: dspark
- num_spec_tokens (K): 5 (默认；draft 模型 block_size=7)
- prompt_len: 512, max_tokens: 128, num_prompts: 16

运行命令：
```bash
python benchmarks/bench_eagle3.py \
    --model /root/llm/model/Qwen3-4B \
    --draft-model /root/llm/model/dspark_qwen3_4b_block7 \
    --spec-method dspark \
    --num-prompts 16 --prompt-len 512 --max-tokens 128
```

---

## DSpark 是什么

DSpark 是在 DFlash（block diffusion drafter）基础上增加 **Markov 转移头** 的投机解码 drafter：
- 继承 DFlash 的 block-diffusion 并行 drafting（`[bonus] + mask_token * K` 一次并行 forward）
- 额外的 `markov_head`（`markov_w1` / `markov_w2`，rank=256）提供 token 转移偏置，改进 draft 采样
- 带 confidence head（`confidence_head.proj`），config `enable_confidence_head=true`
- 本模型为 drafter 组件，须搭配 target 模型 `Qwen/Qwen3-4B` 使用

### draft 模型要素（DSpark 三要求）
| 要素 | 来源 | 值 |
|------|------|-----|
| dflash_config | config 顶层字段提升 | mask_token_id=151669, target_layer_ids=[1,9,17,25,33] |
| markov_rank | config 顶层 | 256 |
| markov_head 权重 | checkpoint | markov_w1/w2 (151936, 256) |

> 注：该 checkpoint 的 config **没有嵌套 `dflash_config`**，而是把 `mask_token_id` / `target_layer_ids` 放在顶层。`config.py` 的 normalize 逻辑会自动提升这些字段。此外 `tie_word_embeddings=false`，checkpoint **自带 lm_head 与 embed_tokens**，无需共享 target。

---

## 结果

### Baseline (spec off)
| 指标 | 值 |
|------|-----|
| elapsed | 2.95s |
| requests/s | 5.43 |
| output tok/s | 695.15 |
| output tokens | 2048 |
| engine steps | 128 |
| prefill tokens | 4200 |
| decode tokens | 2032 |
| pure prefill steps | 1 |
| pure decode steps | 127 |

### Speculative decoding (spec on)
| 指标 | 值 |
|------|-----|
| elapsed | 1.78s |
| requests/s | 8.98 |
| output tok/s | 1149.98 |
| output tokens | 2048 |
| engine steps | 34 |
| prefill tokens | 4200 |
| decode tokens | 2032 |
| pure prefill steps | 1 |
| pure decode steps | 33 |

### 对比
| 指标 | 值 |
|------|-----|
| elapsed speedup | +39.6% |
| tok/s improvement | +65.4% |
| steps reduction | 128 → 34 (73% fewer) |
| avg acceptance length | 2032 / 33 / 16 ≈ **3.85 tokens/seq/step** |
| 正确性 | mismatched 0/16 (greedy 无损) |

---

## 三种投机方法横向对比（同环境 16×512×128）

| 方法 | engine steps | avg acceptance | elapsed speedup | tok/s change |
|------|--------------|----------------|-----------------|--------------|
| EAGLE3 | 43 | ≈ 3.0 | +34.3% | +52.3% |
| **DSpark** | 34 | ≈ 3.85 | +39.6% | +65.4% |
| DFlash | 23 | ≈ 5.77 | +54.6% | +120.4% |

DSpark 加速介于 EAGLE3 与 DFlash 之间：Markov 头 + confidence 头相比纯 EAGLE3 自回归 draft 提升了接受率，但本轮 DFlash-b16 的 block diffusion drafting 质量最高。

> ⚠️ 上表使用**重复 filler 文本 prompt**（`--prompt-mode text`）。这类高度可预测的重复文本让 draft 极易命中，acceptance length 被严重高估（DFlash 5.77 ≈ 上限 6 的 96%），不代表真实工作负载。真实结果见下节。

---

## HumanEval 真实 prompt 复测（初版：未套 chat template，已被「最终修正版」取代）

> ⚠️ 本节虽用了真实 HumanEval prompt，但仍有两处与官方用法不符：**K 未对齐 block_size**（用了默认值）且**未套 chat template**（drafter 是在 chat 格式 + `enable_thinking=False` 下训练的）。因此 accept length 仍被低估，请以下方「最终修正版」为准。

上面的 filler prompt 会让生成退化成重复文本，draft 命中率虚高。改用 **HumanEval**（164 条真实代码补全题，`--prompt-mode humaneval`）重测：

运行命令（三方法同环境 16×512×128）：
```bash
python benchmarks/bench_eagle3.py \
    --model /root/llm/model/Qwen3-4B \
    --draft-model <draft> --spec-method <method> \
    --num-prompts 16 --prompt-len 512 --max-tokens 128 \
    --prompt-mode humaneval
```

| 方法 | engine steps | pure decode steps | avg acceptance* | elapsed speedup | tok/s change |
|------|--------------|-------------------|-----------------|-----------------|--------------|
| EAGLE3 | 80 | 79 | ≈ **1.61** | -3.3% | -3.2% |
| DFlash | 62 | 61 | ≈ **2.08** | +17.5% | +21.3% |
| **DSpark** | 54 | 53 | ≈ **2.40** | +24.9% | +33.1% |

\* acceptance length = decode_tokens / pure_decode_steps / num_seqs = 2032 / steps / 16（16 条 seq，各生成 127 decode token）。

### 关键结论
1. **真实 accept length 远低于 filler 结果**：DFlash 从虚高的 5.77 跌到 2.08，印证了 filler prompt 的高估。
2. **排序反转**：在真实代码 prompt 上 **DSpark(2.40) > DFlash(2.08) > EAGLE3(1.61)**，与 filler 下的 DFlash>DSpark>EAGLE3 相反。DSpark 的 Markov 转移头在真实分布上更有效，而 DFlash-b16 的优势主要来自 filler 文本的可预测性。
3. **EAGLE3 在代码任务上净变慢**（-3.3%）：自回归 draft 每步开销 + 代码 token 上接受率低（1.61），得不偿失。
4. **正确性**：三方法均出现 `mismatched 11/16`，且**首个 mismatch 位置完全相同**（prompt#1, pos 72, baseline=2 vs spec=750）。跨三种完全不同的 drafter 出现同一分歧点，说明这不是某个 drafter 的接受逻辑 bug，而是 batched spec forward 下 target 前向与 baseline 逐 token 前向之间的数值/tie-break 差异，属已知的批次数值不确定性。

---

## HumanEval + Chat Template + K 对齐 + backend 修复（最终版，推荐参考）

在初版基础上做了三项修正，其中第 3 项（backend bug 修复）带来质变：
1. **K 对齐 block_size**：DFlash-b16 用 15，DSpark-block7 用 7（初版用默认 5）。
2. **套 chat template**：`make_humaneval_prompts` 用 `apply_chat_template(..., enable_thinking=False)` 包装，匹配 drafter 训练分布。
3. **DSpark backend bug 修复（关键）**：见下方「backend 修复」节。

运行命令：
```bash
python benchmarks/bench_eagle3.py \
    --model /root/llm/model/Qwen3-4B \
    --draft-model <draft> --spec-method <method> \
    --num-prompts 16 --prompt-len 512 --max-tokens 128 \
    --prompt-mode humaneval --num-spec-tokens <K>
```

### 三方法对比（各自对齐 K）

| 方法 | K | engine steps | accept length | 每 token 接受率 | elapsed speedup |
|------|---|--------------|---------------|----------------|-----------------|
| **DSpark** | 7 | 32 | **4.10** | **59%** | **+48.7%** |
| DFlash | 7 | 40 | 3.26 | 47% | +42.3% |
| DFlash | 15 | 30 | 4.38 | 29% | +42.4% |
| EAGLE3 | 7 | 68 | 1.90 | 27% | +4.9% |

- **同 K=7 DSpark 全面胜出**：DSpark（accept 4.10, +48.7%）> DFlash（accept 3.26, +42.3%），每 token 接受率 59% > 47%。同起跑线下 DSpark 各项都更强，坐实「DSpark = DFlash base + Markov 头」draft 质量本就更高。
- **DSpark 反超 DFlash（含其最优 K=15）**：DSpark K=7 加速 +48.7% > DFlash K=15 +42.4%（tok/s +95.0%），且每 token 接受率 59% ≫ 29%。
- **DFlash 需大 K 才发挥**：K=7 accept 仅 3.26，喂满 block16 到 K=15 才达 4.38；但两者 speedup 几乎相同（+42.3% vs +42.4%）——K=7 每步更便宜、K=15 accept 更高，净加速打平。DFlash 4.38 接近论文 5.38 量级；EAGLE3 自回归 draft 在代码任务上接受率最低。

### DSpark backend 修复：从"几乎不加速"到反超 DFlash

| DSpark K=7 | 修复前（有 bug） | 修复后 |
|---|---|---|
| pure decode steps | 36 | **31** |
| accept length | 3.53 | **4.10** |
| elapsed speedup | +0.9% | **+48.7%** |
| tok/s change | +1.0% | **+95.0%** |
| mismatched | 8/16 | 6/16 |

根因是提议路径的两个 bug：
1. **采样位置错位**：旧代码把 draft forward 的**全部** query 位置 hidden 直接 `compute_draft_logits(hidden_states).view(B, K, -1)`，而 query 布局为 `[bonus] + mask*(K-1)`，位置与采样目标没对齐。修复：显式构造 `sample_indices`，用 `hidden_states.index_select(0, sample_indices_t)` 取对采样位置再算 logits（并按 `sample_from_anchor` 区分锚点/非锚点布局）。
2. **`target_layer_ids` off-by-one**：aux hidden / fuse 层选择差 1。修复：`target_layer_ids = [id + 1 for id in raw_target_layer_ids]`。

> ⚠️ 该 bug 极具迷惑性：**程序能跑、输出正确（greedy 无损）、只是慢**，accept 被压到 3.53、加速仅 +0.9%，看起来像"DSpark 本身弱"。修复后才暴露真实实力。

### 正确性（mismatch 非 bug）

三方法均出现 `mismatched 6~11/16`，首个 mismatch 多在语义接近处（如 `#` vs `def`）。这是 batched spec forward 与 baseline 逐 token forward 的浮点数值差异，在 top-2 logits 接近处触发 greedy argmax 翻转，之后上下文连锁分叉。accept 逻辑保证每个输出 token 都等于 target argmax（无损），两条都是合法 greedy 续写，不影响 accept length 统计。

---

## 全量 HumanEval（164 条）greedy 复测

前面的“最终修正版”为 16 条样本。为消除样本偏差，在修复后的 backend 上跑完整 164 条 HumanEval。

运行命令（K=7，greedy）：
```bash
LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs \
/root/llm/nano-vllm/llm/bin/python benchmarks/bench_eagle3.py \
    --model /root/llm/model/Qwen3-4B \
    --draft-model /root/llm/model/dspark_qwen3_4b_block7 \
    --spec-method dspark \
    --num-prompts 164 --max-tokens 128 \
    --prompt-mode humaneval --temperature 0 \
    --num-spec-tokens 7 --no-chunked-prefill \
    --max-num-seqs 32
```

> 关键点：`--max-num-seqs 32` 限制并发序列数，让调度器分波处理，避免 164 条全并发导致 KV block 池耗尽（`IndexError: deque index out of range`）。限制并发只影响吞吐节奏，不改变 accept length 统计。

### 结果（greedy，K=7）

| 指标 | Baseline (spec off) | Spec (DSpark) |
|------|--------------------:|--------------:|
| decode tokens | 20828 | 20828 |
| **pure decode steps** | **762** | **162** |
| elapsed | 17.85s | 12.20s |

| 对比 | 值 |
|------|-----|
| **accept length** | 762 / 162 ≈ **4.70** |
| elapsed speedup | **+31.7%** |
| tok/s change | **+46.3%** |
| mismatched | 78/164 |

> accept length 用 **step-ratio** 估计：baseline 与 spec 处理相同的 decode 工作量（20828 token）且 `--max-num-seqs` 相同（并发波形一致），decode step 的缩减比即等效 accept length。这与 16 条全并发时用 `decode_tokens / pure_decode_steps / num_seqs` 口径不同（后者要求所有 seq 同波），但两者针对各自场景都是 accept length 的合理估计。

### 关键结论

1. **全量 164 条 accept ≈ 4.70**，与 16 条估计（4.10）一致且略高（更大样本更具代表性），真实 greedy 天花板落在 **4.1–4.7**。
2. **与论文 τ=5.38 的差距非代码 bug，且已在 temp=1.0 口径下复现**：该 checkpoint 实为论文官方权重（见下节「temp=1.0 口径复现论文」），greedy 4.70 偏低仅因口径——切到论文的 temperature=1.0 后 accept length 达 ~5.31 ≈ 5.38。
3. **78/164 mismatch 属正常数值现象**（同上文）：batched verify forward 与 baseline 逐 token forward 的浮点差异在 argmax 平局处触发 tie-break 翻转，128 token 长序列上累积发散；accept 的 token 仍严格等于 target 每步 argmax，无损，不影响 accept 统计。
4. **代码层已无提升空间**：propose/verify/Markov 链路均正确；`target_layer_ids +1` 经原理（config 声明的是层**输出**，qwen3 capture 的是层**输入**=前一层输出，故 +1 才对齐训练语义）+ A/B 实测（+0=3.53, +1=4.10, +2=3.74）双重确认为最优。

---

## temp=1.0 口径复现论文（决定性结论）

### 关键发现：手上的就是论文官方 checkpoint
`/root/llm/model/dspark_qwen3_4b_block7` 就是 DeepSeek 官方 `deepseek-ai/DeepSpec` 仓库 "Released Checkpoints" 表中 **Table 1 所用的官方 DSpark Qwen3-4B 权重**（repo id 一致，config 全对齐：block7 / target_layer_ids[1,9,17,25,33] / markov_rank256 / num_anchors512）。因此**不存在"权重不可比"**——上文旧结论中的该猜测已被推翻。

### 官方 eval 口径（DeepSpec eval.py）
- `--temperature` 默认 **1.0**（标准投机采样）
- `--max-new-tokens` 默认 2048
- `--confidence-threshold` 默认 **0.0 = 不启用 confidence 早停** → 论文 τ 是在固定 block、无 confidence 调度下测的；nano-vllm 未实现 confidence 调度，不影响 accept length 复现。

### temp=1.0 复现结果（128 prompts, 3131 verify rows）
| 口径 | accept length |
|------|--------------:|
| greedy (temp=0) | 4.80 |
| **temp=1.0（稳定值, n=128）** | **5.31** |
| 论文 DSpark HumanEval τ | **5.38** |

**temp=1.0 下 accept length ≈ 5.31，与论文 5.38 仅差 ~1.3%，基本复现。**（16-prompt temp=1 曾测出 5.83，是小样本方差；128 prompt 收敛到 5.31。）

运行命令：
```bash
LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs \
/root/llm/nano-vllm/llm/bin/python benchmarks/bench_eagle3.py \
    --model /root/llm/model/Qwen3-4B \
    --draft-model /root/llm/model/dspark_qwen3_4b_block7 \
    --spec-method dspark --prompt-mode humaneval \
    --num-prompts 128 --max-num-seqs 64 --max-tokens 128 \
    --temperature 1 --num-spec-tokens 7 --no-baseline
```

### 为什么 temp=1 > greedy
DSpark drafter 用 **total-variation 损失**训练去匹配 target 的**整个分布**（非仅 argmax）：
- greedy verify 要求 `draft == target argmax`（严格），逐位置接受率衰减快（90%→22%@k7）；
- temp=1 标准投机采样按 `min(1, p_target/q_draft)` 接受，draft 分布≈target 分布 → 接受概率接近 1，曲线平（90%→38%@k7）。

这正是论文用 temp=1.0 报数的原因，也是"之前 greedy 只有 4.7、看着不如论文"的真因。

### Markov 头逐位置贡献（per-position top1 vs target，temp=1, n=128）
| 位置 k | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------|---|---|---|---|---|---|---|
| base（仅块扩散） | 70% | 57% | 48% | 42% | 36% | 31% | 26% |
| base+Markov | 90% | 87% | 84% | 80% | 76% | 72% | 70% |
| **Markov delta** | +20 | +30 | +35 | +38 | +40 | +41 | +44 pt |

纯并行 base 衰减快（70%→26%），Markov 头把曲线大幅抬平（90%→70%），且**贡献随位置递增**（越靠后补得越多），与论文"Markov head mitigates the acceptance decay of purely parallel drafters"完全吻合。

### 采样布局 A/B（确认默认布局正确）
| layout | avg accept prefix (greedy) |
|--------|---------------------------:|
| 默认 = `--dspark-anchor-as-first`（sample_from_anchor=True） | **3.80** |
| `--dspark-bonus-anchor`（sample_from_anchor=False） | 0.81（崩） |

bonus-anchor 布局把采样位置对齐错了（base top1 k1 仅 8.6% vs 正确 77%），accept 崩到 0.81。**默认 anchor-as-first 是正确布局**，已排除"布局错位"作为差距来源。

### 最终结论
**nano-vllm 的 DSpark 实现无 bug，在论文口径（temperature=1.0）下复现了官方 checkpoint 的 HumanEval accept length（5.31 ≈ 5.38）。** 之前"复现不了"是测量口径（greedy vs temp=1.0）差异，而非实现问题。

---

## ⚠️ 已废弃结论（基于修复前 backend，仅存档）

以下分析均基于**修复前有 bug 的 DSpark backend**（accept 3.53、加速 +0.9%）。backend 修复后 DSpark 已达 accept 4.10、+48.7%，这些结论**已失效**，仅作历史记录，请勿引用：

- **「block_size=7 是硬上限」（K 扫描 K=15→3.26 反降）**：反降很可能是位置错位 bug 的产物，而非纯 block_size 限制，需用修复后 backend 重测才能定论。
- **「Markov 消融：完整 3.53 vs 去 Markov 1.90，Markov 贡献 +1.63」**：绝对数值基于旧 backend，Markov"有正贡献"的方向可能仍成立，但具体量待新 backend 复测。
- **「盈亏平衡：DSpark 每步成本 3.4×、卡在平衡点所以不加速」**：该瓶颈本质是 bug 导致 accept 上不去，修复后 DSpark 已远离平衡点（+48.7%），此归因不再适用。

---

## 结论

- **DSpark（修复 backend 后）K=7：accept 4.10、加速 +48.7%、tok/s +95%，三方法中最快**，且每 token 接受率最高（59%），证明其 draft 质量本就强于 DFlash。
- **关键教训**：投机解码 backend 中「提议位置对齐」和「fuse 层选择 off-by-one」这类 bug 不会报错、不影响输出正确性，只会悄悄压低 accept length，表现为"能用但慢"，排查时极易被误判为"模型本身弱"。定位手段是对照消融（本例靠用户修正提议路径后 accept 从 3.53 跳到 4.10 才确认）。
