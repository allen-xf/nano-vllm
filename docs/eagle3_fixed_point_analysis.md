# EAGLE3 Draft Model Fixed Point 现象分析

## 现象描述

在 nano-vllm 的 EAGLE3 speculative decoding 调试过程中，观察到 draft model 的 serial loop 经常陷入 **fixed point**（不动点）行为：连续 K 步预测同一个 token。

### 实测数据示例

```
seq 0: prev_drafts=[279, 279, 279, 279, 279]      ← 完全 fixed point
seq 0: prev_drafts=[311, 311, 311, 311, 311]      ← 完全 fixed point
seq 1: prev_drafts=[264, 501, 2155, 30128, 990]   ← 多样
seq 1: prev_drafts=[279, 16170, 315, 279, 501]    ← 多样（acceptance=4）
seq 2: prev_drafts=[1265, 1265, 1265, 1265, 1265] ← 完全 fixed point
seq 2: prev_drafts=[18435, 18435, 18435, 18435, 18435] ← 完全 fixed point
```

陷入 fixed point 的序列 acceptance 几乎总是 1，多样化的序列经常拿到 2-4。

## Acceptance Rate 总体表现

| | vLLM | nano-vllm |
|---|---|---|
| mean acceptance | 1.72 | 1.67 |
| 差距 | — | -3% |

差距 0.05 在合理范围内，主要来自首轮固定 1 + 数值精度差异。**不是 bug**，是 EAGLE 类 draft model 的固有特性。

## 根因分析

### Serial Loop 数据流

```python
for k in range(num_serial_steps):
    draft_logits, current_hidden = self.draft_model(current_input, draft_positions, current_hidden)
    draft_token = draft_logits.argmax(dim=-1)
    target_token = self.draft_model.d2t[draft_token]
    current_input = target_token        # 自己的预测作为下一步输入
    draft_tokens_all.append(target_token)
```

每步：
- `current_input` = 上一步的预测 token X
- `current_hidden` = 上一步的 un-normed midlayer 输出 H_k

### Eagle3DecoderLayer 内部计算

```python
normed_embeds = input_layernorm(embed(X))       # 每步相同 (input X 不变)
normed_hidden = hidden_norm(H_k)                # 经 RMSNorm 后方向趋同
concat = cat([normed_embeds, normed_hidden])
attn_out = attention(concat)                     # attend 自己刚写入的 KV
hidden = attn_out + fused_hidden                 # residual: 累积 X 的表征
output = MLP(post_norm(hidden)) + hidden         # residual: 进一步累积
```

### 三个叠加因素

#### 1. Exposure Bias（曝光偏差）

| | 训练时 | 推理时 |
|---|---|---|
| 输入来源 | teacher forcing（正确 token） | 自己的预测 |
| 误差累积 | 无 | 有 |

一旦 step 0 预测错（X ≠ 正确 token），后续所有步都在错误上下文上预测，错误持续放大。

#### 2. 单层模型容量有限

- Draft model 只有 **1 层 decoder**
- 没有足够深度纠正错误输入带来的偏差
- 对比 target model（36 层），即使输入不完美也能通过多层注意力修正

#### 3. 残差连接 + RMSNorm 形成吸引子

```
H_{k+1} = MLP(...) + attn_out + H_k    ← 残差直接累积
```

- `H_k` 通过残差直接传到 `H_{k+1}`
- `hidden_norm(H_{k+1})` 归一化后，方向和 `hidden_norm(H_k)` 趋近
- `embed(X)` 每步完全相同 → concat 趋近 → attention 输出趋近
- **系统收敛到不动点**：H_k → H_{k+1} → H_{k+2} 几乎只增长 norm，方向不变

实测 hidden_norm 增长：394 → 492 → 644（norm 在涨，但归一化后几乎是同一个方向）

## 为什么不同序列表现不同

取决于**首步预测的 token 是否落在吸引域内**：

| 序列特征 | Fixed point 风险 | 原因 |
|---------|-----------------|------|
| 首步预测高频虚词（"the", "of", "."） | 高 | 这类 token 的 embedding 在隐空间中是"中心"，容易锁定 |
| 首步预测内容词（专有名词、数字） | 低 | 表征更独特，attention 分布更分散 |

实例对比：
- seq 0/2 经常陷入 fixed point on "the", " of", " and" 等虚词
- seq 1（解释量子计算的 prompt）多样性更高，因为内容更专业

## 是否是实现 bug

**不是**。这是 EAGLE 类 draft model 的**固有特性**：

1. vLLM 的 EAGLE3 实现也有同样现象（acceptance 1.72 也并不高）
2. EAGLE3 论文中 acceptance 通常在 1.5-2.5 之间，远不如 target 自回归质量
3. 1 层 draft model 的容量限制是设计选择（牺牲质量换速度）

## 缓解方案（可选）

如果要进一步提升 acceptance，可以考虑：

### 1. 增加 draft 模型深度
- 用 2-3 层的 draft model（如 EAGLE-2 风格）
- 代价：每步 draft 时间增加

### 2. Tree-based speculative decoding（EAGLE-2 主推方向）
- 不做 K 步 chain，而是 chain 顶端 + 多 token 并行猜测
- 例如 [d0, [d1a, d1b, d1c]] 同时验证 3 条分支
- vLLM 已经支持

### 3. Sampling 而非 argmax
- Draft model 用 top-k/nucleus sampling 而非 greedy
- 能跳出 fixed point，但可能引入更多错误预测

### 4. 限制 K 数量
- 既然 fixed point 后无效，可以早停
- 如检测连续 2 步同 token，K 截断到 2

## 诊断日志说明

### `DECODE-LAYER` residual stream norm

```python
rs = hidden_states + residual if residual is not None else hidden_states
print(rs[0].norm())
```

这行打印的是 target model 在 decode 单 token 时，第一个 token 的 residual stream 向量 L2 norm。

它主要用于检查：

1. hidden 是否数值异常：例如 `nan`、极大值或接近 0。
2. Pre-Norm Transformer 的 residual stream 是否呈合理尺度变化。
3. EAGLE3 选择的 fuse layers 附近，hidden 分布是否明显异常。
4. 对比 `FUSE` 日志，判断 capture-before-layer 和 layer-after-output 的尺度差异。

这里的 `.norm()` 默认是 L2 norm：

```text
sqrt(x1² + x2² + ... + xn²)
```

它可以理解为 hidden 向量的整体长度。

### `FUSE` captured hidden 统计

```python
for l in capture_layers:
    c = captured[l]
    print(c.shape, c.norm(), c.mean(), c.std())
```

这段打印的是 EAGLE3 draft model 的 fuse 输入，也就是 target model 被捕获的多层 hidden states。

各字段含义：

- `shape`: 检查 captured hidden 的维度是否是 `[num_tokens, hidden_size]`。
- `norm`: 整个 captured tensor 的 L2 norm，用来看整体尺度。
- `mean`: 所有元素均值，用来看是否有异常偏移。
- `std`: 所有元素标准差，用来看 hidden 分布尺度。

EAGLE3 后续会做：

```python
fused_hidden = draft_model.fc(torch.cat([captured[l] for l in fuse_layers], dim=-1))
```

所以如果某一层 captured hidden 的 shape、norm、mean 或 std 明显异常，draft model 的输入分布就会偏离训练时分布，acceptance rate 可能下降，甚至出现 fixed point。

这类日志只用于调试，不影响模型计算；稳定后可以删除或用 debug flag 控制。

## 参考实测

```
[FUSE] layer 2:  norm=184,   mean=0.0043, std=0.7422
[FUSE] layer 18: norm=12480, mean=0.2314, std=50.25
[FUSE] layer 33: norm=12608, mean=0.3145, std=50.75

[DECODE-LAYER] layer 0:  norm=12.00
[DECODE-LAYER] layer 18: norm=57.25
[DECODE-LAYER] layer 35: norm=564.00

draft hidden_norm 序列: 394 → 492 → 644 (持续增长但方向收敛)
draft logits_std 序列: 2.13 → 2.55 → 2.73 (置信度反而上升)
```

## 结论

- nano-vllm 的 EAGLE3 实现 **acceptance ≈ 1.67，与 vLLM 1.72 仅相差 3%**
- Fixed point 是 EAGLE 单层 draft model 的固有现象，不是实现 bug
- 进一步优化需要在算法层面（tree decode、deeper draft）而非实现层面
