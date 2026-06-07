# 学 verl 的大纲（有 vLLM 背景，无 RL 背景）

## 第一阶段：RL 基础概念

1. MDP：状态、动作、策略、奖励
2. 价值函数 V(s) 和 Advantage
3. 策略梯度 → REINFORCE → PPO
4. Reward Model 和 Bradley-Terry
5. Critic 的作用和训练
6. KL 惩罚（防止策略跑偏）

## 第二阶段：RLHF 训练流程

1. 四个模型的角色：Actor、Critic、RM、Ref
2. 完整训练循环：rollout → RM 打分 → 算 Advantage → 更新 Actor/Critic
3. PPO 的 clip loss
4. GAE（Generalized Advantage Estimation）

## 第三阶段：分布式训练背景（已有部分）

1. TP/PP/DP/FSDP（已了解）
2. Ray Actor 模型（已了解）
3. 多模型并行编排的难点

## 第四阶段：verl 架构

1. **资源调度**：4 个模型如何共享 GPU（时分复用 vs 独占）
2. **数据流**：rollout 的数据怎么在 Actor/Critic/RM/Ref 之间流转
3. **Worker 设计**：每个角色的 Ray Actor 组织方式
4. **和 vLLM 的关系**：verl 用 vLLM 做 rollout 阶段的生成（你熟悉的部分）
5. **权重同步**：Actor 更新后怎么把权重同步给 vLLM 推理引擎

## 第五阶段：verl 代码

1. 入口和配置
2. PPO Trainer 主循环
3. Rollout Worker（vLLM 生成）
4. Actor/Critic 的 FSDP 训练
5. 数据管道（DataProto）

---

## 你的优势

vLLM 背景意味着你已经懂：
- 推理调度（Continuous Batching、PagedAttention）
- TP 多卡推理
- Ray 的使用方式

verl 的 rollout 阶段本质就是调 vLLM 做生成，这块不用重新学。重点在于理解训练阶段（Advantage 计算、PPO 更新、权重同步）。

---

## 学完之后做什么项目（求职加分）

### 第一梯队：高含金量（有深度 + 有产出）

#### 1. 推理引擎优化

- **Speculative Decoding 实现**：在 nano-vllm 上加 speculative decoding（小模型 draft + 大模型 verify），展示加速比
- **自定义 CUDA kernel**：手写一个 fused attention kernel 或 fused MoE kernel，和现有实现做 benchmark 对比
- **量化推理**：实现 W4A16 / W8A8 量化推理，展示精度-速度 trade-off

#### 2. Post-training Infra

- **训练-推理混合调度器**：解决 verl 中 rollout（推理）和 training（训练）的 GPU 资源切换问题，实现更高效的时分复用
- **Async RLHF Pipeline**：把 PPO 的串行流程（rollout → reward → update）改成异步流水线，提升 GPU 利用率

### 第二梯队：实用 + 展示工程能力

#### 3. 端到端系统

- **Mini RLHF 框架**：从零实现一个简化版 verl（单机多卡），支持 PPO + GRPO，代码简洁可读（类似 nano-vllm 对 vLLM 的关系）
- **长序列推理优化**：实现 Ring Attention / Sequence Parallelism，支持超长上下文推理

#### 4. Benchmark / 分析工具

- **推理性能 Profiler**：分析 prefill/decode 各阶段的瓶颈（compute-bound vs memory-bound），输出可视化报告
- **多框架对比 Benchmark**：vLLM vs TensorRT-LLM vs SGLang 在不同场景下的吞吐/延迟对比，写成技术博客

### 第三梯队：锦上添花

#### 5. 给主流项目贡献 PR

- vLLM：修 bug、加新模型支持、优化调度
- verl：改进数据流、支持新算法（DPO/GRPO）
- SGLang：RadixAttention 相关优化

### 建议优先做

| 如果你偏 | 建议 |
|---|---|
| 推理方向 | Speculative Decoding + 手写 CUDA kernel |
| 训练方向 | Mini RLHF 框架（从零写一遍理解最深） |
| 两者都想 | Mini RLHF 框架 + 重点优化其中的 rollout 部分 |

**面试最看重的：** 能说清楚"为什么这么设计"、"瓶颈在哪"、"trade-off 是什么"。项目不用大，但要深——一个做透比三个浅尝辄止加分得多。

---

## 8 周学习计划（4-6h/天，推理+训练兼顾）

### 第 1 周：RL 理论补齐

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | MDP、策略、价值函数、Bellman 方程 | 算法笔记整理 |
| Day 3-4 | 策略梯度、REINFORCE、Advantage、Critic | 算法笔记整理 |
| Day 5-6 | PPO（clip loss、GAE）、KL 惩罚 | 算法笔记整理 |
| Day 7 | DPO、GRPO 原理对比 | 算法笔记整理 |

### 第 2 周：RLHF 全流程 + verl 架构

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | RLHF 四模型角色、完整训练循环、RM 训练 | 画流程图 |
| Day 3-4 | verl 架构：资源调度、数据流、Worker 设计 | 读 verl 文档和论文 |
| Day 5-7 | verl 代码：PPO Trainer 主循环、DataProto | 标注代码流程 |

### 第 3 周：verl 代码深入

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | Rollout Worker（怎么调 vLLM 生成） | 对照 nano-vllm 理解 |
| Day 3-4 | Actor/Critic 的 FSDP 训练、权重同步 | 笔记 |
| Day 5-6 | Reward 计算、KL 惩罚、GAE 实现 | 跑通一个小实验 |
| Day 7 | GRPO 实现（对比 PPO 的区别） | 笔记 |

### 第 4 周：nano-vllm 深入 + CUDA 基础

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | nano-vllm 调度器源码（Continuous Batching） | QA.md 补充 |
| Day 3-4 | PagedAttention 实现细节 | QA.md 补充 |
| Day 5-7 | CUDA 编程基础：thread/block/grid、shared memory、bank conflict | 跑通简单 kernel |

### 第 5 周：手写 CUDA Kernel

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-3 | 手写 fused softmax kernel，对比 PyTorch 实现的加速比 | 代码 + benchmark |
| Day 4-7 | 手写 FlashAttention 简化版（单头、无 mask），profile 分析 | 代码 + benchmark |

### 第 6 周：Speculative Decoding

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | 论文阅读（Leviathan et al.），理解 draft-verify 流程 | 笔记 |
| Day 3-5 | 在 nano-vllm 上实现 speculative decoding | 代码 |
| Day 6-7 | benchmark 加速比，不同 draft model 对比 | 技术博客草稿 |

### 第 7 周：Mini RLHF 框架

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | 设计：单机多卡，用 nano-vllm 做 rollout | 架构设计文档 |
| Day 3-5 | 实现 PPO 训练循环（Actor + Critic + RM + Ref） | 代码 |
| Day 6-7 | 支持 GRPO（去掉 Critic，用组内均值） | 代码 |

### 第 8 周：打磨 + 面试准备

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | 补充 benchmark、写 README、整理项目 | GitHub 仓库 |
| Day 3-4 | 写 1-2 篇技术博客（Speculative Decoding / RLHF Infra） | 博客 |
| Day 5-7 | 面试题整理：系统设计题、trade-off 分析、bottleneck 分析 | 面试笔记 |

---

### 里程碑检查

| 时间点 | 应该达到 |
|---|---|
| 第 2 周末 | 能画出 RLHF 完整流程图，说清每个模型的角色 |
| 第 3 周末 | 能跑通 verl 的小规模实验，理解代码主链路 |
| 第 5 周末 | 有自己写的 CUDA kernel，有 benchmark 数据 |
| 第 6 周末 | nano-vllm 上跑通 speculative decoding |
| 第 8 周末 | 有 2 个可展示项目 + 技术博客 + 面试准备完成 |

### 核心原则

1. **先跑通再理解**：不要卡在理论，先把代码跑起来，再回头看为什么
2. **每周有产出**：笔记、代码、或博客，不要只"看"不"做"
3. **深度 > 广度**：Speculative Decoding 做透一个比浮光掠影做三个有用
4. **面试导向**：每个项目都想好"面试时怎么讲"，准备好 3 个深入问题的答案
