# 两个月学习计划：推理 Infra 方向

> 针对职位要求：推理引擎优化、PD分离、跨机EP、请求调度、多模态推理、CUDA/Triton kernel、分布式推理技术栈

## 已有背景

- 数据分布式系统
- nano-vllm 项目（vLLM 源码级理解）
- TP/DP/FSDP 概念
- Ray 使用

---

## 第 1 周：推理引擎核心机制深化

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | vLLM 调度器源码精读：Continuous Batching、Prefix Caching、Chunked Prefill | 笔记 + QA |
| Day 3-4 | SGLang RadixAttention、Constrained Decoding 机制 | 对比 vLLM 的笔记 |
| Day 5-6 | 请求调度策略：FCFS vs SJF vs priority-based、preemption 机制 | 笔记 |
| Day 7 | nano-vllm 中补充 prefix caching 或 chunked prefill 实现 | 代码 |

## 第 2 周：PD 分离 + 分布式推理架构

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | PD分离原理：为什么分离（prefill compute-bound, decode memory-bound）、DistServe/Splitwise 论文 | 笔记 |
| Day 3-4 | PD分离实现：KV Cache 传输、prefill/decode 集群独立扩缩容、调度策略 | 架构图 |
| Day 5-6 | Mooncake 论文/代码：KV Cache 池化、对象存储式 KV 管理 | 笔记 |
| Day 7 | 在 nano-vllm 上设计简单的 PD 分离 prototype | 设计文档 |

## 第 3 周：跨机 EP + 分布式并行策略

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | MoE 模型推理特性：Expert Parallelism 原理、All-to-All 通信 | 笔记 |
| Day 3-4 | 跨机 EP 实现：通信开销分析、负载均衡（expert 热度不均）、DeepSeek-V3 的 EP 策略 | 笔记 |
| Day 5-6 | TP vs EP vs DP 的 trade-off，混合并行策略设计 | 对比分析文档 |
| Day 7 | NCCL 通信原语复习、Ring AllReduce vs Tree AllReduce、通信 overlap 技术 | 笔记 |

## 第 4 周：CUDA/Triton Kernel 开发

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | CUDA 基础强化：memory hierarchy、bank conflict、coalescing、occupancy 分析 | 练习代码 |
| Day 3-4 | 手写 fused softmax kernel + Triton 版本对比 | 代码 + benchmark |
| Day 5-6 | FlashAttention 原理 + 简化版实现（tiling、online softmax） | 代码 + benchmark |
| Day 7 | Flash Decoding：split-KV 并行 + online softmax reduce，理解与 FlashAttention 的区别（prefill 用 FA，decode 用 Flash Decoding），看 vLLM 中实现 | 笔记 + profiling 报告 |

## 第 5 周：Speculative Decoding + 量化推理

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | Speculative Decoding 论文 + 在 nano-vllm 上实现 | 代码 |
| Day 3-4 | 调优 + benchmark 不同 draft model 配置 | benchmark 数据 |
| Day 5-6 | 量化推理：W8A8（SmoothQuant）、W4A16（GPTQ/AWQ）原理与实现 | 笔记 + 代码 |
| Day 7 | 稀疏推理：结构化剪枝、sparse kernel 加速原理 | 笔记 |

## 第 6 周：多模态模型推理 + Omni 架构

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | 多模态模型结构：Vision Encoder + LLM、cross-attention vs early fusion | 笔记 |
| Day 3-4 | Omni 模型推理特性：图片/音频/视频 token 的 prefill 特殊性、动态分辨率 | 笔记 |
| Day 5-6 | vLLM 多模态推理支持源码、多模态 KV Cache 管理 | 代码分析笔记 |
| Day 7 | 多模态推理调度难点：不同模态 compute profile 不同、batching 策略 | 分析文档 |

## 第 7 周：分布式推理技术栈 + 系统集成

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | Ray Serve 用于推理部署、自动扩缩容 | 动手实验 |
| Day 3-4 | NVIDIA Dynamo：推理编排框架、multi-node routing | 文档/代码阅读笔记 |
| Day 5-6 | AIBrix 架构、Kubernetes 上的推理服务部署 | 笔记 |
| Day 7 | RL Infra 概览：verl 架构、online RL 如何依赖推理（rollout），了解即可 | 笔记 |

## 第 8 周：项目打磨 + 面试准备

| 天 | 内容 | 产出 |
|---|---|---|
| Day 1-2 | 完善 nano-vllm 项目（speculative decoding + PD分离 prototype） | GitHub 仓库 |
| Day 3-4 | 写 2 篇技术博客（PD分离原理 / Speculative Decoding 实现） | 博客 |
| Day 5-6 | 系统设计题准备：设计一个支持PD分离+跨机EP的推理系统 | 面试笔记 |
| Day 7 | 面试高频问题整理：bottleneck 分析、trade-off、为什么这么设计 | 面试笔记 |

---

## 里程碑检查

| 时间点 | 应该达到 |
|---|---|
| 第 2 周末 | 能说清 PD 分离的动机、实现方式、Mooncake 的核心思路 |
| 第 3 周末 | 能说清跨机 EP 的通信模式、负载均衡问题、和 TP 的对比 |
| 第 4 周末 | 有手写 CUDA kernel + FlashAttention 简化版，有 benchmark |
| 第 5 周末 | nano-vllm 上跑通 speculative decoding，理解量化推理 |
| 第 6 周末 | 能说清多模态推理的特殊挑战和 Omni 架构特点 |
| 第 8 周末 | 2 个可展示项目 + 博客 + 面试准备完成 |

---

## 核心阅读材料

| 主题 | 材料 |
|---|---|
| PD分离 | DistServe 论文、Splitwise 论文、Mooncake 论文 |
| 跨机EP | DeepSeek-V3 技术报告（EP 部分）、MegaBlocks 论文 |
| Kernel | FlashAttention 1/2 论文、CUTLASS 文档 |
| Speculative Decoding | Leviathan et al. 2023、Medusa、Eagle |
| 多模态 | LLaVA、Qwen-VL、vLLM multimodal 文档 |
| 分布式推理 | Dynamo GitHub、AIBrix 文档、Mooncake 论文 |
| 量化 | SmoothQuant、AWQ、GPTQ 论文 |

---

## 与原 verl_learning.md 计划的区别

| 调整 | 原因 |
|---|---|
| RL 理论从 2 周压缩到第 7 周 1 天概览 | 职位核心是推理 infra，RL 只需了解 |
| 新增 PD 分离专题（1周） | 职位明确要求 |
| 新增跨机 EP 专题（1周） | 职位明确要求 |
| 新增多模态推理（1周） | 职位要求 Omni 模型 |
| 新增 Mooncake/Dynamo/AIBrix | 职位加分项明确列出 |
| CUDA kernel 保留但前移 | 是核心竞争力 |
| Speculative Decoding 保留 | 推理优化核心技术 |
| Mini RLHF 框架改为了解 verl 即可 | 不是目标岗位重点 |

---

## 简历呈现

### 项目经历写法

**1. Speculative Decoding 推理加速（个人项目）**

> - 实现 draft-verify 推理框架，支持多种 draft model 配置
> - 在 Llama-7B 上实测达到 x.x 倍加速，分析不同 acceptance rate 下的收益曲线
> - 优化 verify 阶段 batch 策略，减少无效计算开销

**2. Prefill-Decode 分离架构（个人项目）**

> - 设计并实现单机双卡 PD 分离 prototype，基于 CUDA IPC 实现 KV Cache 零拷贝传输
> - 实现逐层流水线传输，传输延迟与 prefill 计算 overlap，隐藏 90%+ 传输开销
> - 对比分离 vs 混合部署在不同并发下的 TTFT 和吞吐差异

**3. 高性能 Kernel 开发**

> - 手写 CUDA Fused Attention Kernel，对比 PyTorch 原生实现加速 xx%
> - 基于 Triton 实现 FlashAttention 简化版，分析 tiling 策略对不同 seq_len 的影响
> - 使用 nsight compute 做 roofline 分析，定位 memory-bound / compute-bound 瓶颈

**4. 技术博客 / 开源贡献**

> - 发表技术博客：《PD 分离架构原理与实现》《Speculative Decoding 从论文到落地》
> - （如果有）vLLM / SGLang PR 贡献

### 技能栏写法

```
推理优化：vLLM/SGLang 源码级理解，PD分离，Speculative Decoding，KV Cache 优化
分布式推理：TP/EP/DP，跨机 Expert Parallelism，NCCL，Ray Serve
高性能计算：CUDA/Triton Kernel 开发，FlashAttention，量化推理（W8A8/W4A16）
系统工具：Kubernetes，Mooncake，Dynamo，nsight compute，torch profiler
```

### 简历原则

| 原则 | 说明 |
|------|------|
| 量化结果 | "加速 1.8x"、"吞吐提升 40%"、"延迟降低 30%" — 必须有数字 |
| 体现深度不体现学习 | 写"实现了 XX"、"优化了 XX"，不要写"学习了 XX" |
| 对齐 JD 关键词 | PD分离、跨机EP、Speculative Decoding、CUDA Kernel、Mooncake — 直接出现在简历里 |
| 项目名要吸引人 | "nano-vllm" 暗示对标 vLLM 且精简实现 |
| GitHub 链接 | 代码整洁 + README 清晰 + benchmark 图表，面试官会看 |
| 挑重点 | 挑 2-3 个有深度的项目，其余放技能栏，不要全列 |

### 不要写的

- ❌ "自学了 XX 论文" — 面试官不关心你读了什么
- ❌ "了解 XX 原理" — 太弱，要写"实现"或"优化"
- ❌ 把 8 周内容全列上 — 选最有深度的展示

---

## 硬件需求

### 最低要求

| 用途 | 硬件 | 说明 |
|------|------|------|
| CUDA Kernel 开发 | 1 张 GPU（RTX 3090/4090 24GB） | 写 kernel、跑 benchmark 够用 |
| 模型推理实验 | 24GB+ 显存 | Llama-7B FP16 需要 ~14GB |
| Speculative Decoding | 同上 | draft + target 模型一起约 16-20GB |

### 理想配置

| 用途 | 硬件 | 说明 |
|------|------|------|
| 分布式推理（TP/EP） | 2-4 张 GPU 或多机 | 跨机 EP 至少需要 2 节点 |
| PD 分离 prototype | 2+ 张 GPU | 模拟 prefill 和 decode 分离部署 |
| 大模型实验 | A100 40/80GB | 跑 MoE 或多模态模型 |

### 现实方案

| 方案 | 费用 | 适合 |
|------|------|------|
| 自有 4090 单卡 | 一次性 ~1.5w | kernel 开发、单卡推理、speculative decoding，覆盖 60% 内容 |
| 云 GPU 按需租 | A100 约 5-15 元/小时 | 分布式实验、大模型，按需开 |
| AutoDL / 矩池云 | 月卡几百到几千 | 性价比高，适合长期开发 |
| Colab Pro+ | ~200 元/月 | A100 有时间限制，适合轻度实验 |

### 各周硬件需求

| 周 | 最低需要 |
|---|---|
| 第 1 周（调度源码） | 单卡 4090 或纯代码阅读无需 GPU |
| 第 2 周（PD 分离） | 2 卡（可用云） |
| 第 3 周（跨机 EP） | 多卡/多机（云，或理论+代码阅读为主） |
| 第 4 周（CUDA kernel） | 单卡 4090 |
| 第 5 周（Spec Decoding） | 单卡 24GB |
| 第 6 周（多模态） | 单卡 24GB+（或代码阅读为主） |
| 第 7 周（技术栈） | 按需，文档阅读为主 |
| 第 8 周（打磨） | 单卡 |

### 建议

- **如果只买一张卡**：RTX 4090 24GB，能覆盖 70% 的实验内容
- **分布式实验**（PD 分离、跨机 EP）：租云 2-4 卡 A100 用几天就够，不需要长期持有
- 分布式部分也可以侧重代码阅读 + 设计文档，面试时讲清楚原理和 trade-off 比实际跑通更重要

---

## 核心原则

1. **先跑通再理解**：不要卡在理论，先把代码跑起来，再回头看为什么
2. **每周有产出**：笔记、代码、或博客，不要只"看"不"做"
3. **深度 > 广度**：PD分离或 Speculative Decoding 做透一个比浮光掠影做三个有用
4. **面试导向**：每个主题都想好"面试时怎么讲"，准备好 3 个深入问题的答案
5. **结合 nano-vllm**：尽量把学到的东西实现到 nano-vllm 上，形成可展示的项目
