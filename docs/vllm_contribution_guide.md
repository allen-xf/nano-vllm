# 在 vLLM 上做事情 — 贡献与实践指南

> 基于 nano-vllm 源码级理解，向真实 vLLM 项目进阶

## 目录

- [为什么要在 vLLM 上做事情](#为什么要在-vllm-上做事情)
- [路径一：从 Good First Issue 开始贡献](#路径一从-good-first-issue-开始贡献)
- [路径二：性能优化 — 找到瓶颈并修复](#路径二性能优化--找到瓶颈并修复)
- [路径三：补齐功能 — 新模型 / 新特性支持](#路径三补齐功能--新模型--新特性支持)
- [路径四：Kernel 优化](#路径四kernel-优化)
- [路径五：在 vLLM 之上搭建系统](#路径五在-vllm-之上搭建系统)
- [路径六：Benchmark 与分析报告](#路径六benchmark-与分析报告)
- [具体可做的项目清单](#具体可做的项目清单)
- [贡献 PR 的实操流程](#贡献-pr-的实操流程)
- [简历怎么写](#简历怎么写)

---

## 为什么要在 vLLM 上做事情

| 维度 | nano-vllm | vLLM |
|------|-----------|------|
| 作用 | 理解原理 | 证明工程能力 |
| 面试价值 | "我理解推理引擎" | "我给主流框架贡献过代码" |
| 难度 | 代码 2k 行，结构清晰 | 代码 10w+ 行，工程复杂度高 |
| 社区 | 个人项目 | 顶级开源社区，PR 有 review |

nano-vllm 证明你懂原理，vLLM PR 证明你能在复杂工程中落地。两者互补。

---

## 路径一：从 Good First Issue 开始贡献

### 入口

- GitHub Issues 标签 `good first issue`：https://github.com/vllm-project/vllm/labels/good%20first%20issue
- 标签 `help wanted`：通常是社区需要但核心团队没时间做的

### 常见类型

**1. 文档和错误信息改进**

```
难度: ★☆☆☆☆
价值: 熟悉贡献流程，了解代码结构
例子: 改进某个错误场景的 error message，让用户更容易定位问题
```

**2. 新模型支持**

```
难度: ★★☆☆☆ ~ ★★★☆☆
价值: 高。直接增加 vLLM 支持的模型列表，用户可见
例子: 某个新发布的模型还没有 vLLM 支持，添加对应的 model class
```

你有 nano-vllm 的 `qwen3.py` 经验，理解 model class 的结构（QKV proj、MLP、weight_loader），添加新模型对你来说应该比较自然。

**3. Bug 修复**

```
难度: ★★☆☆☆ ~ ★★★★☆
价值: 高。修复真实用户的痛点
方法: 在 Issues 中找 bug 标签，复现 → 定位 → 修复
```

### 第一个 PR 的建议

找一个**你完全理解的模块**相关的 issue。基于 nano-vllm 的学习，你最熟悉的是：
- 调度器（scheduler）
- KV cache 管理（block manager）
- 模型加载（weight loader）
- Attention 计算

在这些模块范围内找 issue，上手最快。

---

## 路径二：性能优化 — 找到瓶颈并修复

### 方法论

```
1. 跑 benchmark → 拿到基线数据
2. Profiling（torch profiler / nsight compute）→ 找到热点
3. 分析瓶颈是 memory-bound 还是 compute-bound
4. 针对性优化
5. 跑 benchmark → 对比数据
```

### 可以做的优化方向

**1. 调度器优化**

vLLM 的调度器是纯 Python 实现，高并发时可能成为 CPU 瓶颈：

```
分析: 在 1000+ 并发下 profile 调度器耗时
优化: 数据结构优化（SortedList → 更高效的优先队列）
     减少每步调度的遍历次数
     批量操作替代逐个操作
```

**2. Prefix Caching 命中率优化**

vLLM 的 prefix caching 用 hash 匹配，但实际场景中命中率受多因素影响：

```
分析: 不同场景下的 cache 命中率（多轮对话、RAG、batch 推理）
优化: 更好的 eviction 策略（LRU vs LFU vs ARC）
     cache-aware 调度（优先调度能命中 cache 的请求）
```

**3. Decode 阶段 Batch 效率**

```
分析: 不同 batch size 下 decode 的 GPU utilization
优化: 更细粒度的 batch padding 策略
     动态 batch size 调整
```

### Profiling 工具

```bash
# torch profiler
python -m torch.profiler.profile --activities cpu,cuda --output trace.json \
    -m vllm.entrypoints.openai.api_server ...

# nsight compute（kernel 级分析）
ncu --set full python benchmark.py

# nsight systems（系统级时间线）
nsys profile python benchmark.py
```

---

## 路径三：补齐功能 — 新模型 / 新特性支持

### 添加新模型支持

这是**最容易入手且价值最高**的贡献方式。步骤：

```
1. 找一个 vLLM 还不支持的模型（看 Issues 中的 model request）
2. 读模型的 HuggingFace 实现（modeling_xxx.py）
3. 参考 vLLM 中类似模型的实现（如 qwen3.py、llama.py）
4. 写 vLLM 版本:
   - 替换 Linear → ColumnParallelLinear / RowParallelLinear
   - 替换 Attention → PagedAttention
   - 实现 weight_loader（处理 TP 切分）
5. 测试: 对比 HF 和 vLLM 输出是否一致
6. 提 PR
```

你在 nano-vllm 中已经理解了这整套流程（qwen3.py 的 QKV proj、weight_loader、TP 切分），做这个方向非常对口。

### 多模态模型支持

vLLM 正在积极扩展多模态支持，机会多：

```
- 新的 VLM（Vision-Language Model）支持
- 音频模型支持
- 视频理解模型支持
- 多模态 KV cache 管理优化
```

### 新采样策略

```
- 更好的 structured output（JSON schema → FSM）
- 新的 guided decoding backend
- 自定义 logits processor
```

---

## 路径四：Kernel 优化

### Triton Kernel

vLLM 中有不少 Triton kernel 可以优化：

```
- store_kvcache: 你在 nano-vllm 中已经理解了这个 kernel
- RoPE: 旋转位置编码的 fused kernel
- RMSNorm: fused add + norm kernel
- 量化相关 kernel: W4A16、W8A8 的 dequant + matmul fusion
```

**具体做法**：

```
1. Profile 现有 kernel 的 roofline
2. 找到 memory-bound 的 kernel
3. 优化: 减少显存访问次数（fusion）、提高访存效率（coalescing）
4. Benchmark 对比
```

### CUDA Kernel

更底层的优化，难度更大但价值更高：

```
- FlashAttention 的特定场景优化（长序列、小 batch）
- Flash Decoding 的改进
- 自定义 CUTLASS GEMM kernel
```

---

## 路径五：在 vLLM 之上搭建系统

不改 vLLM 代码，而是**基于 vLLM 构建上层系统**，同样有很高的展示价值。

### 1. PD 分离 Prototype

```
架构:
  Prefill 集群（vLLM 实例 × N）→ KV Cache 传输 → Decode 集群（vLLM 实例 × M）

实现:
  - 请求路由层（Python/Go）: 接收请求，分发到 prefill 或 decode
  - KV Cache 传输: CUDA IPC / RDMA / TCP
  - 调度策略: prefill 和 decode 独立扩缩容

技术栈: vLLM + Ray Serve + 自定义路由
```

### 2. 多 LoRA 调度系统

```
架构:
  请求 → Router（识别 LoRA ID）→ vLLM 实例池（动态加载/卸载 LoRA）

实现:
  - LoRA 预热和缓存管理
  - 请求亲和性调度（同 LoRA 请求尽量发到同一实例）
  - LoRA 热度统计 + 自动淘汰

技术栈: vLLM + FastAPI + Redis
```

### 3. Speculative Decoding Benchmark 平台

```
功能:
  - 自动化测试不同 draft model 配置
  - 对比 acceptance rate、吞吐、延迟
  - 可视化分析每一步的 accept/reject 情况

技术栈: vLLM + Weights & Biases / Grafana
```

### 4. 推理服务监控系统

```
功能:
  - 实时监控 TTFT、TPOT、ITL、KV cache 利用率
  - 报警: 延迟超标、OOM 预警
  - Grafana dashboard

技术栈: vLLM + Prometheus + Grafana
```

---

## 路径六：Benchmark 与分析报告

**不写代码也能做出有价值的贡献** — 高质量的 benchmark 和分析报告在社区很受欢迎。

### 可以做的 Benchmark

**1. vLLM vs SGLang vs TensorRT-LLM 对比**

```
维度:
  - 不同模型（7B / 13B / 70B / MoE）
  - 不同并发数（1 / 10 / 100 / 1000）
  - 不同序列长度（短对话 / 长文档 / RAG）
  - 不同 GPU（A100 / H100 / L40S / 4090）

指标:
  - TTFT、TPOT、E2E Latency（P50/P95/P99）
  - Throughput (tokens/s)
  - GPU 利用率
  - 显存占用

产出: 博客 + 可复现的 benchmark 脚本
```

**2. Prefix Caching 效果分析**

```
场景:
  - 多轮对话（高缓存命中）
  - RAG（中等命中，共享 system prompt）
  - 独立请求（低命中）

分析:
  - 命中率 vs 吞吐提升曲线
  - cache 大小 vs 命中率
  - eviction 策略对比
```

**3. 量化推理对比**

```
- FP16 vs W8A8 vs W4A16: 精度损失 vs 吞吐提升
- 不同量化方案在不同任务上的精度表现
- 量化对 KV cache 大小的影响
```

---

## 具体可做的项目清单

按**性价比**（投入时间 vs 面试展示价值）排序：

### Tier 1: 高性价比（1-2 周，面试必聊）

| 项目 | 做什么 | 面试价值 |
|------|--------|----------|
| **vLLM 新模型 PR** | 给 vLLM 添加一个新模型支持 | "我给 vLLM 贡献过代码" |
| **Speculative Decoding on nano-vllm** | 在 nano-vllm 上实现 vanilla spec decoding | 深度理解 + 可运行 demo |
| **vLLM vs SGLang Benchmark** | 多维度对比 + 分析报告 | 体现系统性思维 |

### Tier 2: 中等投入（2-4 周，展示深度）

| 项目 | 做什么 | 面试价值 |
|------|--------|----------|
| **PD 分离 Prototype** | 基于 vLLM 搭建简单的 PD 分离系统 | 直接对口 JD 要求 |
| **Prefix Caching 优化 PR** | 改进 vLLM 的 cache eviction 策略 | 体现优化能力 |
| **Triton Kernel 优化** | 优化 vLLM 中的某个 kernel + benchmark | 体现底层能力 |

### Tier 3: 重投入（4+ 周，差异化竞争力）

| 项目 | 做什么 | 面试价值 |
|------|--------|----------|
| **EAGLE on vLLM** | 在 vLLM 上实现或改进 EAGLE spec decoding | 前沿 + 工程深度 |
| **跨机 EP 实验** | MoE 模型的 Expert Parallelism 部署和调优 | 直接对口 JD |
| **推理监控平台** | Prometheus + Grafana 监控 vLLM 服务 | 体现生产级思维 |

---

## 贡献 PR 的实操流程

### 环境搭建

```bash
# Fork + Clone
git clone https://github.com/YOUR_USERNAME/vllm.git
cd vllm

# 开发环境（推荐用 Docker）
docker build -f Dockerfile.dev -t vllm-dev .
# 或直接安装
pip install -e ".[dev]"

# 跑测试确认环境正常
pytest tests/unit_tests/ -x -q
```

### 开发流程

```bash
# 1. 创建分支
git checkout -b feat/add-xxx-model

# 2. 开发 + 本地测试
pytest tests/models/test_xxx.py -x -v

# 3. 代码规范检查
pre-commit run --all-files

# 4. 提 PR
# - 标题简洁: "Add support for XXX model"
# - 描述清楚: 做了什么、为什么、怎么测的
# - 附 benchmark 数据（如果是性能优化）
```

### PR 被接受的关键

```
1. 解决真实问题（对应一个 Issue）
2. 代码符合项目规范（跑通 CI）
3. 有测试（不只是 happy path）
4. PR description 清晰
5. 及时响应 reviewer 的 comments
```

### 常见被拒原因

```
- 没有对应 Issue，凭空加功能
- 改动太大，没有拆分
- 没有测试
- 代码风格不符合项目规范
- 性能优化没有 benchmark 数据
```

---

## 简历怎么写

### 有 vLLM PR 的写法

```
vLLM 开源贡献 (github.com/vllm-project/vllm)
  - 实现 XXX 模型推理支持，包含 TP 并行、weight loading、KV cache 适配 (PR #xxxx, merged)
  - 优化 prefix caching 的 eviction 策略，在多轮对话场景下命中率提升 15% (PR #xxxx)
  - 修复 XXX 场景下的 OOM 问题，影响 XX 个用户 (PR #xxxx, merged)
```

### 有 vLLM 上层系统的写法

```
基于 vLLM 的 Prefill-Decode 分离推理系统
  - 设计 PD 分离架构，prefill 和 decode 集群独立部署和扩缩容
  - 实现 KV Cache 跨节点传输，通过流水线 overlap 隐藏 90%+ 传输延迟
  - 在 XX 并发下 TTFT 降低 40%，整体吞吐提升 2.3x
```

### 关键原则

| 做法 | 说明 |
|------|------|
| **PR 编号** | 写上 PR #xxxx，面试官会去看 |
| **merged** | 如果合入了一定标注，比 open 状态的 PR 有说服力得多 |
| **量化** | 必须有数字：提升 xx%、减少 xx ms、影响 xx 用户 |
| **体现理解深度** | 不只是"实现了"，要写为什么这么做、解决了什么问题 |

---

## 推荐的起步顺序

```
第1步: 跑通 vLLM 开发环境，读源码，理解和 nano-vllm 的对应关系
  ↓
第2步: 在 Issues 中找一个你能做的（新模型支持 / bug fix），提第一个 PR
  ↓
第3步: 做一个有深度的项目（spec decoding / PD 分离 / kernel 优化）
  ↓
第4步: 写博客，整理 benchmark 数据
  ↓
第5步: 面试时能讲清楚: "我在 vLLM 上做了 XX，遇到了 YY 问题，用 ZZ 方法解决"
```

核心原则：**一个 merged PR 胜过十篇学习笔记**。先从简单的开始，建立贡献记录，再做有深度的项目。
