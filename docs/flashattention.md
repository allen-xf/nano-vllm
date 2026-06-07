# FlashAttention 学习笔记

## GPU 架构背景

### 内存层次

GPU 有多级内存，越快的越小：

```
┌─────────────────────────────────────────────────┐
│  CPU DRAM (主存)                                 │
│  容量: > 1 TB     带宽: 12.8 GB/s               │
│  最慢，CPU 和 GPU 之间的数据传输经过这里           │
└────────────────────┬────────────────────────────┘
                     │ PCIe / NVLink
┌────────────────────▼────────────────────────────┐
│  GPU HBM (High Bandwidth Memory，高带宽显存)     │
│  容量: 40-80 GB    带宽: 1.5-2.0 TB/s            │
│  GPU 的"主内存"，所有张量默认存在这里              │
│  PyTorch 里说的"显存"就是 HBM                     │
└────────────────────┬────────────────────────────┘
                     │ 片上总线
┌────────────────────▼────────────────────────────┐
│  GPU SRAM (Static RAM，片上缓存)                  │
│  容量: ~192 KB/SM   带宽: ~19 TB/s               │
│  每个 Streaming Multiprocessor (SM) 私有          │
│  速度是 HBM 的 ~10 倍，但容量小 ~1000 倍          │
│  在 CUDA 中称为 "shared memory"                   │
└─────────────────────────────────────────────────┘
```

以 A100 为例：SRAM 带宽是 HBM 的约 **13 倍**（19 vs 1.5 TB/s），但容量只有 HBM 的约 **1/200,000**（192KB vs 40GB）。

### Streaming Multiprocessor (SM)

GPU 由多个 SM 组成（A100 有 108 个）。每个 SM：
- 有自己的 **SRAM**（shared memory + L1 cache，共 ~192 KB）
- 有大量线程（用于并行执行 CUDA 线程块）
- 执行一个 **kernel**（GPU 上的一个函数调用）

### CUDA 执行模型：Grid、Block 与 SM 的关系

CUDA 有软件和硬件两套层级，通过调度映射连接：

```
软件（逻辑）                    硬件（物理）
Grid（一次 kernel 启动）
  └── Block（线程块）  ──调度──→  SM（Streaming Multiprocessor）
        └── Thread（线程）         └── 执行被分配到它上面的 Block
```

- **Grid** 是软件概念：一次 kernel launch 产生一个 grid，包含多个 block
- **SM** 是硬件概念：GPU 上的物理计算单元
- GPU 调度器把 grid 中的 block 分配到可用的 SM 上执行
- 一个 SM 可以同时跑多个 block（取决于寄存器、shared memory 等资源），一个 block 只会在一个 SM 上执行

```
Grid (软件)              SM (硬件)
┌──────────┐            ┌────────┐
│ Block 0  │ ──调度──→  │  SM 0  │ ← 可同时跑多个 block
│ Block 1  │ ──调度──→  │  SM 0  │
│ Block 2  │ ──调度──→  │  SM 1  │
│ Block 3  │ ──调度──→  │  SM 1  │
│ Block 4  │ ──调度──→  │  SM 2  │
│  ...     │            │  ...   │
└──────────┘            └────────┘
```

这种映射关系在 DeepSeek-V3 的部署中被直接利用：将 132 个 SM 分成 108 + 24，计算 kernel 的 block 只调度到 108 个 SM 上，通信 kernel 的 block 只调度到另外 24 个 SM 上，物理隔离避免互相抢资源。

### Kernel 是什么

Kernel 就是**一次提交给 GPU 执行的函数调用**。可以类比理解：

```
CPU 的世界:  调用一个函数 → 在 CPU 上顺序执行
GPU 的世界:  启动一个 kernel → 在成千上万个线程上并行执行
```

在 PyTorch 中，每一个基础操作通常对应一个 kernel：

```python
S = Q @ K.T          # kernel 1: 矩阵乘法（调用 cuBLAS 的 GEMM kernel）
P = torch.softmax(S) # kernel 2: softmax kernel
O = P @ V            # kernel 3: 又一个矩阵乘法 kernel
```

每个 kernel 是**独立调度**的，GPU 在执行完一个 kernel 后才启动下一个。这意味着：

- 每个 kernel 的**输入必须在 HBM 中**（SRAM 是临时工作空间，kernel 结束就没了）
- 每个 kernel 的**输出必须写回 HBM**（否则下一个 kernel 拿不到）
- kernel 之间**无法共享 SRAM 数据**

为什么无法共享？因为 SRAM 的生命周期绑定在**线程块（thread block）**上。线程块是 kernel 的执行单元，kernel 结束时线程块销毁，SRAM 随之释放：

```
Kernel 1 启动 → 分配线程块到各 SM → 每个线程块获得一块 SRAM
                                      ↓
                                   计算完毕
                                      ↓
Kernel 1 结束 → 线程块销毁 → SRAM 释放（数据丢失）
                                      ↓
Kernel 2 启动 → 分配新的线程块 → 获得全新的 SRAM（之前的数据已经没了）
```

SRAM 就像函数的局部变量——函数返回后栈空间就释放了。所以 kernel 之间唯一的数据交换通道就是 **HBM**。

这就是为什么标准 attention 的 3 个 kernel 必须把中间矩阵 S、P 写入 HBM——不是因为想存，而是**不存就丢了**。

FlashAttention 把所有操作**融合成 1 个 kernel**，数据在 SRAM 上从头算到尾，中间结果不用落地到 HBM。

同一个 kernel 内，不同线程块在不同 SM 上**独立运行，各用各的 SRAM**，互相看不到：

```
Kernel 启动，分配多个线程块:

SM 0: 线程块 0 → SRAM 0 → 算一部分结果 → 写回 HBM 对应位置
SM 1: 线程块 1 → SRAM 1 → 算一部分结果 → 写回 HBM 对应位置
SM 2: 线程块 2 → SRAM 2 → 算一部分结果 → 写回 HBM 对应位置
...
```

每个线程块独立地把自己那部分结果写到 HBM 的**不同地址**，不需要线程块之间通信。以 FlashAttention 为例：

```
线程块 0: 负责 Q 的第 0~63 行  → 算出 O[0:64]   → 写到 HBM
线程块 1: 负责 Q 的第 64~127 行 → 算出 O[64:128] → 写到 HBM
...
```

任务划分时就保证了**各算各的，互不重叠**，最终 HBM 上的 O 矩阵被各线程块拼完。

### Kernel 执行模型

一个 kernel 的生命周期：

```
1. 从 HBM 加载数据到寄存器/SRAM
2. 在 SRAM 上计算
3. 把结果写回 HBM
```

关键限制：**kernel 之间无法共享 SRAM**。一个 kernel 结束后，SRAM 中的数据就丢了，下一个 kernel 必须从 HBM 重新加载。这就是标准 attention 需要把中间矩阵 S、P 写回 HBM 的原因——它们是不同 kernel 的输出/输入。

### 矩阵乘法基础

矩阵乘法是 LLM 推理中最核心的操作（QKV 投影、attention score、MLP 等都是矩阵乘法）。

#### 朴素实现（CPU 三重循环）

C = A × B，其中 A: (M, K)，B: (K, N)，结果 C: (M, N)：

```python
# C[i][j] = A 的第 i 行 和 B 的第 j 列 的内积
for i in range(M):        # 遍历 A 的每一行
    for j in range(N):    # 遍历 B 的每一列
        for k in range(K):  # 对应元素相乘累加
            C[i][j] += A[i][k] * B[k][j]
```

计算量 = M × N × K 次乘加，每个元素读多次，效率极低。

#### GPU Tiling 实现

GPU 不会逐元素算，而是把矩阵切成小块（tile），每个线程块负责一个输出 tile，在 SRAM 中完成计算：

```python
# 伪代码：每个线程块计算 C 的一个 tile
TILE = 2  # tile 大小，由 SRAM 容量决定（实际中通常为 64 等）

# 线程块 (bi, bj) 负责 C[bi*TILE:(bi+1)*TILE, bj*TILE:(bj+1)*TILE]
for bi in range(0, M, TILE):        # 并行：各线程块同时执行
    for bj in range(0, N, TILE):    # 并行
        C_tile = 0                   # 在 SRAM 中初始化为全零，准备逐步累加
        for bk in range(0, K, TILE):  # 串行遍历 K 维度
            A_tile = load_to_SRAM(A[bi:bi+TILE, bk:bk+TILE])  # 从 HBM 加载
            B_tile = load_to_SRAM(B[bk:bk+TILE, bj:bj+TILE])  # 从 HBM 加载
            C_tile += A_tile @ B_tile   # 在 SRAM 中算小矩阵乘法，累加到 C_tile
        write_to_HBM(C[bi:bi+TILE, bj:bj+TILE], C_tile)  # 结果写回 HBM
```

#### 具体示例（TILE=2）

以 A (6×8) × B (8×6) = C (6×6) 为例，每个 tile 是 2×2：

```
A (6×8) = 3行×4列 tile          B (8×6) = 4行×3列 tile          C (6×6) = 3行×3列 tile
┌─────┬─────┬─────┬─────┐      ┌─────┬─────┬─────┐           ┌────┬────┬────┐
│ A0  │ A1  │ A2  │ A3  │      │ B0  │ B1  │ B2  │           │ c0 │ c1 │ c2 │
├─────┼─────┼─────┼─────┤      ├─────┼─────┼─────┤           ├────┼────┼────┤
│ A4  │ A5  │ A6  │ A7  │      │ B3  │ B4  │ B5  │           │ c3 │ c4 │ c5 │
├─────┼─────┼─────┼─────┤      ├─────┼─────┼─────┤           ├────┼────┼────┤
│ A8  │ A9  │ A10 │ A11 │      │ B6  │ B7  │ B8  │           │ c6 │ c7 │ c8 │
└─────┴─────┴─────┴─────┘      ├─────┼─────┼─────┤           └────┴────┴────┘
                                │ B9  │ B10 │ B11 │
                                └─────┴─────┴─────┘
```

计算 c4（A 的第 1 行 tile × B 的第 1 列 tile），沿 K 维度遍历 4 次：

```
c4 = A4 × B1 + A5 × B4 + A6 × B7 + A7 × B10
     ──┬──   ──┬──   ──┬──   ──┬──
      bk=0    bk=1    bk=2    bk=3
```

用具体数字：

```
A4 = │17 18│  A5 = │19 20│  A6 = │21 22│  A7 = │23 24│
     │25 26│       │27 28│       │29 30│       │31 32│

B1 = │ 3  4│  B4 = │15 16│  B7 = │27 28│  B10= │39 40│
     │ 9 10│       │21 22│       │33 34│       │45 46│
```

逐步累加：

```
c4 = 0                                         ← 初始化全零

bk=0: c4 += A4 × B1  = │ 213  248│             ← 第 1 块
                        │ 309  360│

bk=1: c4 += A5 × B4  = │ 213+705    248+744 │  ← 累加第 2 块
                        │ 309+993   360+1048 │
                      = │ 918  992│
                        │1302 1408│

bk=2: c4 += A6 × B7  = │ 918+1293   992+1336│  ← 累加第 3 块
                        │1302+1773  1408+1832│
                      = │2211 2328│
                        │3075 3240│

bk=3: c4 += A7 × B10 = │2211+1977  2328+2024│  ← 累加第 4 块
                        │3075+2649  3240+2712│
                      = │4188 4352│
                        │5724 5952│

write(c4)                                      ← 写回 HBM
```

SRAM 中始终只放 3 个 2×2 tile（A_tile + B_tile + C_tile = 12 个元素），而不是整个 6×8 和 8×6 矩阵。9 个线程块各自独立算各自的 c_tile，并行执行。

#### Tile 大小由 SRAM 容量决定

SRAM 要同时放下 A_tile、B_tile、C_tile 三个 tile：

```
总共: 3 × TILE² × 元素大小

以 A100（SRAM ~192 KB/SM）、float16（2 bytes）为例：
TILE=64:  3 × 64² × 2 = 24 KB   ← 放得下，常用选择
TILE=128: 3 × 128² × 2 = 96 KB  ← 勉强
TILE=256: 3 × 256² × 2 = 384 KB ← 超出，放不下
```

TILE 太小数据复用不充分，太大 SRAM 放不下，所以选 SRAM 容量约束下的最大值。

#### 线程块内部如何协作

一个线程块有多个线程（比如 256 个），协作完成一个 tile 的计算：

```
1. 协作加载：256 个线程一起把 A_tile 和 B_tile 从 HBM 搬到 SRAM，每个线程搬几行
2. 协作计算：C_tile 有 TILE×TILE 个元素，线程分摊，每个线程算几个元素的内积
3. 同步：__syncthreads() 等所有线程算完
4. 重复：加载下一对 tile，继续累加
5. 协作写回：所有线程一起把 C_tile 从 SRAM 写到 HBM
```

所有线程共享同一块 SRAM，加载一次数据所有线程都能读到，这就是数据复用的来源。

#### 为什么 Tiling 快

| | 朴素实现 | Tiling |
|---|---|---|
| 并行度 | 逐元素串行 | (M/TILE) × (N/TILE) 个线程块并行 |
| 数据复用 | 同一元素反复从 HBM 读（见下文） | 一个 tile 读入 SRAM 后被复用 TILE 次 |
| HBM 访问次数 | M×N×K 次 | M×N×K / TILE 次（减少 TILE 倍） |

**朴素实现为什么没有数据复用**：以 `A[1][0]` 为例，算 `C[1][0]` 时内层循环从 HBM 读了它，算完就丢了。算 `C[1][1]` 时又要从 HBM 重新读一次。没有任何地方把它缓存起来——矩阵很大时，CPU/GPU 的自动缓存（cache line）早就被后续数据挤掉了。A 的每个元素被 N 列各读一次，B 的每个元素被 M 行各读一次。

**Tiling 的数据复用**：把 A_tile 手动加载到 SRAM 后，算 C_tile 的所有列都直接从 SRAM 读，不用回 HBM：

```
加载 A_tile = │17 18│ 到 SRAM（1 次 HBM 读取）
              │25 26│

算 C_tile[0][0]: 17×3 + 18×9    ← 从 SRAM 读 17, 18
算 C_tile[0][1]: 17×4 + 18×10   ← 从 SRAM 读 17, 18（复用！不回 HBM）
算 C_tile[1][0]: 25×3 + 26×9    ← 从 SRAM 读 25, 26
算 C_tile[1][1]: 25×4 + 26×10   ← 从 SRAM 读 25, 26（复用！）
```

TILE=2 时每个元素复用 2 次，TILE=64 时复用 64 次。不过跨线程块没有复用（各线程块的 SRAM 互相看不到），同一个 B_tile 会被不同线程块各自从 HBM 加载一次。

这就是 Flash Attention 中 tiling 的基础思想，只不过 Flash Attention 还额外加了 online softmax 来处理 attention 的归一化。

### Compute-bound vs Memory-bound

| 类型 | 瓶颈 | 典型操作 |
|---|---|---|
| Compute-bound | 算力（FLOPs） | 大矩阵乘法、大卷积 |
| Memory-bound | 带宽（HBM 读写） | 逐元素操作（softmax、dropout、LayerNorm） |

Attention 中的 softmax、masking、dropout 都是 **memory-bound**——计算量很小，但每个元素都要从 HBM 读一次、写一次。这意味着**减少 HBM 访问次数比减少 FLOPs 更重要**。

### Kernel Fusion（算子融合）

解决 memory-bound 问题的标准思路：把多个操作融合成一个 kernel，数据在 SRAM 上流转，避免反复写回/读取 HBM。

```
不融合（3 个 kernel）:
  HBM → SRAM → 算 S → HBM → SRAM → softmax → HBM → SRAM → 乘 V → HBM
                        ↑ 写回            ↑ 写回

融合（1 个 kernel）:
  HBM → SRAM → 算 S → softmax → 乘 V → HBM
                 全在 SRAM 上完成，中间不碰 HBM
```

FlashAttention 本质上就是一个**手写的融合 CUDA kernel**，把 attention 的全部操作（matmul + mask + softmax + dropout + matmul）融合在一起，配合 tiling 让数据块刚好放进 SRAM。

---

# Online Softmax 推导（数学归纳法）

## 目标

证明在处理完前 i 个元素后，始终满足：

$$d_i = \sum_{j=1}^{i} e^{x_j - m_i}, \quad m_i = \max(x_1, \dots, x_i)$$

## 递推公式是怎么来的（正向推导）

我们想要一遍扫描就维护好 $d_i$。已知：

$$d_i = \sum_{j=1}^{i} e^{x_j - m_i}$$

把第 $i$ 项从求和中拆出来：

$$d_i = \underbrace{\sum_{j=1}^{i-1} e^{x_j - m_i}}_{\text{历史项（但 max 变了）}} + \underbrace{e^{x_i - m_i}}_{\text{新元素贡献}}$$

问题在于：我们手里存的是 $d_{i-1} = \sum_{j=1}^{i-1} e^{x_j - m_{i-1}}$，分母用的是**旧 max** $m_{i-1}$，而我们需要的是用**新 max** $m_i$ 的版本。

怎么把旧的变成新的？对每一项乘一个修正因子：

$$e^{x_j - m_{i-1}} \cdot e^{m_{i-1} - m_i} = e^{x_j - m_i}$$

这个修正因子 $e^{m_{i-1} - m_i}$ 对所有历史项 $j$ 都一样，所以可以提到求和号外面，直接乘到 $d_{i-1}$ 上：

$$\sum_{j=1}^{i-1} e^{x_j - m_i} = \sum_{j=1}^{i-1} e^{x_j - m_{i-1}} \cdot e^{m_{i-1} - m_i} = d_{i-1} \cdot e^{m_{i-1} - m_i}$$

代回去就得到递推公式：

$$\boxed{d_i = d_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}}$$

> **总结**：这个公式不是凭空定义的，而是从 $d_i$ 的定义出发，把"用新 max 重算所有历史项"这个 $O(i)$ 操作，用一次 $O(1)$ 的整体缩放代替了。能这样做的原因是修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 不依赖 $j$。

### 结论：Online Softmax 优化了什么

Online softmax 把 safe softmax 的 3-pass 优化成了 2-pass：

```
Safe softmax (3-pass):
  Pass 1: 遍历 x，求 m = max(x)
  Pass 2: 遍历 x，求 l = Σ e^{x_j - m}
  Pass 3: 遍历 x，求 softmax(x_j) = e^{x_j - m} / l

Online softmax (2-pass):
  Pass 1: 遍历 x，同时求 m 和 l        ← 合并了 Pass 1+2
  Pass 2: 遍历 x，求 softmax(x_j) = e^{x_j - m} / l
```

其中 $x$ 是 attention 中的 score 向量：$x = q_i K^T$，即某一行 query 和所有 key 点乘的结果，长度为 $N$。

Pass 2 无法省掉——要算出每个元素的 softmax 值 $e^{x_j - m}/l$，必须等 $m$ 和 $l$ 都确定后才能计算。因此对**纯 softmax**（输出完整概率向量）而言，2-pass 是 online softmax 的极限。

FlashAttention 能进一步到 **1-pass**，是因为 attention 的目标不是 softmax 本身，而是加权和 $O = \text{softmax}(s) \cdot V$。加权和的分子 $o$ 可以和分母 $d$ 用相同的修正因子同步递推，最后一除即可，不需要单独算出每个 softmax 值。

### 补充：pre-softmax logits 存到 HBM vs 不存重算

$x = q_i K^T$ 是 pre-softmax logits，长度为 $N$。其中 $N$ 为序列长度（输入 token 的数量），$d$ 为 head 维度（每个 Q/K/V 向量的维度）。典型值：GPT-2 中 $N=1024, d=64$。

如果 SRAM 放不下，有两个选择：

#### 选择 1：存 x 到 HBM（标准 attention 的做法 3 pass）

```python
# 对于第 i 行 query:

# 算一次 x，存到 HBM
for j in range(N):
    k_j = load_from_HBM(K[j])        # 读 d 个元素
    x[j] = q_i · k_j                 # 标量
    store_to_HBM(x[j])               # 写 1 个元素
# IO: 读 N×d (K) + 写 N (x)

# ---- x 现在在 HBM 里，后续直接读 ----

# Pass 1: 求 max
for j in range(N):
    x_j = load_from_HBM(x[j])        # 读 1 个元素
    m = max(m, x_j)
# IO: 读 N

# Pass 2: 求 sum
for j in range(N):
    x_j = load_from_HBM(x[j])        # 读 1 个元素
    l += exp(x_j - m)
# IO: 读 N

# Pass 3: 求 output
for j in range(N):
    x_j = load_from_HBM(x[j])        # 读 1 个元素
    v_j = load_from_HBM(V[j])        # 读 d 个元素
    o += (exp(x_j - m) / l) * v_j
# IO: 读 N + N×d
```

存了 x 之后，Pass 1/2/3 只需要读**标量**，不用再碰 K。

```
每行 IO = N·d + N + N + N + N·d = O(N·d)
额外显存 = N（一行 x），全部 N 行 → O(N^2)
```

为什么是 $O(N^2)$ 而不是 $O(N)$？虽然每行的 x 用完可以丢弃，逐行串行处理的话确实只需 $O(N)$ 显存。但 GPU 有成千上万个线程，逐行处理意味着每次矩阵乘法规模只有 $(1, d) \times (d, N)$，**太小了，喂不饱 GPU**，算力浪费严重。所以标准 attention 选择一次算出整个 $S = QK^T$（大矩阵乘法，GPU 满载），代价就是 $O(N^2)$ 显存。FlashAttention 的折中是**分块**——每块 $B_r$ 行一起算，块大小刚好塞满 SRAM，既有足够的并行度，又不需要 $O(N^2)$ 显存。

#### 选择 2：不存 x，每次重算（safe softmax 版）

```python
# 对于第 i 行 query:

# Pass 1: 求 max —— 要重算 x
for j in range(N):
    k_j = load_from_HBM(K[j])        # 读 d 个元素
    x_j = q_i · k_j                  # 实时算
    m = max(m, x_j)
    # x_j 丢弃，不存
# IO: 读 N×d

# Pass 2: 求 sum —— 又要重算 x
for j in range(N):
    k_j = load_from_HBM(K[j])        # 又读 d 个元素！
    x_j = q_i · k_j                  # 又算一遍！
    l += exp(x_j - m)
# IO: 读 N×d

# Pass 3: 求 output —— 再次重算 x
for j in range(N):
    k_j = load_from_HBM(K[j])        # 第三次读 K！
    v_j = load_from_HBM(V[j])        # 读 d 个元素
    x_j = q_i · k_j                  # 第三次算！
    o += (exp(x_j - m) / l) * v_j
# IO: 读 N×d + N×d
```

每次访问 x 都要**重新读 K 的 d 维向量并做点乘**，比从 HBM 读 1 个标量贵 d 倍。

```
每行 IO = N·d + N·d + N·d + N·d = O(N·d)，但常数是 4
额外显存 = O(1)（只存 m, l, o）
```

#### 对比

| | 存 x 到 HBM | 不存重算 |
|---|---|---|
| 每行读 K 次数 | 1 次 | 3 次 (safe) / 2 次 (online) |
| 每行读 x 方式 | 从 HBM 读标量 | 从 K 重算（读 d 维向量） |
| 每行 IO | ~2Nd + 3N | ~4Nd (safe) / ~3Nd (online) |
| 全部 IO | $O(N^2 + Nd)$ | $O(N^2 d)$ |
| 额外显存 | $O(N^2)$ | $O(N)$ |

存 x 的优势：后续每次访问 x 只读 1 个标量，而重算要读 d 个元素再做点乘。代价是 $O(N^2)$ 显存。

不存的优势：显存从 $O(N^2)$ 降到 $O(N)$，但 IO 恶化了约 d 倍。

**FlashAttention 是第三条路**：分块存——每次只在 SRAM 里存一小块 x（$B_r \times B_c$），既不写 HBM，K 也只遍历 1 次。

---

## Base case (i = 1)

$$m_1 = \max(-\infty,\; x_1) = x_1$$

$$d_1 = 0 \cdot e^{-\infty - x_1} + e^{x_1 - x_1} = 1$$

验证：

$$\sum_{j=1}^{1} e^{x_j - m_1} = e^{x_1 - x_1} = 1 \quad \checkmark$$

---

## Inductive step

### 归纳假设

处理完前 $i-1$ 个元素后成立：

$$d_{i-1} = \sum_{j=1}^{i-1} e^{x_j - m_{i-1}}, \quad m_{i-1} = \max(x_1, \dots, x_{i-1})$$

### 证明 i 时也成立

将递推公式展开：

$$d_i = d_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}$$

代入归纳假设：

$$d_i = \left[\sum_{j=1}^{i-1} e^{x_j - m_{i-1}}\right] \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}$$

利用指数加法律 $e^a \cdot e^b = e^{a+b}$，将 $e^{m_{i-1} - m_i}$ 乘进求和号：

$$d_i = \sum_{j=1}^{i-1} e^{(x_j - m_{i-1}) + (m_{i-1} - m_i)} + e^{x_i - m_i}$$

指数中 $m_{i-1}$ 被抵消：

$$x_j - m_{i-1} + m_{i-1} - m_i = x_j - m_i$$

因此：

$$d_i = \sum_{j=1}^{i-1} e^{x_j - m_i} + e^{x_i - m_i} = \sum_{j=1}^{i} e^{x_j - m_i} \quad \checkmark$$

同时 $m_i = \max(m_{i-1},\; x_i) = \max(x_1, \dots, x_i)$ 显然成立。$\checkmark$

---

## 本质

归纳步骤的核心就是一条指数律：

$$e^{x_j - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} = e^{x_j - m_{\text{new}}}$$

旧 max 在指数中被抵消，换成新 max。整体缩放是 $O(1)$ 的乘法，无需逐个修改历史项。

---

## 扩展到 Flash Attention

Attention 还需要加权和 $o = \sum_i \text{softmax}(s_i) \cdot v_i$，同理可在线维护：

$$o_i = o_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i} \cdot v_i$$

最终 $o = o_N / d_N$。修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 对 $o$ 和 $d$ 同时作用，保证比值正确。

---

## 三种方案对比：标准 / 朴素 Online / Flash Attention

### 设定

- $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 为序列长度，$d$ 为 head dim
- SRAM 大小为 $M$ 字节，远小于 $N^2$
- 所有数据初始在 HBM 中

---

### 方案 1：标准 Attention（三个独立 Kernel）

```python
# Kernel 1: 矩阵乘法（cuBLAS）
S = Q @ K.T                  # (N, N) — 写入 HBM

# Kernel 2: Softmax
P = softmax(S, dim=-1)       # (N, N) — 从 HBM 读 S，写 P 到 HBM

# Kernel 3: 矩阵乘法
O = P @ V                    # (N, d) — 从 HBM 读 P，写 O 到 HBM
```

| 项目 | 值 |
|------|----|
| 额外显存 | $O(N^2)$（存 S 和 P） |
| HBM IO | $O(N^2 + Nd)$（主要是读写 S、P 各一次） |
| 计算量 | $O(N^2 d)$ |

**问题**：N 大时 $N^2$ 的 S、P 矩阵占满显存，且读写它们是 IO 瓶颈。

---

### 方案 2：朴素 Online Attention（不存 S，逐行计算）

核心思想：一行一行处理 Q，每行用 online softmax 边扫描 K 边算，**不物化 S 矩阵**。

```python
# 伪代码，展示 HBM 访问模式
O = zeros(N, d)                           # 输出矩阵在 HBM

for i in range(N):                        # 遍历 Q 的每一行
    q_i = load_from_HBM(Q[i])             # load 1 行 Q: (1, d)

    m = -inf
    d_sum = 0.0
    o_i = zeros(d)

    for j in range(N):                    # 遍历 K 的每一行
        k_j = load_from_HBM(K[j])        # load 1 行 K: (1, d) ← 每个 i 都要重新 load!
        v_j = load_from_HBM(V[j])        # load 1 行 V: (1, d)

        # 计算 attention score
        s_ij = dot(q_i, k_j)             # 标量

        # online softmax 更新
        m_new = max(m, s_ij)
        d_sum = d_sum * exp(m - m_new) + exp(s_ij - m_new)
        o_i   = o_i   * exp(m - m_new) + exp(s_ij - m_new) * v_j
        m = m_new

    O[i] = o_i / d_sum                   # 写回 1 行到 HBM
```

| 项目 | 值 |
|------|----|
| 额外显存 | $O(d)$（只存一行的 $m, d, o_i$） |
| HBM IO | $O(N^2 d)$（对每一行 Q，都要 load 整个 K 和 V） |
| 计算量 | $O(N^2 d)$（和标准一样） |

**对比标准方案**：
- ✅ 显存从 $O(N^2)$ 降到 $O(d)$ — **大幅节省**
- ❌ HBM IO 从 $O(N^2 + Nd)$ 升到 $O(N^2 d)$ — **大幅恶化**
- K、V 被 load 了 $N$ 遍（每行 Q 都重新 load 一次完整的 K、V）

**这就是"省内存但费 IO"的中间产物。**

---

### 方案 3：Flash Attention（分块 + online softmax + kernel 融合）

关键改进：**用分块 (tiling) 把 K、V 的重复 load 次数从 $N$ 降到 $N/B_c$**，并让每次 load 的块刚好放进 SRAM。

```python
# Q 分成 T_r = ceil(N / B_r) 个块, 每块 B_r 行
# K, V 分成 T_c = ceil(N / B_c) 个块, 每块 B_c 行
# B_r, B_c 选择使得 Q块 + K块 + V块 + 中间变量 ≤ SRAM 大小 M

O = zeros(N, d)
M_vec = full(N, -inf)    # 每行的 running max
D_vec = zeros(N)          # 每行的 running sum

for j in range(T_c):                        # ← 外层遍历 K, V 的块
    K_j = load_from_HBM(K[j*B_c : (j+1)*B_c])   # load K 块到 SRAM: (B_c, d)
    V_j = load_from_HBM(V[j*B_c : (j+1)*B_c])   # load V 块到 SRAM: (B_c, d)

    for i in range(T_r):                     # ← 内层遍历 Q 的块
        Q_i = load_from_HBM(Q[i*B_r : (i+1)*B_r])   # load Q 块: (B_r, d)
        O_i = load_from_HBM(O[i*B_r : (i+1)*B_r])   # load 当前 O 块
        m_i = load_from_HBM(M_vec[i*B_r : (i+1)*B_r])
        d_i = load_from_HBM(D_vec[i*B_r : (i+1)*B_r])

        # --- 以下全在 SRAM 中计算 ---
        S_block = Q_i @ K_j.T                # (B_r, B_c) ← 小矩阵，放得下 SRAM

        m_new = max(m_i, rowmax(S_block))
        correction = exp(m_i - m_new)
        P_block = exp(S_block - m_new)       # (B_r, B_c)

        d_i = d_i * correction + rowsum(P_block)
        O_i = O_i * correction + P_block @ V_j

        m_i = m_new
        # --- SRAM 计算结束 ---

        store_to_HBM(O_i, m_i, d_i)          # 写回 HBM

# 最终归一化
O = O / D_vec
```

| 项目 | 值 |
|------|----|
| 额外显存 | $O(N)$（只存 $m$, $d$ 向量，不存 $N \times N$ 矩阵） |
| HBM IO | $O(N^2 d^2 / M)$（$M$ = SRAM 大小，远小于 $O(N^2 d)$） |
| 计算量 | $O(N^2 d)$（和标准一样） |

#### 为什么外层遍历 K/V，内层遍历 Q？

两种循环顺序的 IO 渐近复杂度相同（都是 $O(N^2 d^2 / M)$），区别在常数和访问模式：

```
方案 A: 外层 K/V，内层 Q（论文选择）
  K_j, V_j 在外层加载 → 每块只加载 1 次
  Q_i, O_i 在内层加载 → 每块加载 T_c 次（每轮外层都要遍历一遍）

  IO = T_c × 2·B_c·d          (K, V 各加载一次)
     + T_c × T_r × 3·B_r·d    (Q, O, 统计量每轮内层都要读写)

方案 B: 外层 Q，内层 K/V（反过来）
  Q_i, O_i 在外层加载 → 每块只加载 1 次
  K_j, V_j 在内层加载 → 每块加载 T_r 次（每轮外层都要遍历一遍）

  IO = T_r × 3·B_r·d          (Q, O, 统计量各加载一次)
     + T_r × T_c × 2·B_c·d    (K, V 每轮内层都要读)
```

两种方案的主项都是 $T_r \times T_c \times B \times d = O(N^2 d^2 / M)$，渐近复杂度一样。

论文选择方案 A 的实际考虑：**O 的写回模式更友好**。在方案 A 中，固定一个 K/V 块后，内层连续遍历所有 Q 块，每个 Q 块的 $O_i$ 被连续更新然后写回。而方案 B 中，固定一个 Q 块后，$O_i$ 需要随着每个 K/V 块不断读出-更新-写回，同一块 $O_i$ 被反复读写的间隔更短，对 HBM 带宽的利用更集中。

---

### 三种方案总结

| | 标准 | 朴素 Online | Flash Attention |
|--|------|------------|-----------------|
| 额外显存 | $O(N^2)$ | $O(d)$ | $O(N)$ |
| HBM IO | $O(N^2 + Nd)$ | $O(N^2 d)$ 😱 | $O(N^2 d^2 / M)$ ✅ |
| 计算量 | $O(N^2 d)$ | $O(N^2 d)$ | $O(N^2 d)$ |
| Kernel 数 | 3 个 | 1 个 | 1 个 |
| 物化 S 矩阵 | 是 | 否 | 否 |

朴素 Online 是 Flash Attention 的"半成品"：它解决了显存问题，但因为没有分块，IO 反而更差。Flash Attention 加上 tiling 后，IO 也优于标准方案（因为 $M$ 在分母，SRAM 越大 IO 越少）。

#### 朴素 Online 和 Flash Attention 的核心差异

两者都需要每个 Q 看到所有 K/V，区别在于 **K/V 能否被多个 Q 共享复用**：

```
朴素 Online（每个 Q 独占加载 K/V）:
  q_0 独立加载 K 全部 → 用完丢掉
  q_1 独立加载 K 全部 → 用完丢掉   ← 同样的 K，又加载一遍
  q_2 独立加载 K 全部 → 用完丢掉
  ...
  每个 Q 行各自加载一遍完整的 K/V → K/V 总共被加载 N 次

FlashAttention（多个 Q 共享加载 K/V）:
  加载 K_0 块到 SRAM → q_0, q_1, q_2, ... 全部 Q 块轮流用这份 K_0
  加载 K_1 块到 SRAM → q_0, q_1, q_2, ... 全部 Q 块轮流用这份 K_1
  ...
  每块 K/V 加载 1 次，服务所有 Q → K/V 总共被加载 T_c 次
```

每个 Q 最终**看到了所有 K/V**（分多轮看完），但 K/V 块在 SRAM 上**被复用了 $T_r$ 次**。

一句话总结：**朴素 Online 是每个 Q 独占一次 K/V 的加载，FlashAttention 是多个 Q 共享一次 K/V 的加载。** 这就是 IO 从 $O(N^2 d)$ 降到 $O(N^2 d^2 / M)$ 的本质原因。

---

## 补充：Memory Efficient Attention

来自 Rabe & Staub 2021 年的论文 *"Self-attention Does Not Need O(n²) Memory"*，定位在朴素 Online（方案 2）和 Flash Attention（方案 3）之间。

### 核心思路

和朴素 Online 一样，**不物化 $N \times N$ 的 S 矩阵**，用 online softmax 逐块计算。但比方案 2 更精细：

- 对 K/V 也做了**分块（chunking）**，每次只处理一个 chunk
- 用 **gradient checkpointing** 来节省反向传播的显存
- 显存从 $O(N^2)$ 降到 $O(\sqrt{N})$ 或 $O(N)$（取决于 chunk 大小）

### 和 Flash Attention 的关键区别

| | Memory Efficient Attention | Flash Attention |
|--|---|---|
| 优化目标 | 显存（memory） | IO（HBM 读写次数） |
| 是否感知硬件层级 | 否，不区分 SRAM/HBM | 是，tiling 大小由 SRAM 容量 $M$ 决定 |
| Kernel 实现 | 用 PyTorch 原生算子组合，依赖编译器 | 手写 CUDA kernel，融合所有操作 |
| HBM IO | 仍然是 $O(N^2 d)$ 级别 | $O(N^2 d^2 / M)$，显著更优 |
| 实际速度 | 省显存，但不一定快 | 既省显存又快（2-4x wallclock speedup） |

### 总结

Memory Efficient Attention 解决了**显存瓶颈**（不存 $N^2$ 矩阵），但没有解决 **IO 瓶颈**（没有针对 SRAM/HBM 层级做 tiling）。Flash Attention 在它的基础上加了 **IO-awareness**，把分块大小和硬件 SRAM 对齐，才同时解决了显存和速度两个问题。

在上面的框架里，Memory Efficient Attention ≈ 方案 2 的"工程加强版"（加了 chunking + gradient checkpointing），但本质上还没跨到方案 3 那一步。

---

## 补充：Q 和 K 的点乘

标准 Attention 中 Q 和 K 是**点乘（dot product）**，单对 $q_i$ 和 $k_j$ 点乘的结果是一个**标量**：

$$s_{ij} = q_i \cdot k_j = \sum_{l=1}^{d} q_{i,l} \cdot k_{j,l}$$

所有 $N^2$ 个配对做完后排成 $S \in \mathbb{R}^{N \times N}$ 矩阵，再除以 $\sqrt{d}$（scaled dot-product attention）防止点乘值过大导致 softmax 饱和：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

这里的"点乘"是向量内积（结果为标量），不是 element-wise 乘法（Hadamard product）。选点乘的原因是它能高效用矩阵乘实现，GPU 上有高度优化的 GEMM kernel。

---

## 补充：为什么 Online Attention 可以边扫描边乘 $v_j$

### Online Attention 的目标

标准 attention 对第 $i$ 行 query 的计算是：

$$o_i = \text{softmax}(q_i K^T) \cdot V = \sum_{j=1}^{N} \frac{e^{s_{ij} - m}}{\sum_k e^{s_{ik} - m}} \cdot v_j$$

其中 softmax 权重是标量，$v_j$ 是 $d$ 维向量，所以 $o_i$ 是 $d$ 维向量（N 个 $d$ 维向量的加权和）。

这需要先算完所有 score $s_{ij}$，再做 softmax，最后乘 V——三步串行。Online attention 的目标是把这三步**合并成一次遍历**：边扫描 K/V，边算 softmax 的分母 $d$，边累积输出 $o$，最后一除得到结果。

### $o_i$ 递推公式的推导

首先，把最终输出的分子分母拆开：

$$o_i^{(\text{final})} = \sum_{j=1}^{N} \frac{e^{s_{ij} - m}}{\sum_k e^{s_{ik} - m}} \cdot v_j = \frac{1}{\underbrace{\sum_k e^{s_{ik} - m}}_{d_N}} \cdot \underbrace{\sum_{j=1}^{N} e^{s_{ij} - m} \cdot v_j}_{o_N^{(\text{unnorm})}}$$

分子和分母可以**独立累加**，所以拆成两个量分别在线维护。固定某一行 query，用 $i$ 表示"已经处理了前 $i$ 个 key"，$s_j$ 表示当前 query 与第 $j$ 个 key 的 score（省略 query 行下标）。处理到第 $i$ 个 key 时，只看到前 $i$ 项：

$$d_i = \sum_{j=1}^{i} e^{s_j - m_i} \qquad o_i = \sum_{j=1}^{i} e^{s_j - m_i} \cdot v_j$$

其中 $o_i$ 是**中间状态的分子（未归一化）**，不是最终输出。扫完所有 $N$ 个 key 后，最终输出为 $O = o_N / d_N$。

现在推导 $o_i$ 的递推公式。和 $d_i$ 的递推完全一样，只是多乘了一个 $v_j$。

**第一步**：把第 $i$ 项（新元素）从求和中拆出来：

$$o_i = \sum_{j=1}^{i} e^{s_j - m_i} \cdot v_j = \underbrace{\sum_{j=1}^{i-1} e^{s_j - m_i} \cdot v_j}_{\text{历史项（但 max 变了）}} + \underbrace{e^{s_i - m_i} \cdot v_i}_{\text{第 i 个元素的贡献}}$$

注意这里有两种下标：$v_j$ 是求和里的循环变量（$j$ 从 1 到 $i-1$），$v_i$ 是单独拆出来的第 $i$ 个新元素。

**第二步**：处理历史项。我们手里存的是旧的 $o_{i-1} = \sum_{j=1}^{i-1} e^{s_j - m_{i-1}} \cdot v_j$，用的是旧 max $m_{i-1}$。而历史项需要的是新 max $m_i$。对求和中的每一项，乘修正因子把旧 max 换成新 max：

$$e^{s_j - m_{i-1}} \cdot v_j \cdot e^{m_{i-1} - m_i} = e^{(s_j - m_{i-1}) + (m_{i-1} - m_i)} \cdot v_j = e^{s_j - m_i} \cdot v_j$$

**第三步**：修正因子 $e^{m_{i-1} - m_i}$ 不依赖 $j$（对每一项都一样），所以可以提到求和号外面：

$$\sum_{j=1}^{i-1} e^{s_j - m_i} \cdot v_j = \sum_{j=1}^{i-1} e^{s_j - m_{i-1}} \cdot e^{m_{i-1} - m_i} \cdot v_j = e^{m_{i-1} - m_i} \cdot \underbrace{\sum_{j=1}^{i-1} e^{s_j - m_{i-1}} \cdot v_j}_{= \, o_{i-1}} = e^{m_{i-1} - m_i} \cdot o_{i-1}$$

**第四步**：把历史项和新元素代回第一步的拆分：

$$\boxed{o_i = o_{i-1} \cdot e^{m_{i-1} - m_i} + e^{s_i - m_i} \cdot v_i}$$

**本质**：$v_j$ 只是每项的"系数"，不影响修正因子的提取（因为修正因子只涉及 max 的变化，和 $v_j$ 无关）。$d_i$ 的递推相当于 $v_j = 1$ 的特殊情况。

### 核心问题

有了递推公式之后：

$$o_i = o_{i-1} \cdot e^{m_{i-1} - m_i} + e^{s_i - m_i} \cdot v_i$$

看起来 softmax 权重还没算完，为什么就能直接乘 $v_i$？

关键在于 $o_i$ 存的是**未归一化的加权和（分子）**，不是最终输出（含义见上面的定义）。

每处理一个新的 key（从第 $i-1$ 步到第 $i$ 步），两步操作：
1. 用 $e^{m_{i-1} - m_i}$ 把历史累加值（$o_{i-1}$ 和 $d_{i-1}$）缩放到新 max 下
2. 把第 $i$ 个元素的贡献直接加上去

分子和分母可以**独立累加**，因为求和运算满足结合律，每一项可以独立贡献，不需要等所有项到齐。

等全部 $N$ 个 key 扫完之后，**最后一步除法**才是真正的 softmax 归一化：

$$O = o_N / d_N$$

### 具体数值例子

这个例子只展示**一个 query** $q_i$ 的计算过程。完整的 $S = QK^T$ 是 $N \times N$ 矩阵（这里 $3 \times 3$），但 online attention 是逐行处理的，每行就是一个 $q_i$ 对所有 key 的点乘结果——3 个标量，不是矩阵。

假设 $N=3$（3 个 key），$d=2$（每个 value 是 2 维向量），当前 query $q_i$ 与 3 个 key 点乘后得到的 scores 和对应的 values 分别是：

$$s_1 = 2,\; s_2 = 4,\; s_3 = 3 \qquad v_1 = \begin{pmatrix}10\\1\end{pmatrix},\; v_2 = \begin{pmatrix}20\\2\end{pmatrix},\; v_3 = \begin{pmatrix}30\\3\end{pmatrix}$$

最终 $o$ 是一个 2 维向量：每个 value 按 softmax 权重加权求和，**逐分量独立计算**。

#### 标准做法（等全部算完再乘）

$$m = 4, \quad d = e^{-2} + 1 + e^{-1} \approx 0.135 + 1 + 0.368 = 1.503$$

$$w_1 \approx 0.090, \quad w_2 \approx 0.665, \quad w_3 \approx 0.245$$

$$o = 0.090 \times \begin{pmatrix}10\\1\end{pmatrix} + 0.665 \times \begin{pmatrix}20\\2\end{pmatrix} + 0.245 \times \begin{pmatrix}30\\3\end{pmatrix} = \begin{pmatrix}21.55\\2.155\end{pmatrix}$$

#### Online 做法（边扫描边累加）

**处理 $j=1$**：$s_1=2, v_1=(10, 1)^T$

$$m = 2, \quad d = 1, \quad o = e^{2-2} \times \begin{pmatrix}10\\1\end{pmatrix} = \begin{pmatrix}10\\1\end{pmatrix}$$

**处理 $j=2$**：$s_2=4, v_2=(20, 2)^T$

$$m_{\text{new}} = \max(2, 4) = 4$$

$$d = 1 \times e^{2-4} + e^{4-4} \approx 1.135$$

$$o = \begin{pmatrix}10\\1\end{pmatrix} \times e^{2-4} + e^{4-4} \times \begin{pmatrix}20\\2\end{pmatrix} = \begin{pmatrix}1.35\\0.135\end{pmatrix} + \begin{pmatrix}20\\2\end{pmatrix} = \begin{pmatrix}21.35\\2.135\end{pmatrix}$$

**处理 $j=3$**：$s_3=3, v_3=(30, 3)^T$

$$m_{\text{new}} = \max(4, 3) = 4$$

$$d = 1.135 + e^{3-4} = 1.503$$

$$o = \begin{pmatrix}21.35\\2.135\end{pmatrix} \times e^{4-4} + e^{3-4} \times \begin{pmatrix}30\\3\end{pmatrix} = \begin{pmatrix}21.35\\2.135\end{pmatrix} + \begin{pmatrix}11.04\\1.104\end{pmatrix} = \begin{pmatrix}32.39\\3.239\end{pmatrix}$$

**最后归一化**：

$$O = o / d = \begin{pmatrix}32.39\\3.239\end{pmatrix} / 1.503 = \begin{pmatrix}21.55\\2.155\end{pmatrix} \quad \checkmark$$

和标准做法结果一致。$o$ 是一个 $d$ 维向量，每个分量独立地做同样的"缩放 + 累加"操作，标量的修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 对每个分量都一样。

---

## Flash Decoding

### 背景：Flash Attention 在 Decode 阶段的短板

Flash Attention 沿 **Q 的 seq_len** 分块获得并行度，在 prefill 阶段（Q 有数千行）效果极好。但在 **decode 阶段**（自回归生成），每步只产生 1 个新 token：

```
q: (batch, n_heads, 1, head_dim)        ← 只有 1 行
k: (batch, n_heads, seq_len, head_dim)  ← 整个 KV cache，可能几千几万
v: (batch, n_heads, seq_len, head_dim)
```

Q 只有 1 行，沿 Q 分块退化为"1 块"，并行度只有 `batch × n_heads`。典型场景 batch=1、heads=32 时，只有 32 个线程块，A100 有 108 个 SM，**大量 SM 闲置**，GPU 利用率极低。

### 核心思路：沿 KV 的 seq_len 并行拆分

既然 Q 没得分，那就**分 KV**——把 KV cache 沿序列维度切成 $S$ 块，每块独立计算局部注意力，最后用 log-sum-exp 合并。

```
标准做法 (decode):
  1 个线程块串行扫描整个 KV cache → 并行度 = batch × n_heads

Flash Decoding:
  S 个线程块并行处理 KV 的不同块 → 并行度 = batch × n_heads × S
```

### 算法流程

#### 步骤 1：并行计算局部注意力（S 个线程块同时执行）

将 KV cache（假设 seq_len=4096）拆成 $S=128$ 块，每块 32 个 token：

```
Block 0:   q × K[0:32]^T / √d → local_scores_0
           partial_o_0   = softmax(local_scores_0) × V[0:32]    — 局部加权和 (head_dim,)
           partial_lse_0 = log(Σ exp(local_scores_0))           — 局部 log-sum-exp (标量)

Block 1:   q × K[32:64]^T / √d → local_scores_1
           partial_o_1   = softmax(local_scores_1) × V[32:64]
           partial_lse_1 = log(Σ exp(local_scores_1))

...

Block 127: q × K[4064:4096]^T / √d → ...
```

每个块输出两个东西：
- `partial_o_i`：局部注意力输出，形状 `(head_dim,)`
- `partial_lse_i`：局部 log-sum-exp，标量

#### 步骤 2：Reduction —— 用 log-sum-exp 正确合并

```python
# 所有块的 lse 收集起来
global_lse = log(Σ_i exp(partial_lse_i))

# 每块的合并权重
weight_i = exp(partial_lse_i - global_lse)

# 最终输出 = 各块按权重加权
final_o = Σ_i weight_i × partial_o_i
```

### 为什么 log-sum-exp 合并是正确的

每个块 $i$ 独立计算的局部结果：

$$\text{partial\_o}_i = \frac{\sum_{j \in \text{block}_i} e^{s_j} \cdot v_j}{\sum_{j \in \text{block}_i} e^{s_j}}, \quad \text{partial\_lse}_i = \log \sum_{j \in \text{block}_i} e^{s_j}$$

全局正确结果应该是：

$$o = \frac{\sum_{j=1}^{N} e^{s_j} \cdot v_j}{\sum_{j=1}^{N} e^{s_j}}$$

把分子按块拆开：

$$\sum_{j=1}^{N} e^{s_j} \cdot v_j = \sum_i \left(\sum_{j \in \text{block}_i} e^{s_j}\right) \cdot \text{partial\_o}_i = \sum_i e^{\text{lse}_i} \cdot \text{partial\_o}_i$$

分母同样按块拆开：

$$\sum_{j=1}^{N} e^{s_j} = \sum_i e^{\text{lse}_i}$$

所以：

$$o = \frac{\sum_i e^{\text{lse}_i} \cdot \text{partial\_o}_i}{\sum_i e^{\text{lse}_i}} = \sum_i \underbrace{\frac{e^{\text{lse}_i}}{\sum_k e^{\text{lse}_k}}}_{\text{weight}_i} \cdot \text{partial\_o}_i$$

每块的权重就是该块的 exp-sum 占全局 exp-sum 的比例。用 log-sum-exp 技巧避免数值溢出：

$$\text{weight}_i = \exp(\text{lse}_i - \text{global\_lse})$$

本质和 online softmax 的修正因子一样——**分子分母独立累加，最后一除**。

### 数值例子

假设 seq_len=6，拆成 2 块（每块 3 个 token），head_dim=2：

```
scores = [2, 4, 3, 1, 5, 2]
values = [[10,1], [20,2], [30,3], [40,4], [50,5], [60,6]]

Block 0: scores=[2,4,3], values=[[10,1],[20,2],[30,3]]
  exp_scores = [e², e⁴, e³] = [7.39, 54.60, 20.09]
  sum_exp = 82.08
  lse_0 = log(82.08) = 4.408
  partial_o_0 = (7.39×[10,1] + 54.60×[20,2] + 20.09×[30,3]) / 82.08
              = [1768.5/82.08, 176.85/82.08] ≈ [21.55, 2.155]

Block 1: scores=[1,5,2], values=[[40,4],[50,5],[60,6]]
  exp_scores = [e¹, e⁵, e²] = [2.72, 148.41, 7.39]
  sum_exp = 158.52
  lse_1 = log(158.52) = 5.066
  partial_o_1 = (2.72×[40,4] + 148.41×[50,5] + 7.39×[60,6]) / 158.52
              ≈ [49.77, 4.977]

合并:
  global_lse = log(exp(4.408) + exp(5.066)) = log(82.08 + 158.52) = log(240.60) = 5.483
  weight_0 = exp(4.408 - 5.483) = exp(-1.075) ≈ 0.341
  weight_1 = exp(5.066 - 5.483) = exp(-0.417) ≈ 0.659

  final_o = 0.341 × [21.55, 2.155] + 0.659 × [49.77, 4.977]
          ≈ [7.35, 0.735] + [32.80, 3.280]
          = [40.15, 4.015]
```

可以验证，这和不分块直接对所有 6 个 token 做标准 attention 的结果一致。

### 并行度对比

| | Flash Attention (decode) | Flash Decoding |
|---|---|---|
| 并行维度 | batch × n_heads | batch × n_heads × **S** |
| 典型值 (batch=1, heads=32, S=128) | 32 个线程块 | **4096** 个线程块 |
| A100 SM 利用率 (108 SMs) | ~30% | ~100% |

### 与 Flash Attention 的关系

| | Flash Attention | Flash Decoding |
|---|---|---|
| 优化阶段 | Prefill（长序列并行） | Decode（逐 token 生成） |
| 分块维度 | 沿 Q 的 seq_len | 沿 KV 的 seq_len |
| 解决的问题 | IO 瓶颈（避免物化 N² 矩阵） | 并行度不足（Q 只有 1 行） |
| 共同数学基础 | online softmax / log-sum-exp 修正 | 同左 |

两者**逻辑上互不依赖**（Flash Decoding 不需要 Flash Attention 也能工作），但实际部署中通常组合使用：prefill 用 Flash Attention，decode 用 Flash Decoding，覆盖推理全流程。

### Flash Decoding 的额外开销

分块并行不是免费的——多了一次 **reduction** 步骤（步骤 2）。但因为 reduction 只涉及 $S$ 个 `(head_dim,)` 向量的加权和，计算量远小于步骤 1 的矩阵乘法，属于可忽略的开销。真正的收益是把 GPU 利用率从 30% 拉到接近 100%，decode 延迟（TPOT）显著降低。
