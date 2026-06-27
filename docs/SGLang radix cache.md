# SGLang Radix Cache：压缩 Trie / Radix Tree 基础

## 什么是压缩 Trie / Radix Tree

压缩 Trie，也叫 **radix tree** 或 **Patricia trie**，是一种优化版 Trie。

普通 Trie 通常是：

> 一个字符一层。

压缩 Trie 的核心优化是：

> 把只有一个分支的连续节点压缩成一条边。

也就是说，radix tree 的边上存的不是单个字符，而是一段字符串片段。

## 示例

假设要存这些字符串：

```text
car
card
care
cat
```

普通 Trie 可以表示成：

```text
c
└── a
    ├── r
    │   ├── end
    │   ├── d
    │   └── e
    └── t
```

压缩 Trie / radix tree 会把没有分叉的链压缩掉，例如：

```text
ca
├── r
│   ├── end
│   ├── d
│   └── e
└── t
```

其中 `ca` 就是把普通 Trie 里的 `c -> a` 这条无分叉路径压缩成了一条边。

再比如：

```text
inter
internet
internal
```

普通 Trie 里会有很长的单字符链：

```text
i -> n -> t -> e -> r -> ...
```

压缩后可以表示成：

```text
inter
├── end
├── net
└── nal
```

## `end` 是什么意思

`end` 表示：

> 这里有一个完整字符串到此结束。

例如：

```text
car
card
care
cat
```

走到：

```text
c -> a -> r
```

时，已经匹配到了完整字符串 `car`，所以需要有一个结束标记。

否则如果只有：

```text
c -> a -> r -> d
c -> a -> r -> e
```

只能知道有 `card` 和 `care`，但不知道 `car` 本身也是一个合法字符串。

## 为什么有些单词没有画 `end`

严格来说，**每个完整字符串都应该有结束标记**。

更完整的表示应该是：

```text
ca
├── r
│   ├── end        表示 car
│   ├── d
│   │   └── end    表示 card
│   └── e
│       └── end    表示 care
└── t
    └── end        表示 cat
```

有些示意图会省略叶子节点的 `end`，因为叶子天然表示一个字符串结束。

但是当一个字符串是另一个字符串的前缀时，必须显式标记 `end`。

例如：

```text
car
card
```

这里 `car` 是 `card` 的前缀，所以在 `car` 对应的位置必须有结束标记，否则无法区分：

- 只有 `card`
- 同时有 `car` 和 `card`

## 实际代码里的表示

实际实现里通常不会真的创建一个叫 `end` 的子节点，而是在节点上存一个字段，例如：

```python
class Node:
    children: dict
    is_end: bool
    value: object | None
```

对于 key-value 存储，`end` 位置通常还会存这个 key 对应的 value。

也就是说：

- `is_end = True` 表示这个位置是一个完整 key 的结尾。
- `value != None` 表示这个完整 key 对应某个值。

## SGLang Radix Cache 和 vLLM KV Cache 对比

### 一句话区别

vLLM 的 automatic prefix caching 更接近：

```text
完整 KV block 的 hash -> physical KV block
```

SGLang 的 radix cache 更接近：

```text
token prefix -> KV slot / page / block indices
```

所以：

- vLLM 通常是 **block 级复用**。
- SGLang 是 **token-prefix 级匹配**，底层 KV 仍然可以按 page/block 存。

这里不要把两者理解成完全替代关系：

```text
PagedAttention / paged KV:
    解决 KV 怎么按 page/block 存、怎么被 attention kernel 访问。

Radix cache:
    解决请求来了以后，prompt 的哪些 prefix 已经算过、可以直接复用。
```

Radix cache 可以叠在 paged KV pool 之上。

### vLLM 怎么判断 prefix 是否一样

vLLM 的 prefix cache 通常按 block 做 hash。

假设 block size = 16：

```text
prompt tokens = [1, 2, ..., 16, 17, ..., 32, 33]
```

切成：

```text
block0 = [1..16]
block1 = [17..32]
partial = [33]
```

然后对完整 block 建 hash：

```text
hash(block0 tokens + extra metadata)
hash(block1 tokens + previous block hash + extra metadata)
```

命中时复用完整 block 的 KV。

因此如果两个请求共享 23 个 token，block size = 16，那么 vLLM 稳定可复用的一般是：

```text
floor(23 / 16) * 16 = 16 tokens
```

最后 7 个 partial block token 通常需要重新 prefill。

### SGLang 怎么判断 prefix 是否一样

SGLang radix cache 维护一棵 radix tree，边上存 token 序列片段。

例如已有：

```text
root
 └── [1, 2, 3, 4, 5] -> [kv10, kv11, kv12, kv13, kv14]
```

新请求：

```text
[1, 2, 3, 9]
```

它会沿树做 longest prefix match，并直接比较 token ids：

```text
common_prefix([1,2,3,4,5], [1,2,3,9]) = [1,2,3]
```

然后 split radix node：

```text
root
 └── [1, 2, 3] -> [kv10, kv11, kv12]
      ├── [4, 5] -> [kv13, kv14]
      └── [9]    -> [new_kv]
```

这个 split 主要是 CPU 侧元信息更新：

- token span 被切成公共 prefix 和 suffix。
- KV index list 被切成两段。
- GPU 上真实的 KV tensor / KV slot 通常不搬。

### 显存分配方式

SGLang 通常会预先维护一个 GPU KV pool：

```text
KV pool:
slot 0
slot 1
slot 2
...
slot N
```

radix tree 里存的是：

```text
token prefix -> KV slot/page/block indices
```

请求命中 prefix 后，请求自己的 KV 映射可能是：

```text
token pos:  0   1   2   3   4   5
kv slot:   10  11  12  90  91  300
```

含义：

```text
pos 0-2: 复用 radix cache 里的 prefix KV
pos 3-4: 当前请求新算的 suffix KV
pos 5:   decode append 的新 KV
```

所以它有点像 COW，但不是传统写时复制：

```text
传统 COW:
    共享 page，写的时候复制。

Radix cache:
    历史 prefix KV 是 immutable，只读共享；新 token append 到新的 KV slot。
```

真正要保护的是共享 KV slot 的生命周期：

- active request 正在读的 KV 不能释放。
- radix tree 中仍可复用的 KV 不能被错误回收。
- eviction 时需要根据 refcount / lock / LRU 等信息释放安全的节点。

### 和 PagedAttention 的关系

PagedAttention 的核心是：

```text
logical block -> physical KV block
```

也就是逻辑上连续的 sequence，物理上可以分散在不同 page/block 中。

Radix cache 的核心是：

```text
token prefix -> 可复用的 KV indices
```

所以两者层次不同：

```text
Radix cache:
    找到哪些 prefix KV 可以复用。

PagedAttention:
    让 attention kernel 高效访问这些可能不连续的 KV blocks。
```

SGLang RadixAttention 可以理解成在 paged/block KV 管理之上，再加了一层 prefix reuse。

### 多轮对话里的差异

对于普通 append-only 多轮对话：

```text
common_prefix_len = L
block_size = B
```

vLLM 大致可复用：

```text
floor(L / B) * B
```

SGLang radix cache 大致可复用：

```text
L
```

差距通常小于一个 block：

```text
L - floor(L / B) * B < B
```

如果 `B = 16`，单次请求最多也就差 0 到 15 个 token。

所以在普通单线多轮 chat 里，vLLM 的 block prefix cache 已经很接近 radix cache；SGLang 的主要额外计算收益往往就是最后那个不完整 block。

### 高 fanout 场景为什么差距会放大

高 fanout 指：

```text
同一个 prefix 后面分叉出很多不同请求 / 候选分支。
```

例如：

```text
shared prefix
 ├── branch 1
 ├── branch 2
 ├── branch 3
 ├── ...
 └── branch 1000
```

常见于：

- parallel sampling
- best-of-N
- beam search / tree search
- agent planning
- 同一个 prompt 多个 temperature / seed 的评测

如果 vLLM 因为 block 粒度每个分支多算最多 15 tokens，那么：

```text
fanout = 1000
extra prefill = 15 * 1000 = 15000 tokens
```

这时 radix cache 的 token-prefix 级复用优势会被放大。

### SGLang radix cache 相对 vLLM block prefix cache 的优势

1. **复用粒度更细**

   ```text
   vLLM:    完整 block 级复用
   SGLang:  token prefix 级匹配
   ```

2. **更自然表达 prefix tree**

   Radix tree 可以直接表示：

   ```text
   system prompt
    ├── conversation A
    ├── conversation B
    └── conversation C
   ```

3. **适合高共享 prefix workload**

   例如长 system prompt、few-shot、agent 多分支、parallel sampling。

4. **非 block 对齐 prefix 更友好**

   如果公共 prefix 长度不是 block size 的整数倍，radix cache 可以逻辑上复用到任意 token 边界。

### SGLang radix cache 的代价 / 劣势

这些劣势不是说它比 PagedAttention 更差，而是相比只做 block/page KV 管理，多了一层 radix prefix cache 的管理成本。

1. **CPU metadata 更复杂**

   需要维护：

   ```text
   radix node split / merge
   children map
   parent pointer
   lock_ref / refcount
   last_access_time
   LRU eviction
   ```

2. **eviction 更受约束**

   Radix tree 中间节点可能被多个子分支依赖，通常更适合 evict unlocked leaf nodes。

   例如：

   ```text
   root
    └── system prompt
         ├── branch A
         ├── branch B
         └── branch C
   ```

   `system prompt` 这个共享 prefix 不能随便释放，否则子分支都会失效。

3. **低 prefix 复用时收益小**

   如果每个请求 prompt 都不同，radix cache 命中率低，只剩 lookup / insert / eviction 成本。

4. **batch shape 更动态**

   每个请求命中的 prefix 长度可能不同：

   ```text
   req1 hit 4096 tokens
   req2 hit 128 tokens
   req3 hit 0 tokens
   req4 hit 7000 tokens
   ```

   剩余要 prefill 的 suffix 长度也不同，这会让调度、bucket、CUDA graph 更难。

5. **共享 KV 生命周期更复杂**

   一个 KV page/block 可能同时被：

   - active request 使用；
   - radix cache node 持有；
   - 多个后续请求共享。

   因此必须精确维护引用和锁，避免 KV 被提前回收。

### 总结

如果对比的是：

```text
SGLang RadixAttention vs 只有 PagedAttention、没有 prefix cache
```

SGLang 的收益可能很大，因为它可以省掉大量重复 prefill。

如果对比的是：

```text
SGLang radix cache vs vLLM automatic prefix caching
```

两者都能复用 prefix，差异主要变成：

```text
vLLM:    block hash，完整 block 复用
SGLang:  radix tree，token-prefix 级匹配
```

在普通 append-only 多轮对话中，vLLM 的主要损失通常只是最后一个不完整 block。

在高 fanout、短 suffix、大量共享 prefix、非 block 对齐复用的 workload 中，SGLang radix cache 的优势会更明显。
