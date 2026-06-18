"""
Chunked Prefill 优势场景 Benchmark
===================================

场景：模拟在线服务中"长短混合"的请求模式
- 短 prompt + 长 prompt 混合提交
- 通过 enable_chunked_prefill 开关对比

核心指标：
1. 整体耗时
2. 吞吐量（output tok/s）

对比方式：
- enable_chunked_prefill=False（全量 prefill，prefill/decode 互斥）
- enable_chunked_prefill=True（chunked prefill，prefill/decode 可混合）
"""

import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def bench(label, enable_chunked_prefill):
    seed(42)
    path = os.path.expanduser("//root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096,
              max_num_batched_tokens=6556,
              collect_metrics=True,
              enable_chunked_prefill=enable_chunked_prefill)

    # ---- 构造请求 ----
    # 一批短 prompt（模拟已有的在线请求，正在 decode）
    num_short = 64
    short_prompts = [[randint(0, 10000) for _ in range(randint(50, 200))] for _ in range(num_short)]
    short_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256) for _ in range(num_short)]

    # 几个超长 prompt（模拟突然到来的长请求，会阻塞 decode）
    num_long = 64
    long_prompts = [[randint(0, 10000) for _ in range(3500)] for _ in range(num_long)]
    long_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32) for _ in range(num_long)]

    # 合并：短请求在前，长请求在后
    all_prompts = short_prompts + long_prompts
    all_params = short_params + long_params

    # warmup
    llm.generate(["warmup"], SamplingParams())

    # ---- 运行 ----
    t0 = time.time()
    llm.generate(all_prompts, all_params, use_tqdm=False)
    elapsed = time.time() - t0

    total_output_tokens = sum(sp.max_tokens for sp in all_params)
    throughput = total_output_tokens / elapsed
    metrics = llm.scheduler.get_metrics()

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  enable_chunked_prefill = {enable_chunked_prefill}")
    print(f"  短请求: {num_short} 条, prompt 50~200 tok, output 256 tok")
    print(f"  长请求: {num_long} 条, prompt 3500 tok, output 32 tok")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  吞吐量: {throughput:.1f} tok/s")
    if metrics:
        print(f"  总步数: {metrics['step_count']}")
        print(f"  纯 prefill 步数: {metrics['pure_prefill_steps']}")
        print(f"  纯 decode 步数:  {metrics['pure_decode_steps']}")
        print(f"  混合步数:        {metrics['mixed_steps']}")
        print(f"  平均利用率:      {metrics['avg_utilization']}")
    print(f"{'='*50}\n")

    llm.exit()
    del llm
    return elapsed, throughput


if __name__ == "__main__":
    # 全量 prefill：prefill/decode 互斥，长 prompt 阻塞 decode
    t1, tp1 = bench("全量 Prefill (baseline)", enable_chunked_prefill=False)

    # Chunked prefill：长 prompt 分 chunk，decode 不被阻塞
    t2, tp2 = bench("Chunked Prefill", enable_chunked_prefill=True)

    print("\n" + "=" * 50)
    print("  对比结果")
    print("=" * 50)
    print(f"  全量 Prefill:    {t1:.2f}s, {tp1:.1f} tok/s")
    print(f"  Chunked Prefill: {t2:.2f}s, {tp2:.1f} tok/s")
    speedup = (t1 - t2) / t1 * 100
    if speedup > 0:
        print(f"  Chunked Prefill 快了 {speedup:.1f}%")
    else:
        print(f"  全量 Prefill 快了 {-speedup:.1f}%")
    print("=" * 50)
