"""
验证 Chunked Prefill 调度正确性
"""
import os
from time import perf_counter
from random import randint, seed
from nanovllm import LLM, SamplingParams


def verify(label, enable_chunked_prefill):
    seed(42)
    path = os.path.expanduser("//root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B")
    budget = 4096
    llm = LLM(path, enforce_eager=True, max_model_len=4096,
              max_num_batched_tokens=budget,
              collect_metrics=True,
              enable_chunked_prefill=enable_chunked_prefill)

    # 少量请求验证调度
    num_short = 8
    short_prompts = [[randint(0, 10000) for _ in range(randint(50, 200))] for _ in range(num_short)]
    short_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64) for _ in range(num_short)]

    num_long = 2
    long_prompts = [[randint(0, 10000) for _ in range(3500)] for _ in range(num_long)]
    long_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16) for _ in range(num_long)]

    all_prompts = short_prompts + long_prompts
    all_params = short_params + long_params

    # warmup
    llm.generate(["warmup"], SamplingParams())

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  enable_chunked_prefill = {enable_chunked_prefill}")
    print(f"  短请求: {num_short} 条, output 64 tok")
    print(f"  长请求: {num_long} 条, prompt 3500 tok, output 16 tok")
    print(f"  budget = {budget}")
    print(f"{'='*60}")
    print("前 15 步调度情况：\n")

    # 添加请求
    for prompt, sp in zip(all_prompts, all_params):
        llm.add_request(prompt, sp)

    # 手动跑前 15 步，打印调度
    for i in range(15):
        if llm.is_finished():
            break

        t0 = perf_counter()
        prefill_seqs, decode_seqs = llm.scheduler.schedule()
        llm.scheduler.record_step(prefill_seqs, decode_seqs)
        prefill_tokens = sum(s.scheduled_chunk_size for s in prefill_seqs) if prefill_seqs else 0
        decode_tokens = len(decode_seqs)

        token_ids = llm.model_runner.call("run", prefill_seqs, decode_seqs)
        llm.scheduler.postprocess(prefill_seqs, decode_seqs, token_ids)
        step_time = perf_counter() - t0

        utilization = (prefill_tokens + decode_tokens) / budget * 100

        # 分类统计
        if prefill_seqs and decode_seqs:
            step_type = "mixed "
        elif prefill_seqs:
            step_type = "prefill"
        else:
            step_type = "decode"

        print(f"Step {i+1:2d}: prefill={len(prefill_seqs):2d} ({prefill_tokens:4d} tok), decode={len(decode_seqs):2d} ({decode_tokens:2d} tok), util={utilization:5.1f}%, time={step_time*1000:6.1f}ms [{step_type}]")

    # 跑完剩余请求
    while not llm.is_finished():
        llm.step()

    metrics = llm.scheduler.get_metrics()
    print(f"\n{'='*60}")
    print(f"  统计汇总 ({metrics['step_count']} steps)")
    print(f"{'='*60}")
    print(f"  纯 prefill 步数: {metrics['pure_prefill_steps']}")
    print(f"  纯 decode 步数:  {metrics['pure_decode_steps']}")
    print(f"  混合步数:        {metrics['mixed_steps']}")
    print(f"  平均利用率:      {metrics['avg_utilization']}")
    print(f"{'='*60}\n")

    llm.exit()
    del llm


if __name__ == "__main__":
    verify("全量 Prefill", enable_chunked_prefill=False)
    verify("Chunked Prefill", enable_chunked_prefill=True)
