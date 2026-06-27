"""
vLLM EAGLE3 speculative decoding acceptance rate 测试脚本。
对照 tests/verify_spec_correctness.py，使用相同的 prompts 和参数。

用法:
    # 方式1: 直接运行（自动打印 acceptance rate）
    python vllm_test.py

    # 方式2: 先启动 server，再查看日志中的 spec decoding stats
    # VLLM_LOGGING_LEVEL=INFO vllm serve /root/llm/model/Qwen3-4B \
    #   --spec-model /root/llm/model/Qwen3-4B_eagle3 \
    #   --spec-tokens 5 \
    #   --gpu-memory-utilization 0.9

输出:
    - 每轮 spec decoding 的 acceptance rate（从 vLLM engine stats 提取）
    - spec vs non-spec 输出文本对比
    - 端到端延迟对比
"""
import argparse
import os
import sys
import time

# ========== 配置 ==========
MODEL_PATH = "/root/llm/model/Qwen3-4B"
DRAFT_MODEL_PATH = "/root/llm/model/Qwen3-4B_eagle3"
MAX_TOKENS = 64

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
]


def run_baseline():
    """不带 speculative decoding 的基线测试"""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print("  Baseline (no speculative decoding)")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, sp)
    elapsed = time.perf_counter() - t0

    print(f"\nBaseline elapsed: {elapsed:.2f}s")
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        print(f"  Prompt {i}: {tokens} tokens, text={text[:80]!r}")

    # 释放显存
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return outputs, elapsed


def run_spec():
    """带 EAGLE3 speculative decoding 的测试"""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print("  Speculative decoding (EAGLE3)")
    print(f"  draft_model = {DRAFT_MODEL_PATH}")
    print(f"  num_speculative_tokens = 5")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL_PATH,
        spec_model=DRAFT_MODEL_PATH,
        spec_tokens=5,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        disable_log_stats=False,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, sp)
    elapsed = time.perf_counter() - t0

    print(f"\nSpec elapsed: {elapsed:.2f}s")
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        print(f"  Prompt {i}: {tokens} tokens, text={text[:80]!r}")

    # ===== 提取 spec decoding stats =====
    print(f"\n{'='*60}")
    print("  Spec Decoding Stats")
    print(f"{'='*60}")

    try:
        import numpy as np
        engine = llm.llm_engine
        logger_manager = getattr(engine, 'logger_manager', None)

        if logger_manager is None:
            print("  (logger_manager 不可用，可能未启用 stats logging)")
        else:
            # vLLM 0.23+ v1 架构: spec decoding 统计存储在
            # logger_manager.stat_loggers[*].spec_decoding_logging 中
            spec_log = None
            for sl in logger_manager.stat_loggers:
                spec_log = getattr(sl, 'spec_decoding_logging', None)
                if spec_log is None:
                    # PerEngineStatLoggerAdapter 包装的情况
                    per_engine = getattr(sl, 'per_engine_stat_loggers', None)
                    if per_engine:
                        spec_log = getattr(
                            per_engine.get(0), 'spec_decoding_logging', None)
                if spec_log is not None:
                    break

            if spec_log is not None:
                # 读取累积数据（log() 会 reset，所以先读）
                nd = spec_log.num_drafts
                ndt = spec_log.num_draft_tokens
                nat = spec_log.num_accepted_tokens
                num_drafts = int(np.sum(nd)) if nd else 0
                num_draft_tokens = int(np.sum(ndt)) if ndt else 0
                num_accepted = int(np.sum(nat)) if nat else 0

                if num_draft_tokens > 0:
                    rate = num_accepted / num_draft_tokens * 100
                    mean_len = 1 + (num_accepted / num_drafts) \
                        if num_drafts > 0 else 0
                    print(f"  num_drafts:             {num_drafts}")
                    print(f"  num_draft_tokens:       {num_draft_tokens}")
                    print(f"  num_accepted_tokens:    {num_accepted}")
                    print(f"  draft_acceptance_rate:  {rate:.1f}%")
                    print(f"  mean_acceptance_length: {mean_len:.2f}")
                else:
                    print("  (统计数据已在 generate 期间被周期性日志输出，"
                          "见上方 vLLM 日志)")

                # 强制 flush 剩余的 per-position 详细日志
                spec_log.log()
            else:
                print("  (未找到 spec_decoding_logging)")
    except Exception as e:
        print(f"  (stats extraction failed: {e})")

    # 释放显存
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return outputs, elapsed


def compare_outputs(baseline_outputs, spec_outputs, baseline_time, spec_time):
    """对比 spec 和 non-spec 输出"""
    print(f"\n{'='*60}")
    print("  Comparison")
    print(f"{'='*60}")

    all_match = True
    for i in range(len(PROMPTS)):
        b_text = baseline_outputs[i].outputs[0].text
        s_text = spec_outputs[i].outputs[0].text
        b_tokens = baseline_outputs[i].outputs[0].token_ids
        s_tokens = spec_outputs[i].outputs[0].token_ids
        match = (b_tokens == s_tokens)

        print(f"\nPrompt {i}: \"{PROMPTS[i][:50]}...\"")
        print(f"  Baseline ({len(b_tokens)} tokens): {b_text[:100]!r}")
        print(f"  Spec     ({len(s_tokens)} tokens): {s_text[:100]!r}")
        print(f"  Match: {'YES' if match else 'NO'}")

        if not match:
            all_match = False
            for j in range(max(len(b_tokens), len(s_tokens))):
                b = b_tokens[j] if j < len(b_tokens) else None
                s = s_tokens[j] if j < len(s_tokens) else None
                if b != s:
                    print(f"  First mismatch at position {j}: baseline={b}, spec={s}")
                    break

    print(f"\n{'='*60}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Spec time:     {spec_time:.2f}s")
    print(f"  Speedup:       {baseline_time/spec_time:.2f}x")
    if all_match:
        print("  ALL OUTPUTS MATCH: spec and non-spec are identical")
    else:
        print("  OUTPUTS DIFFER")
    print(f"{'='*60}")

    return 0 if all_match else 1


def main():
    global MODEL_PATH, DRAFT_MODEL_PATH, MAX_TOKENS
    parser = argparse.ArgumentParser(description="vLLM EAGLE3 acceptance rate test")
    parser.add_argument("--model", default=MODEL_PATH, help="Target model path")
    parser.add_argument("--draft-model", default=DRAFT_MODEL_PATH, help="EAGLE3 draft model path")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline run")
    args = parser.parse_args()

    MODEL_PATH = args.model
    DRAFT_MODEL_PATH = args.draft_model
    MAX_TOKENS = args.max_tokens

    if args.skip_baseline:
        spec_outputs, spec_time = run_spec()
        return 0

    baseline_outputs, baseline_time = run_baseline()
    spec_outputs, spec_time = run_spec()
    return compare_outputs(baseline_outputs, spec_outputs, baseline_time, spec_time)


if __name__ == "__main__":
    # 设置日志级别，确保 spec decoding stats 被打印
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    sys.exit(main())
