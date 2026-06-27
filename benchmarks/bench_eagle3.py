"""
Benchmark speculative decoding on/off.

Usage:
    python benchmarks/bench_eagle3.py \
        --model /root/llm/model/Qwen3-4B \
        --draft-model /root/llm/model/Qwen3-4B_eagle3 \
        --num-prompts 16 \
        --prompt-len 512 \
        --max-tokens 128 \
        --no-chunked-prefill 
"""

import argparse
import gc
import os
import random
import sys
from dataclasses import dataclass
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm import LLM, SamplingParams


TEXT_PROMPT_SEEDS = [
    "Explain how attention mechanisms help language models reason over long context. Include a concrete example.",
    "Write a Python implementation plan for a small HTTP cache, including data structures and edge cases.",
    "Summarize the causes and consequences of the Industrial Revolution in clear chronological order.",
    "Compare TCP and UDP for a backend service engineer deciding how to build a realtime messaging system.",
    "Describe how matrix multiplication is optimized on GPUs, focusing on memory locality and tiling.",
    "Give a concise tutorial on debugging a distributed system when latency suddenly increases.",
    "Explain speculative decoding for large language model inference and discuss when it helps performance.",
    "Write a short technical design for batching requests in an inference engine with KV cache management.",
]

TEXT_FILLER = (
    "The explanation should be precise, practical, and organized. "
    "Use natural language, avoid random symbols, and keep the discussion coherent. "
    "When useful, include tradeoffs, examples, and implementation details. "
)


@dataclass
class BenchResult:
    label: str
    elapsed: float
    num_prompts: int
    target_output_tokens: int
    actual_output_tokens: int
    step_count: int
    prefill_tokens: int
    decode_tokens: int
    scheduler_metrics: dict
    outputs: list[list[int]]

    @property
    def requests_per_second(self) -> float:
        return self.num_prompts / self.elapsed

    @property
    def target_output_tps(self) -> float:
        return self.target_output_tokens / self.elapsed

    @property
    def actual_output_tps(self) -> float:
        return self.actual_output_tokens / self.elapsed


def make_random_token_prompts(num_prompts: int, prompt_len: int, token_upper: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(0, token_upper) for _ in range(prompt_len)] for _ in range(num_prompts)]


def make_text_prompt(index: int, approx_prompt_tokens: int) -> str:
    seed_text = TEXT_PROMPT_SEEDS[index % len(TEXT_PROMPT_SEEDS)]
    # English text is roughly 1.2-1.5 tokens/word for Qwen-style tokenizers.
    # Repeat coherent filler instead of random token ids so EAGLE sees an in-distribution context.
    target_words = max(16, int(approx_prompt_tokens * 0.75))
    words = seed_text.split()
    filler_words = TEXT_FILLER.split()
    while len(words) < target_words:
        words.extend(filler_words)
    return " ".join(words[:target_words])


def make_text_prompts(num_prompts: int, approx_prompt_tokens: int) -> list[str]:
    return [make_text_prompt(i, approx_prompt_tokens) for i in range(num_prompts)]


def make_warmup_prompt(args: argparse.Namespace) -> str | list[int]:
    if args.prompt_mode == "random-token":
        return make_random_token_prompts(1, args.prompt_len, args.token_upper, args.seed + 1000003)[0]
    return make_text_prompt(10_000, min(args.prompt_len, 256))


def cleanup_llm(llm: LLM | None) -> None:
    if llm is not None:
        llm.exit()
        del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def run_once(
    label: str,
    args: argparse.Namespace,
    prompts: list[str] | list[list[int]],
    draft_model: str | None,
) -> BenchResult:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  draft_model = {draft_model}")
    print(f"{'=' * 60}")

    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass

    llm = None
    try:
        llm_kwargs = dict(
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_chunked_prefill=args.enable_chunked_prefill,
            spec_profile=args.spec_profile,
            spec_debug=args.spec_debug,
            collect_metrics=True,
        )
        if draft_model:
            llm_kwargs["draft_model"] = draft_model
            llm_kwargs["num_spec_tokens"] = args.num_spec_tokens

        llm = LLM(args.model, **llm_kwargs)

        if args.warmup_tokens > 0:
            warmup_params = SamplingParams(
                temperature=0,
                ignore_eos=True,
                max_tokens=args.warmup_tokens,
            )
            llm.generate([make_warmup_prompt(args)], warmup_params, use_tqdm=False)
            llm.scheduler.reset_metrics()
            if args.spec_profile:
                llm.model_runner.call("reset_spec_profile_metrics")

        sampling_params = [
            SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.max_tokens)
            for _ in prompts
        ]

        outputs: dict[int, list[int]] = {}
        prefill_tokens = 0
        decode_tokens = 0
        step_count = 0

        for prompt, sp in zip(prompts, sampling_params):
            llm.add_request(prompt, sp)

        start = perf_counter()
        while not llm.is_finished():
            finished, num_prefill_tokens, num_decode_tokens = llm.step(verbose=args.verbose)
            step_count += 1
            prefill_tokens += num_prefill_tokens
            decode_tokens += num_decode_tokens
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids
        elapsed = perf_counter() - start

        output_token_ids = [outputs[seq_id] for seq_id in sorted(outputs)]
        actual_output_tokens = sum(len(token_ids) for token_ids in output_token_ids)
        target_output_tokens = len(prompts) * args.max_tokens
        scheduler_metrics = llm.scheduler.get_metrics()

        return BenchResult(
            label=label,
            elapsed=elapsed,
            num_prompts=len(prompts),
            target_output_tokens=target_output_tokens,
            actual_output_tokens=actual_output_tokens,
            step_count=step_count,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            scheduler_metrics=scheduler_metrics,
            outputs=output_token_ids,
        )
    finally:
        cleanup_llm(llm)


def compare_outputs(baseline: BenchResult, spec: BenchResult, max_tokens: int) -> None:
    mismatches = 0
    length_mismatches = 0
    for base_ids, spec_ids in zip(baseline.outputs, spec.outputs):
        if len(base_ids) != len(spec_ids):
            length_mismatches += 1
        if base_ids[:max_tokens] != spec_ids[:max_tokens]:
            mismatches += 1

    print("\nOutput check (greedy, compared up to --max-tokens):")
    print(f"  mismatched prompts: {mismatches}/{baseline.num_prompts}")
    if length_mismatches:
        print(f"  length mismatches:  {length_mismatches}/{baseline.num_prompts}")


def print_result(result: BenchResult) -> None:
    print(f"\n{result.label}:")
    print(f"  elapsed:             {result.elapsed:.2f}s")
    print(f"  requests/s:          {result.requests_per_second:.2f}")
    print(f"  target output tok/s: {result.target_output_tps:.2f}")
    print(f"  actual output tok/s: {result.actual_output_tps:.2f}")
    print(f"  target output toks:  {result.target_output_tokens}")
    print(f"  actual output toks:  {result.actual_output_tokens}")
    print(f"  engine steps:        {result.step_count}")
    print(f"  prefill tokens:      {result.prefill_tokens}")
    print(f"  decode tokens:       {result.decode_tokens}")
    if result.scheduler_metrics:
        metrics = result.scheduler_metrics
        print(f"  scheduler steps:     {metrics['step_count']}")
        print(f"  pure prefill steps:  {metrics['pure_prefill_steps']}")
        print(f"  pure decode steps:   {metrics['pure_decode_steps']}")
        print(f"  mixed steps:         {metrics['mixed_steps']}")
        print(f"  avg utilization:     {metrics['avg_utilization']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare nano-vLLM performance with spec on/off")
    parser.add_argument("--model", required=True, help="Target model path")
    parser.add_argument("--draft-model", required=True, help="EAGLE3 draft model path")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-spec-tokens", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--prompt-mode", choices=["text", "random-token"], default="text",
                        help="Use natural text prompts by default; random-token is mostly for stress testing/OOD behavior")
    parser.add_argument("--token-upper", type=int, default=10000, help="Upper bound for random prompt token ids")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    parser.add_argument("--skip-output-check", action="store_true")
    parser.add_argument("--spec-profile", action="store_true",
                        help="Enable speculative decoding timing with CUDA synchronizations")
    parser.add_argument("--spec-debug", action="store_true",
                        help="Enable EAGLE3/Qwen3 speculative decoding debug prints")
    parser.add_argument("--verbose", action="store_true", help="Print per-step engine/spec details")
    parser.set_defaults(enable_chunked_prefill=True)
    args = parser.parse_args()

    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be > 0")
    if args.prompt_len <= 0:
        raise ValueError("--prompt-len must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.prompt_mode == "random-token" and args.token_upper <= 0:
        raise ValueError("--token-upper must be > 0")

    if args.prompt_mode == "random-token":
        prompts = make_random_token_prompts(args.num_prompts, args.prompt_len, args.token_upper, args.seed)
    else:
        prompts = make_text_prompts(args.num_prompts, args.prompt_len)

    print(f"Prompt mode: {args.prompt_mode}")
    if args.prompt_mode == "text":
        print("  using natural text prompts (recommended for spec decoding benchmark)")
    else:
        print("  using random token ids (OOD; useful only for stress testing, not spec speedup)")

    baseline = run_once("Baseline (spec off)", args, prompts, draft_model=None)
    spec = run_once("Speculative decoding (spec on)", args, prompts, draft_model=args.draft_model)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print_result(baseline)
    print_result(spec)

    elapsed_speedup = (baseline.elapsed - spec.elapsed) / baseline.elapsed * 100
    tps_speedup = (spec.target_output_tps - baseline.target_output_tps) / baseline.target_output_tps * 100
    print("\nComparison:")
    print(f"  elapsed speedup:     {elapsed_speedup:+.1f}%")
    print(f"  target tok/s change: {tps_speedup:+.1f}%")

    if not args.skip_output_check:
        compare_outputs(baseline, spec, args.max_tokens)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
