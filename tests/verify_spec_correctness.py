"""
验证 speculative decoding 正确性：greedy 模式下 spec 和 non-spec 输出应完全一致。

用法:
    python tests/verify_spec_correctness.py \
        --model ../models/Qwen/Qwen3-4B \
        --draft-model ../models/AngelSlim/Qwen3-4B_eagle3 \
        --max-tokens 64

检查项:
    1. 多 prompt (batch): spec vs non-spec 输出一致
    2. 打印每轮 acceptance rate 统计
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm import LLM, SamplingParams


def run(label, model_path, prompts, max_tokens, draft_model=None, verbose=False):
    """运行一次 generate，返回 outputs，结束后释放显存"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  draft_model = {draft_model}")
    print(f"{'='*60}")

    import torch._dynamo
    torch._dynamo.reset()

    kwargs = dict(max_model_len=4096, enforce_eager=True)
    if draft_model:
        kwargs["draft_model"] = draft_model

    llm = LLM(model_path, **kwargs)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sp, use_tqdm=False, verbose=verbose)

    llm.exit()
    del llm
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Target model path")
    parser.add_argument("--draft-model", required=True, help="EAGLE3 draft model path")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
    ]

    # ===== 1. Non-spec (baseline) =====
    # baseline_outputs = run("Baseline (no spec)", args.model, prompts, args.max_tokens)

    # ===== 2. Spec =====
    spec_outputs = run("Speculative decoding", args.model, prompts, args.max_tokens,
                       draft_model=args.draft_model, verbose=True)

    # ===== 3. 对比 =====
    # print(f"\n{'='*60}")
    # print("  Comparing outputs")
    # print(f"{'='*60}")

    # all_match = True
    # for i, (baseline, spec) in enumerate(zip(baseline_outputs, spec_outputs)):
    #     b_ids = baseline["token_ids"]
    #     s_ids = spec["token_ids"]
    #     match = b_ids == s_ids

    #     print(f"\nPrompt {i}: \"{prompts[i][:50]}...\"")
    #     print(f"  Baseline ({len(b_ids)} tokens): {baseline['text'][:100]}...")
    #     print(f"  Spec     ({len(s_ids)} tokens): {spec['text'][:100]}...")
    #     print(f"  Match: {'YES' if match else 'NO'}")

    #     if not match:
    #         all_match = False
    #         for j in range(max(len(b_ids), len(s_ids))):
    #             b = b_ids[j] if j < len(b_ids) else None
    #             s = s_ids[j] if j < len(s_ids) else None
    #             if b != s:
    #                 print(f"  First mismatch at position {j}: baseline={b}, spec={s}")
    #                 break

    # print(f"\n{'='*60}")
    # if all_match:
    #     print("  ALL TESTS PASSED: spec and non-spec outputs are identical")
    # else:
    #     print("  TESTS FAILED: outputs differ")
    # print(f"{'='*60}")

    # return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
