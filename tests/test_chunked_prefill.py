"""
Chunked Prefill 功能验证
========================
用相同的输入和固定随机种子，分别在 enable_chunked_prefill=True/False 下生成，
验证两者输出 token_ids 完全一致。
"""

import os
from random import randint, seed
from nanovllm import LLM, SamplingParams


def run(enable_chunked_prefill):
    seed(0)
    path = os.path.expanduser("//root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=True, max_model_len=4096,
              enable_chunked_prefill=enable_chunked_prefill)

    # 构造多种长度的 prompt，覆盖短/中/长
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(50)],     # 短
        [randint(0, 10000) for _ in range(500)],    # 中
        [randint(0, 10000) for _ in range(2000)],   # 长（会被 chunk）
        [randint(0, 10000) for _ in range(3500)],   # 超长
    ]
    # 贪心采样，确保输出确定性
    sampling_params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=64)

    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    result = [out["token_ids"] for out in outputs]
    llm.exit()
    del llm
    return result


if __name__ == "__main__":
    print("Running with enable_chunked_prefill=False ...")
    result_off = run(False)

    print("Running with enable_chunked_prefill=True ...")
    result_on = run(True)

    all_match = True
    for i, (off, on) in enumerate(zip(result_off, result_on)):
        if off == on:
            print(f"  Seq {i}: PASS ({len(off)} tokens)")
        else:
            print(f"  Seq {i}: FAIL")
            print(f"    OFF: {off[:10]}...")
            print(f"    ON:  {on[:10]}...")
            all_match = False

    if all_match:
        print("\nAll sequences match! Chunked prefill is functionally correct.")
    else:
        print("\nMISMATCH detected! Chunked prefill has a bug.")
