#!/bin/sh
set -eu

MODEL=${MODEL:-/root/repos/models/Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-/root/repos/models/AngelSlim/Qwen3-4B_eagle3}
NUM_PROMPTS=${NUM_PROMPTS:-8}
PROMPT_LEN=${PROMPT_LEN:-512}
MAX_TOKENS=${MAX_TOKENS:-128}
KS=${KS:-"2 3 4 5"}

cd "$(dirname "$0")/.."

for K in ${KS}; do
  echo
  echo "============================================================"
  echo "  EAGLE3 benchmark: num_spec_tokens=${K}"
  echo "============================================================"
  python benchmarks/bench_eagle3.py \
    --model "${MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --prompt-len "${PROMPT_LEN}" \
    --max-tokens "${MAX_TOKENS}" \
    --num-spec-tokens "${K}"
done

if [ -n "${BEST_K:-}" ]; then
  echo
  echo "============================================================"
  echo "  EAGLE3 profile: num_spec_tokens=${BEST_K}"
  echo "============================================================"
  python benchmarks/bench_eagle3.py \
    --model "${MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --prompt-len "${PROMPT_LEN}" \
    --max-tokens "${MAX_TOKENS}" \
    --num-spec-tokens "${BEST_K}" \
    --spec-profile
else
  echo
  echo "K sweep finished. To profile one K, run for example:"
  echo "  BEST_K=3 benchmarks/run_eagle3_sweep.sh"
fi
