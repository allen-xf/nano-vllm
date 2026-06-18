# Chunked Prefill Benchmark Summary

## Setup
- Model: Qwen3-0.6B
- Short requests: 64, prompt 50~200 tok, output 256 tok
- Long requests: 64, prompt 3500 tok, output 32 tok

## Results

| Metric | Full Prefill (baseline) | Chunked Prefill |
|--------|------------------------|-----------------|
| Total time | 11.19s | 9.79s |
| Throughput | 1646.5 tok/s | 1883.0 tok/s |
| Total steps | 384 | 321 |
| Pure prefill steps | 66 | 2 |
| Pure decode steps | 318 | 283 |
| Mixed steps | 0 | 36 |
| Avg utilization | 9.9% | 11.9% |

## Conclusion

Chunked Prefill is **12.6% faster** than Full Prefill, with higher GPU utilization (11.9% vs 9.9%) by interleaving prefill chunks with decode steps instead of blocking decode during long prefills.
