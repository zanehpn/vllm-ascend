# QK-Norm Prefill K-Once Results

Date: 2026-08-16

Hardware and software:

- Ascend 910B3
- CANN 9.0.1
- BF16, Qwen3-0.6B head shape (16 Q heads, 8 KV heads, head dim 128)
- Single causal sequence
- Cache block size 128

The K-once path runs a K/V-only Triton kernel first. It normalizes and rotates
each K exactly once and writes K/V directly into the official paged-cache
layout. The attention kernel then reads K/V from those cache slots instead of
recomputing K for every query tile.

## Prefill latency

| Tokens | Official materialized + FIA | Original fused (K recompute) | K-once cached fused | K-once speedup | K/V-only cache write |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.495 ms | 23.672 ms | 10.003 ms | 2.367x | 0.852 ms |
| 512 | 0.775 ms | 228.880 ms | 35.322 ms | 6.480x | 3.147 ms |
| 2,048 | 0.602 ms | 3,216.099 ms | 150.158 ms | 21.418x | 12.410 ms |
| 4,096 | 1.165 ms | 12,539.857 ms | 333.051 ms | 37.651x | 24.815 ms |

The 128/512 measurements use 3/5 timed iterations respectively, 2K uses two,
and 4K uses one because the original recomputing prototype takes 12.5 seconds
per call. Every timed region synchronizes the NPU before and after measurement.

Removing K recomputation is a large improvement to the prototype, and its
benefit grows with sequence length. It does not make the custom attention loop
competitive with the official FIA kernel. The useful integration direction is
therefore the K/V-only cache publisher combined with official FIA, not replacing
FIA with this Triton attention loop.

## Cache traffic and chunking

| Tokens | K/V-only + official write | Direct cache write | Direct speedup | Chunk size | Chunked direct write |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.811 ms | 0.795 ms | 1.020x | 64 | 0.810 ms |
| 512 | 3.110 ms | 3.106 ms | 1.001x | 128 | 3.157 ms |
| 2,048 | 12.229 ms | 12.263 ms | 0.997x | 512 | 12.316 ms |
| 4,096 | 24.584 ms | 24.650 ms | 0.997x | 512 | 24.778 ms |

Payload bandwidth counts the K and V bytes published to cache and is about
0.66-0.68 GB/s. Direct publication is effectively tied with producing
contiguous K/V and calling the official cache writer: it gains 2% only at 128
tokens and is 0.3% slower at 2K/4K. Chunking adds 0.5-1.9% latency in the
repeat run. The modeled raw-K reread traffic removed by the K-once path is
0.003/0.061/0.992/3.984 GiB at 128/512/2K/4K respectively.

## Correctness

The focused NPU suite passed 10 tests, including:

- fused attention versus the materialized reference;
- paged K/V cache contents with non-identity slot mapping;
- K/V-only kernel versus the normalization/RoPE reference;
- cache publication in 8- and 17-token chunks followed by a decode token,
  compared with a full unchunked causal-attention reference.

Command:

```bash
env -u ASCEND_RT_VISIBLE_DEVICES pytest -q \
  --confcutdir=tests/e2e/nightly/single_node/ops \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qknorm_rope_prefill_attention.py
```

Result: `10 passed` in 83.94 seconds. The repository-wide e2e conftest could
not load in this checkout because the installed upstream vLLM lacks
`FusedMoEFactory`; the focused ops conftest was used to avoid that unrelated
version mismatch.
