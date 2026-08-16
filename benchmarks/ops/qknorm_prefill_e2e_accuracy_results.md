# QK-Norm prefill end-to-end accuracy

## Scope

This validation starts a real vLLM V1 engine and compares the stock Ascend
attention path with `VLLM_ASCEND_ENABLE_QKNORM_PREFILL_ATTENTION=1`. Both runs
use BF16 eager execution, a 256-token scheduler budget, chunked prefill, prefix
caching, an eight-request scheduler batch, and eight greedy decode tokens per
request. The comparison records:

- generated token IDs and top-20 log probabilities for every decode step;
- every raw logits tensor returned by the model;
- the written paged key and value cache blocks in every decoder layer; and
- prompt lengths and the scheduler-reported prefix-cache hit count.

The reusable runner is
`benchmarks/scripts/validate_qknorm_prefill_e2e.py`.

## Mismatch found and fixed

Before the dispatch guard was added, a 149-token Qwen3 prefix-seed request used
the experimental non-materializing attention kernel. Its fifth generated token
diverged from the stock path, with a 12.28125 maximum logits error and a
4.934 maximum top-logprob error. Direct operator tests had only established the
kernel's accuracy through 128 tokens.

The runtime now limits this dispatch to complete single-sequence prefills of at
most 128 tokens. Longer and chunked requests use the established paged FIA path.
This preserves the optimized path for its validated range while removing the
observed end-to-end token mismatch. A 128/129-token dispatch boundary regression
test covers the guard.

## Qwen3-0.6B result

The Qwen3 run used `/tmp/models/Qwen3-0.6B` and exercised the following real
scheduler inputs.

| Scenario | Prompt tokens | Prefix-cache hit | Requests | Decode tokens/request |
| --- | ---: | ---: | ---: | ---: |
| Chunked prefill | 407 | 0 | 1 | 8 |
| Mixed batch | 1, 5, 7, 129 | 0, 0, 0, 0 | 4 | 8 |
| Prefix seed | 149 | 0 | 1 | 8 |
| Prefix hit | 149 | 128 | 1 | 8 |
| Multi-request scheduling | 13 each | 0 each | 8 | 8 |
| Optimized full prefill | 11 | 0 | 1 | 8 |

The baseline and optimized runs produced exactly the same 128 generated token
IDs across all 16 requests. All 49 captured logits calls and all written cache
blocks in all 28 layers passed the BF16 tolerances.

| Metric | Result |
| --- | ---: |
| Generated-token mismatches | 0 |
| Maximum logits absolute error | 0.21875 |
| Maximum logits mean absolute error | 0.0419323221 |
| Maximum common top-logprob absolute error | 0.1733694077 |
| Top-20 boundary-set differences | 1 request (`full_prefill[0]`) |
| Maximum KV-cache absolute error | 0.5 |
| Maximum KV-cache mean absolute error | 0.0004477098 |

The one top-20 set difference is at the rank-20 boundary; it did not change any
selected token. The full vocabulary logits comparison is authoritative and is
included independently of top-20 reporting.

### Qwen3 per-layer paged-cache error

`K max/mean` and `V max/mean` below compare all written BF16 cache blocks after
the full prompt-to-decode scenario sequence.

| Layer | K max | K mean | V max | V mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 |
| 9 | 0 | 0 | 0 | 0 |
| 10 | 0 | 0 | 0 | 0 |
| 11 | 0 | 0 | 0 | 0 |
| 12 | 0 | 0 | 0 | 0 |
| 13 | 0.03125 | 1.1125992e-06 | 0.015625 | 8.7566707e-07 |
| 14 | 0.125 | 1.4704159e-05 | 0.03125 | 1.1786309e-05 |
| 15 | 0.25 | 3.0853636e-05 | 0.0625 | 3.0528990e-05 |
| 16 | 0.125 | 4.0872412e-05 | 0.05078125 | 4.1916053e-05 |
| 17 | 0.25 | 3.9423558e-05 | 0.125 | 7.7579796e-05 |
| 18 | 0.125 | 4.7541955e-05 | 0.125 | 8.3626728e-05 |
| 19 | 0.125 | 4.4649016e-05 | 0.125 | 0.0001114259 |
| 20 | 0.125 | 5.1084378e-05 | 0.125 | 0.0001296983 |
| 21 | 0.25 | 5.1085372e-05 | 0.25 | 0.0001893036 |
| 22 | 0.125 | 4.6960409e-05 | 0.21875 | 0.0002114222 |
| 23 | 0.3125 | 6.6162625e-05 | 0.25 | 0.0002302972 |
| 24 | 0.28515625 | 7.1771938e-05 | 0.2578125 | 0.0003019762 |
| 25 | 0.25 | 7.0351693e-05 | 0.5 | 0.0003946748 |
| 26 | 0.125 | 6.0583741e-05 | 0.5 | 0.0004465688 |
| 27 | 0.25 | 5.5044100e-05 | 0.5 | 0.0004477098 |

## Qwen3-VL-2B-Instruct result

The multimodal run used the complete `Qwen/Qwen3-VL-2B-Instruct` checkpoint
(LFS SHA-256
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`).
It ran the same text scheduler matrix plus a real processor-generated prompt
containing a synthetic 224 by 224 RGB image. `ignore_eos=True` forced all 17
requests through eight decode steps.

| Scenario | Prompt tokens | Prefix-cache hit | Requests | Decode tokens/request |
| --- | ---: | ---: | ---: | ---: |
| Chunked prefill | 407 | 0 | 1 | 8 |
| Mixed batch | 1, 5, 7, 129 | 0, 0, 0, 0 | 4 | 8 |
| Prefix seed | 149 | 0 | 1 | 8 |
| Prefix hit | 149 | 128 | 1 | 8 |
| Multi-request scheduling | 13 each | 0 each | 8 | 8 |
| Full prefill | 11 | 0 | 1 | 8 |
| Multimodal image | 82 | 0 | 1 | 8 |

The two runs were bitwise identical: zero generated-token or top-logprob-set
mismatches, zero error in all 57 logits calls, and zero K or V cache error in
all 28 language-model layers. This is 136 exactly matching generated tokens.

Qwen3-VL uses multimodal rotary embeddings, so the current experimental Qwen3
non-materializing dispatch remains ineligible and safely follows the stock
attention path. This validation therefore establishes that enabling the feature
does not alter Qwen3-VL prompt, image, chunked-prefill, prefix-cache, scheduling,
or decode behavior.

### Qwen3-VL startup compatibility fix

CANN 9.0.1 lacks `aclnnAddRmsNormBias`, requiring the supported
`VLLM_BATCH_INVARIANT=1` fallback. The fallback's linear implementation
previously asserted that every input was 2D, while the Qwen3-VL vision encoder
passes 3D `[batch, sequence, hidden]` tensors. The implementation now flattens
all leading dimensions for the persistent matrix multiplication and restores
them afterward. An NPU regression test compares a 3D biased linear operation
with the stock PyTorch result.

## Reproduction

The verified upstream vLLM source is commit
`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`. Preserve the existing CANN
`PYTHONPATH` entries when prepending both source trees. CANN 9.0.1 in this
environment also requires the supported batch-invariant RMSNorm fallback.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/tmp/vllm-ascend-qknorm-attention-prefill:/tmp/vllm-qknorm-attention-prefill:$PYTHONPATH
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=1

VLLM_ASCEND_ENABLE_QKNORM_PREFILL_ATTENTION=0 \
  env -u ASCEND_RT_VISIBLE_DEVICES python \
  benchmarks/scripts/validate_qknorm_prefill_e2e.py \
  --model /tmp/models/Qwen3-0.6B --output-prefix /tmp/qknorm_e2e/baseline

VLLM_ASCEND_ENABLE_QKNORM_PREFILL_ATTENTION=1 \
  env -u ASCEND_RT_VISIBLE_DEVICES python \
  benchmarks/scripts/validate_qknorm_prefill_e2e.py \
  --model /tmp/models/Qwen3-0.6B --output-prefix /tmp/qknorm_e2e/optimized

python benchmarks/scripts/validate_qknorm_prefill_e2e.py \
  --compare /tmp/qknorm_e2e/baseline /tmp/qknorm_e2e/optimized \
  --report /tmp/qknorm_e2e/qwen3_report.json

VLLM_ASCEND_ENABLE_QKNORM_PREFILL_ATTENTION=0 \
  env -u ASCEND_RT_VISIBLE_DEVICES python \
  benchmarks/scripts/validate_qknorm_prefill_e2e.py \
  --model /tmp/models/Qwen3-VL-2B-Instruct --multimodal \
  --output-prefix /tmp/qknorm_e2e/qwen3vl_baseline

VLLM_ASCEND_ENABLE_QKNORM_PREFILL_ATTENTION=1 \
  env -u ASCEND_RT_VISIBLE_DEVICES python \
  benchmarks/scripts/validate_qknorm_prefill_e2e.py \
  --model /tmp/models/Qwen3-VL-2B-Instruct --multimodal \
  --output-prefix /tmp/qknorm_e2e/qwen3vl_optimized

python benchmarks/scripts/validate_qknorm_prefill_e2e.py \
  --compare /tmp/qknorm_e2e/qwen3vl_baseline \
  /tmp/qknorm_e2e/qwen3vl_optimized \
  --report /tmp/qknorm_e2e/qwen3vl_report.json
```
