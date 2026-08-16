# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure single-request Qwen3 time-to-first-token on Ascend.

Run this benchmark in fresh processes with
``VLLM_ASCEND_ENABLE_QKNORM_PREFILL_ATTENTION`` disabled and enabled. Prefix
caching is disabled so every sample executes a complete prefill.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def _prompt(length: int, salt: int) -> TokensPrompt:
    # Keep ids well inside the Qwen3 vocabulary and vary every request so an
    # accidental cache cannot turn a measured prefill into a cache hit.
    token_ids = [100 + ((index * 37 + salt * 101) % 30000) for index in range(length)]
    return TokensPrompt(prompt_token_ids=token_ids)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/tmp/models/Qwen3-0.6B")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 8, 17, 64, 128])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        gpu_memory_utilization=0.7,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    samples: dict[str, dict[str, float | list[float]]] = {}
    salt = 0
    for length in args.lengths:
        for _ in range(args.warmups):
            llm.generate([_prompt(length, salt)], sampling, use_tqdm=False)
            salt += 1
        torch.npu.synchronize()

        elapsed_ms = []
        for _ in range(args.repeats):
            prompt = _prompt(length, salt)
            salt += 1
            torch.npu.synchronize()
            start = time.perf_counter()
            llm.generate([prompt], sampling, use_tqdm=False)
            torch.npu.synchronize()
            elapsed_ms.append((time.perf_counter() - start) * 1000)
        samples[str(length)] = {
            "median_ms": statistics.median(elapsed_ms),
            "mean_ms": statistics.mean(elapsed_ms),
            "p95_ms": _percentile(elapsed_ms, 0.95),
            "samples_ms": elapsed_ms,
        }

    result = {
        "model": args.model,
        "optimization_enabled": os.getenv("VLLM_ASCEND_ENABLE_QKNORM_PREFILL_ATTENTION") == "1",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
