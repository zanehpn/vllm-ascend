# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare materialized QK-Norm/RoPE + FIA with the fused prototype."""

import argparse
import math
import time

import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.device.device_op import DeviceOperator

HEAD_DIM = 128
MODEL_HEADS = {
    "qwen3-0.6b": (16, 8),
    "qwen3-1.7b": (16, 8),
    "qwen3-8b": (32, 8),
}


def timed_ms(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_HEADS, default="qwen3-0.6b")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    num_query_heads, num_kv_heads = MODEL_HEADS[args.model]
    q_size = num_query_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(args.seq_len, q_size + 2 * kv_size, dtype=torch.bfloat16, device="npu")
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="npu")
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(args.seq_len, dtype=torch.int64, device="npu")
    angles = torch.randn(args.seq_len, HEAD_DIM // 2, dtype=torch.float32, device="npu")
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def baseline():
        q, k, v = DeviceOperator.split_qkv_rmsnorm_rope(
            input=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_size,
            kv_hidden_size=kv_size,
            head_dim=HEAD_DIM,
            eps=1e-6,
            q_bias=None,
            k_bias=None,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
        )
        return torch_npu.npu_fused_infer_attention_score(
            query=q.view(args.seq_len, num_query_heads, HEAD_DIM),
            key=k.view(args.seq_len, num_kv_heads, HEAD_DIM),
            value=v.view(args.seq_len, num_kv_heads, HEAD_DIM),
            input_layout="TND",
            actual_seq_lengths=[args.seq_len],
            actual_seq_lengths_kv=[args.seq_len],
            num_heads=num_query_heads,
            num_key_value_heads=num_kv_heads,
            sparse_mode=3,
            scale=scale,
        )[0]

    def fused():
        return torch.ops.vllm.qknorm_rope_prefill_attention(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            num_query_heads,
            num_kv_heads,
            1e-6,
            scale,
        )

    baseline_output = baseline()
    fused_output = fused()
    torch.npu.synchronize()
    torch.testing.assert_close(fused_output, baseline_output, rtol=2e-2, atol=2e-2)

    baseline_ms = timed_ms(baseline, args.warmup, args.repeats)
    fused_ms = timed_ms(fused, args.warmup, args.repeats)
    qk_bytes = args.seq_len * (q_size + kv_size) * qkv.element_size()
    print(f"model={args.model} seq_len={args.seq_len}")
    print(f"materialized baseline: {baseline_ms:.3f} ms")
    print(f"non-materializing fused: {fused_ms:.3f} ms")
    print(f"speedup: {baseline_ms / fused_ms:.3f}x")
    print(f"eliminated full-QK HBM traffic: {2 * qk_bytes / 1024**2:.2f} MiB (one write + one read)")


if __name__ == "__main__":
    main()
