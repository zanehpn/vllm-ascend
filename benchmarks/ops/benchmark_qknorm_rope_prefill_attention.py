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
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

HEAD_DIM = 128
CACHE_BLOCK_SIZE = 128
FIA_CAUSAL_MASK_SIZE = 2048
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
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Also time cache-only writes split into chunks of this size.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Skip attention and compare only K/V cache publication paths.",
    )
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    init_device_properties_triton()

    num_query_heads, num_kv_heads = MODEL_HEADS[args.model]
    q_size = num_query_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(
        args.seq_len,
        q_size + 2 * kv_size,
        dtype=torch.bfloat16,
        device=args.device,
    )
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=args.device)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=args.device)
    positions = torch.arange(args.seq_len, dtype=torch.int64, device=args.device)
    angles = torch.randn(args.seq_len, HEAD_DIM // 2, dtype=torch.float32, device=args.device)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    num_cache_blocks = math.ceil(args.seq_len / CACHE_BLOCK_SIZE)
    cache_shape = (
        num_cache_blocks,
        CACHE_BLOCK_SIZE,
        num_kv_heads,
        HEAD_DIM,
    )
    key_cache = torch.empty(cache_shape, dtype=qkv.dtype, device=args.device)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(
        args.seq_len, dtype=torch.int64, device=args.device
    )
    causal_mask = torch.ones(
        FIA_CAUSAL_MASK_SIZE,
        FIA_CAUSAL_MASK_SIZE,
        dtype=torch.bool,
        device=args.device,
    ).triu_(diagonal=1)

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
            atten_mask=causal_mask,
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

    def fused_cached():
        return torch.ops.vllm.qknorm_rope_prefill_attention_with_cache(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            key_cache,
            value_cache,
            slot_mapping,
            num_query_heads,
            num_kv_heads,
            1e-6,
            scale,
        )

    def kv_cache_only():
        torch.ops.vllm.qknorm_rope_kv_cache(
            qkv,
            k_weight,
            cos_sin_cache,
            positions,
            key_cache,
            value_cache,
            slot_mapping,
            num_query_heads,
            num_kv_heads,
            1e-6,
        )

    def kv_then_official_cache():
        key, value = torch.ops.vllm.qknorm_rope_kv(
            qkv,
            k_weight,
            cos_sin_cache,
            positions,
            num_query_heads,
            num_kv_heads,
            1e-6,
        )
        DeviceOperator.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        )

    def chunked_kv_cache_only():
        for start in range(0, args.seq_len, args.chunk_size):
            end = min(start + args.chunk_size, args.seq_len)
            torch.ops.vllm.qknorm_rope_kv_cache(
                qkv[start:end],
                k_weight,
                cos_sin_cache,
                positions[start:end],
                key_cache,
                value_cache,
                slot_mapping[start:end],
                num_query_heads,
                num_kv_heads,
                1e-6,
            )

    if args.cache_only:
        kv_then_official_cache_ms = timed_ms(
            kv_then_official_cache, args.warmup, args.repeats
        )
        direct_cache_ms = timed_ms(
            kv_cache_only, args.warmup, args.repeats
        )
        payload_bytes = 2 * args.seq_len * kv_size * qkv.element_size()
        print(f"model={args.model} seq_len={args.seq_len}")
        print(
            "K/V-only + official cache write: "
            f"{kv_then_official_cache_ms:.3f} ms"
        )
        print(f"K/V-only direct paged-cache write: {direct_cache_ms:.3f} ms")
        print(
            "direct-vs-official speedup: "
            f"{kv_then_official_cache_ms / direct_cache_ms:.3f}x"
        )
        print(
            "direct cache payload bandwidth: "
            f"{payload_bytes / (direct_cache_ms / 1000) / 1e9:.2f} GB/s"
        )
        if args.chunk_size:
            chunked_ms = timed_ms(
                chunked_kv_cache_only, args.warmup, args.repeats
            )
            print(f"chunk_size={args.chunk_size}")
            print(f"chunked K/V-only cache write: {chunked_ms:.3f} ms")
        return

    baseline_output = baseline()
    fused_output = fused()
    fused_cached_output = fused_cached()
    torch.npu.synchronize()
    torch.testing.assert_close(fused_output, baseline_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        fused_cached_output, baseline_output, rtol=2e-2, atol=2e-2
    )

    _, expected_key, expected_value = DeviceOperator.split_qkv_rmsnorm_rope(
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
    torch.testing.assert_close(
        key_cache.view(-1, num_kv_heads, HEAD_DIM)[: args.seq_len],
        expected_key.view(args.seq_len, num_kv_heads, HEAD_DIM),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        value_cache.view(-1, num_kv_heads, HEAD_DIM)[: args.seq_len],
        expected_value.view(args.seq_len, num_kv_heads, HEAD_DIM),
        rtol=0,
        atol=0,
    )

    baseline_ms = timed_ms(baseline, args.warmup, args.repeats)
    fused_ms = timed_ms(fused, args.warmup, args.repeats)
    fused_cached_ms = timed_ms(fused_cached, args.warmup, args.repeats)
    kv_cache_ms = timed_ms(kv_cache_only, args.warmup, args.repeats)
    qk_bytes = args.seq_len * (q_size + kv_size) * qkv.element_size()
    block_m = 8 if num_query_heads // num_kv_heads == 4 else 16
    query_blocks = math.ceil(args.seq_len / block_m)
    # The old fused kernel rereads raw K twice (RMS reduction plus gamma/RoPE)
    # for every query block. The cached path reads it twice only once.
    avoided_k_read_bytes = (
        2
        * (query_blocks - 1)
        * args.seq_len
        * kv_size
        * qkv.element_size()
    )
    cache_payload_bytes = 2 * args.seq_len * kv_size * qkv.element_size()
    print(f"model={args.model} seq_len={args.seq_len}")
    print(f"materialized baseline: {baseline_ms:.3f} ms")
    print(f"non-materializing fused: {fused_ms:.3f} ms")
    print(f"K-once paged-cache fused: {fused_cached_ms:.3f} ms")
    print(f"K/V-only direct paged-cache write: {kv_cache_ms:.3f} ms")
    print(f"speedup: {baseline_ms / fused_ms:.3f}x")
    print(f"K-once speedup vs recompute: {fused_ms / fused_cached_ms:.3f}x")
    print(
        "modeled K reread traffic eliminated: "
        f"{avoided_k_read_bytes / 1024**3:.3f} GiB"
    )
    print(
        "modeled avoided-K bandwidth: "
        f"{avoided_k_read_bytes / (fused_cached_ms / 1000) / 1e9:.2f} GB/s"
    )
    print(
        "direct cache payload bandwidth: "
        f"{cache_payload_bytes / (kv_cache_ms / 1000) / 1e9:.2f} GB/s"
    )
    if args.chunk_size:
        chunked_ms = timed_ms(
            chunked_kv_cache_only, args.warmup, args.repeats
        )
        print(f"chunk_size={args.chunk_size}")
        print(f"chunked K/V-only cache write: {chunked_ms:.3f} ms")
        print(
            "chunked cache payload bandwidth: "
            f"{cache_payload_bytes / (chunked_ms / 1000) / 1e9:.2f} GB/s"
        )
    print(f"eliminated full-QK HBM traffic: {2 * qk_bytes / 1024**2:.2f} MiB (one write + one read)")


if __name__ == "__main__":
    main()
