# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental non-materializing QK-Norm/RoPE prefill attention.

This is the executable prototype for the AscendC kernel.  It deliberately has
a narrow contract: one causal sequence, BF16, head dimension 128, and no KV
cache.  Q and K are normalized and rotated inside the attention program and
are never returned or stored in global memory.
"""

import math

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

HEAD_DIM: tl.constexpr = 128
HALF_HEAD_DIM: tl.constexpr = 64


@triton.jit
def qknorm_rope_kv_kernel(
    qkv_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    key_ptr,
    value_ptr,
    seq_len: tl.constexpr,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    eps: tl.constexpr,
):
    program_id = tl.program_id(0)
    token = program_id // num_kv_heads
    kv_head = program_id % num_kv_heads
    dims = tl.arange(0, HEAD_DIM)
    half_dims = dims % HALF_HEAD_DIM
    valid = token < seq_len
    k_base = token * total_hidden_size + q_hidden_size + kv_head * HEAD_DIM

    k_raw = tl.load(qkv_ptr + k_base + dims, mask=valid, other=0.0).to(tl.float32)
    k_inv_rms = 1.0 / tl.sqrt(tl.sum(k_raw * k_raw, axis=0) / HEAD_DIM + eps)
    k_first = (
        tl.load(qkv_ptr + k_base + half_dims, mask=valid, other=0.0).to(tl.float32)
        * k_inv_rms
        * tl.load(k_weight_ptr + half_dims).to(tl.float32)
    ).to(tl.bfloat16)
    k_second = (
        tl.load(qkv_ptr + k_base + half_dims + HALF_HEAD_DIM, mask=valid, other=0.0).to(tl.float32)
        * k_inv_rms
        * tl.load(k_weight_ptr + half_dims + HALF_HEAD_DIM).to(tl.float32)
    ).to(tl.bfloat16)
    position = tl.load(positions_ptr + token, mask=valid, other=0)
    rope_base = position * HEAD_DIM + half_dims
    cosine = tl.load(cos_sin_cache_ptr + rope_base, mask=valid, other=0.0).to(tl.float32)
    sine = tl.load(cos_sin_cache_ptr + rope_base + HALF_HEAD_DIM, mask=valid, other=0.0).to(tl.float32)
    key = tl.where(
        dims < HALF_HEAD_DIM,
        k_first * cosine - k_second * sine,
        k_second * cosine + k_first * sine,
    ).to(tl.bfloat16)

    output_base = (token * num_kv_heads + kv_head) * HEAD_DIM
    value_base = token * total_hidden_size + q_hidden_size + kv_hidden_size + kv_head * HEAD_DIM
    value = tl.load(qkv_ptr + value_base + dims, mask=valid, other=0.0)
    tl.store(key_ptr + output_base + dims, key, mask=valid)
    tl.store(value_ptr + output_base + dims, value, mask=valid)


@triton.jit
def qknorm_rope_prefill_attention_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    seq_len: tl.constexpr,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    eps: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    program_id = tl.program_id(0)
    query_block = program_id // num_query_heads
    query_head = program_id % num_query_heads
    kv_group_size: tl.constexpr = num_query_heads // num_kv_heads
    kv_head = query_head // kv_group_size

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    half_dims = dims % HALF_HEAD_DIM
    row_mask = rows < seq_len

    q_base = rows[:, None] * total_hidden_size + query_head * HEAD_DIM
    q_raw = tl.load(qkv_ptr + q_base + dims[None, :], mask=row_mask[:, None], other=0.0)
    q_fp32 = q_raw.to(tl.float32)
    q_inv_rms = 1.0 / tl.sqrt(tl.sum(q_fp32 * q_fp32, axis=1) / HEAD_DIM + eps)

    q_first = tl.load(
        qkv_ptr + q_base + half_dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    q_second = tl.load(
        qkv_ptr + q_base + half_dims[None, :] + HALF_HEAD_DIM,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    # Match the established fused QK-Norm/RoPE numerical boundary: normalize
    # and apply gamma in FP32, round to BF16, then rotate.  The values remain
    # on chip; this does not materialize normalized Q/K in global memory.
    q_first = (
        q_first
        * q_inv_rms[:, None]
        * tl.load(q_weight_ptr + half_dims)[None, :].to(tl.float32)
    ).to(tl.bfloat16)
    q_second = (
        q_second
        * q_inv_rms[:, None]
        * tl.load(q_weight_ptr + half_dims + HALF_HEAD_DIM)[None, :].to(tl.float32)
    ).to(tl.bfloat16)

    q_positions = tl.load(positions_ptr + rows, mask=row_mask, other=0)
    q_cache_base = q_positions[:, None] * HEAD_DIM + half_dims[None, :]
    q_cos = tl.load(cos_sin_cache_ptr + q_cache_base, mask=row_mask[:, None], other=0.0).to(tl.float32)
    q_sin = tl.load(
        cos_sin_cache_ptr + q_cache_base + HALF_HEAD_DIM,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    q_rotated = tl.where(
        dims[None, :] < HALF_HEAD_DIM,
        q_first * q_cos - q_second * q_sin,
        q_second * q_cos + q_first * q_sin,
    )
    q = q_rotated.to(tl.bfloat16)

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for key_start in tl.range(0, seq_len, BLOCK_N):
        cols = key_start + tl.arange(0, BLOCK_N)
        col_mask = cols < seq_len
        k_base = cols[:, None] * total_hidden_size + q_hidden_size + kv_head * HEAD_DIM

        k_raw = tl.load(qkv_ptr + k_base + dims[None, :], mask=col_mask[:, None], other=0.0)
        k_fp32 = k_raw.to(tl.float32)
        k_inv_rms = 1.0 / tl.sqrt(tl.sum(k_fp32 * k_fp32, axis=1) / HEAD_DIM + eps)

        k_first = tl.load(
            qkv_ptr + k_base + half_dims[None, :],
            mask=col_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        k_second = tl.load(
            qkv_ptr + k_base + half_dims[None, :] + HALF_HEAD_DIM,
            mask=col_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        k_first = (
            k_first
            * k_inv_rms[:, None]
            * tl.load(k_weight_ptr + half_dims)[None, :].to(tl.float32)
        ).to(tl.bfloat16)
        k_second = (
            k_second
            * k_inv_rms[:, None]
            * tl.load(k_weight_ptr + half_dims + HALF_HEAD_DIM)[None, :].to(tl.float32)
        ).to(tl.bfloat16)

        k_positions = tl.load(positions_ptr + cols, mask=col_mask, other=0)
        k_cache_base = k_positions[:, None] * HEAD_DIM + half_dims[None, :]
        k_cos = tl.load(cos_sin_cache_ptr + k_cache_base, mask=col_mask[:, None], other=0.0).to(tl.float32)
        k_sin = tl.load(
            cos_sin_cache_ptr + k_cache_base + HALF_HEAD_DIM,
            mask=col_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        k_rotated = tl.where(
            dims[None, :] < HALF_HEAD_DIM,
            k_first * k_cos - k_second * k_sin,
            k_second * k_cos + k_first * k_sin,
        )
        k = k_rotated.to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * softmax_scale
        causal_mask = col_mask[None, :] & row_mask[:, None] & (cols[None, :] <= rows[:, None])
        # A partial query tile contains padding rows.  Leaving every score in
        # such a row at -inf makes online softmax evaluate -inf - (-inf), and
        # NaNs can contaminate valid rows inside the Cube tile.  Give padding
        # rows one harmless sentinel key; their outputs are never stored.
        causal_mask |= (~row_mask[:, None]) & (cols[None, :] == 0)
        scores = tl.where(causal_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        block_sum = tl.sum(probabilities, axis=1)

        v_base = (
            cols[:, None] * total_hidden_size
            + q_hidden_size
            + kv_hidden_size
            + kv_head * HEAD_DIM
        )
        value = tl.load(
            qkv_ptr + v_base + dims[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        if STORE_CACHE:
            # The cache layout is [num_blocks, block_size, KV heads, D].
            # A flattened slot therefore addresses one [KV heads, D] row.
            # Only the first query block and one query head per GQA group
            # publish this KV head, so every cache element has one writer.
            slots = tl.load(slot_mapping_ptr + cols, mask=col_mask, other=-1)
            cache_mask = (
                col_mask[:, None]
                & (slots[:, None] >= 0)
                & (query_block == 0)
                & ((query_head % kv_group_size) == 0)
            )
            cache_offset = (
                slots[:, None] * num_kv_heads * HEAD_DIM
                + kv_head * HEAD_DIM
                + dims[None, :]
            )
            tl.store(key_cache_ptr + cache_offset, k, mask=cache_mask)
            tl.store(value_cache_ptr + cache_offset, value, mask=cache_mask)
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), value, input_precision="ieee"
        )
        running_sum = running_sum * correction + block_sum
        running_max = new_max

    output = accumulator / running_sum[:, None]
    output_offset = rows[:, None] * num_query_heads * HEAD_DIM + query_head * HEAD_DIM + dims[None, :]
    tl.store(output_ptr + output_offset, output, mask=row_mask[:, None])


def qknorm_rope_prefill_attention_impl(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float,
    scale: float,
) -> torch.Tensor:
    if qkv.dtype != torch.bfloat16:
        raise ValueError(f"only bfloat16 is supported, got {qkv.dtype}")
    if qkv.ndim != 2:
        raise ValueError(f"qkv must have shape [T, H], got {tuple(qkv.shape)}")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    q_hidden_size = num_query_heads * HEAD_DIM
    kv_hidden_size = num_kv_heads * HEAD_DIM
    expected_hidden_size = q_hidden_size + 2 * kv_hidden_size
    if qkv.shape[1] != expected_hidden_size:
        raise ValueError(f"qkv hidden size must be {expected_hidden_size}, got {qkv.shape[1]}")
    if q_weight.numel() != HEAD_DIM or k_weight.numel() != HEAD_DIM:
        raise ValueError("Q/K RMSNorm weights must each contain 128 elements")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != HEAD_DIM:
        raise ValueError("cos_sin_cache must have shape [max_position, 128]")
    if positions.ndim != 1 or positions.shape[0] != qkv.shape[0]:
        raise ValueError("positions must have shape [T]")

    seq_len = qkv.shape[0]
    output = torch.empty(
        (seq_len, num_query_heads, HEAD_DIM),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    block_m = 16
    block_n = 32
    grid = (triton.cdiv(seq_len, block_m) * num_query_heads, 1, 1)
    qknorm_rope_prefill_attention_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        output,
        qkv,
        qkv,
        positions,
        seq_len,
        q_hidden_size,
        kv_hidden_size,
        expected_hidden_size,
        num_query_heads,
        num_kv_heads,
        eps,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        STORE_CACHE=False,
        num_warps=4,
        num_stages=2,
    )
    return output


def qknorm_rope_prefill_attention_with_cache_impl(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float,
    scale: float,
) -> torch.Tensor:
    q_hidden_size = num_query_heads * HEAD_DIM
    kv_hidden_size = num_kv_heads * HEAD_DIM
    expected_hidden_size = q_hidden_size + 2 * kv_hidden_size
    if qkv.dtype != torch.bfloat16 or key_cache.dtype != qkv.dtype or value_cache.dtype != qkv.dtype:
        raise ValueError("QKV and KV cache must be bfloat16")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] < qkv.shape[0]:
        raise ValueError("slot_mapping must contain one slot per token")
    if key_cache.numel() != value_cache.numel():
        raise ValueError("key and value caches must have equal size")

    seq_len = qkv.shape[0]
    output = torch.empty(
        (seq_len, num_query_heads, HEAD_DIM), dtype=qkv.dtype, device=qkv.device
    )
    block_m = 16
    block_n = 32
    grid = (triton.cdiv(seq_len, block_m) * num_query_heads, 1, 1)
    qknorm_rope_prefill_attention_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        output,
        key_cache,
        value_cache,
        slot_mapping,
        seq_len,
        q_hidden_size,
        kv_hidden_size,
        expected_hidden_size,
        num_query_heads,
        num_kv_heads,
        eps,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        STORE_CACHE=True,
        num_warps=4,
        num_stages=2,
    )
    return output


def qknorm_rope_prefill_attention_with_cache_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float,
    scale: float,
) -> torch.Tensor:
    del q_weight, k_weight, cos_sin_cache, positions, key_cache, value_cache
    del slot_mapping, num_kv_heads, eps, scale
    return torch.empty(
        (qkv.shape[0], num_query_heads, HEAD_DIM), dtype=qkv.dtype, device=qkv.device
    )


def qknorm_rope_kv_impl(
    qkv: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_hidden_size = num_query_heads * HEAD_DIM
    kv_hidden_size = num_kv_heads * HEAD_DIM
    total_hidden_size = q_hidden_size + 2 * kv_hidden_size
    if qkv.dtype != torch.bfloat16 or qkv.ndim != 2:
        raise ValueError("qkv must be a 2D bfloat16 tensor")
    if qkv.shape[1] != total_hidden_size:
        raise ValueError(f"qkv hidden size must be {total_hidden_size}")
    key = torch.empty((qkv.shape[0], num_kv_heads, HEAD_DIM), dtype=qkv.dtype, device=qkv.device)
    value = torch.empty_like(key)
    grid = (qkv.shape[0] * num_kv_heads, 1, 1)
    qknorm_rope_kv_kernel[grid](
        qkv,
        k_weight,
        cos_sin_cache,
        positions,
        key,
        value,
        qkv.shape[0],
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        num_kv_heads,
        eps,
        num_warps=4,
        num_stages=1,
    )
    return key, value


def qknorm_rope_kv_fake(
    qkv: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k_weight, cos_sin_cache, positions, num_query_heads, eps
    shape = (qkv.shape[0], num_kv_heads, HEAD_DIM)
    return torch.empty(shape, dtype=qkv.dtype, device=qkv.device), torch.empty(
        shape, dtype=qkv.dtype, device=qkv.device
    )


def qknorm_rope_prefill_attention_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float,
    scale: float,
) -> torch.Tensor:
    del q_weight, k_weight, cos_sin_cache, positions, num_kv_heads, eps, scale
    return torch.empty(
        (qkv.shape[0], num_query_heads, HEAD_DIM),
        dtype=qkv.dtype,
        device=qkv.device,
    )


direct_register_custom_op(
    op_name="qknorm_rope_prefill_attention",
    op_func=qknorm_rope_prefill_attention_impl,
    fake_impl=qknorm_rope_prefill_attention_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="qknorm_rope_kv",
    op_func=qknorm_rope_kv_impl,
    fake_impl=qknorm_rope_kv_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="qknorm_rope_prefill_attention_with_cache",
    op_func=qknorm_rope_prefill_attention_with_cache_impl,
    fake_impl=qknorm_rope_prefill_attention_with_cache_fake,
    mutates_args=["key_cache", "value_cache"],
    dispatch_key="PrivateUse1",
)


def default_scale() -> float:
    return 1.0 / math.sqrt(HEAD_DIM)
