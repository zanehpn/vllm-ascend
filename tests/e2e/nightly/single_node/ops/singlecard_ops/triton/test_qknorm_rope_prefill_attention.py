# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.qwen3_qknorm_prefill_attention import (
    SUPPORTED_MAX_SEQ_LEN,
    _can_use_non_materializing_prefill,
)

HEAD_DIM = 128


@pytest.mark.parametrize(
    "seq_len,expected",
    [(SUPPORTED_MAX_SEQ_LEN, True), (SUPPORTED_MAX_SEQ_LEN + 1, False)],
)
def test_prefill_dispatch_rejects_unvalidated_long_sequences(seq_len: int, expected: bool):
    device = torch.device("npu")
    qkv = torch.empty(seq_len, 4096, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    cos_sin_cache = torch.empty(256, HEAD_DIM, dtype=torch.bfloat16, device=device)
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[seq_len],
        attn_state=AscendAttentionState.PrefillNoCache,
        causal=True,
    )
    assert (
        _can_use_non_materializing_prefill(
            qkv,
            positions,
            cos_sin_cache,
            metadata,
            16,
            8,
            HEAD_DIM,
            256,
        )
        is expected
    )


def reference_attention(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float,
) -> torch.Tensor:
    seq_len = qkv.shape[0]
    q_size = num_query_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(seq_len, num_query_heads, HEAD_DIM).float()
    k = k.view(seq_len, num_kv_heads, HEAD_DIM).float()
    v = v.view(seq_len, num_kv_heads, HEAD_DIM).float()
    q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + eps) * q_weight.float()
    k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + eps) * k_weight.float()

    cos, sin = cos_sin_cache[positions].float().chunk(2, dim=-1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        cos_by_head = cos[:, None, :]
        sin_by_head = sin[:, None, :]
        return torch.cat(
            (first * cos_by_head - second * sin_by_head, second * cos_by_head + first * sin_by_head),
            dim=-1,
        )

    q = rotate(q)
    k = rotate(k).repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(HEAD_DIM)
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=qkv.device).tril()
    probabilities = torch.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
    return torch.einsum("hts,shd->thd", probabilities, v).to(qkv.dtype)


@pytest.mark.parametrize("seq_len", [8, 17, 128])
@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(16, 8), (32, 8)])
def test_qknorm_rope_prefill_attention(seq_len: int, num_query_heads: int, num_kv_heads: int):
    torch.manual_seed(7)
    device = torch.device("npu")
    q_size = num_query_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(seq_len, q_size + 2 * kv_size, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    angles = torch.randn(seq_len, HEAD_DIM // 2, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    eps = 1e-6

    actual = torch.ops.vllm.qknorm_rope_prefill_attention(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_query_heads,
        num_kv_heads,
        eps,
        1.0 / math.sqrt(HEAD_DIM),
    )
    expected = reference_attention(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_query_heads,
        num_kv_heads,
        eps,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qknorm_rope_prefill_attention_writes_paged_kv_cache():
    torch.manual_seed(11)
    device = torch.device("npu")
    seq_len = 8
    num_query_heads = 16
    num_kv_heads = 8
    q_size = num_query_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(seq_len, q_size + 2 * kv_size, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    angles = torch.randn(seq_len, HEAD_DIM // 2, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    slot_mapping = torch.arange(seq_len - 1, -1, -1, dtype=torch.int32, device=device)
    key_cache = torch.zeros(2, 128, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros_like(key_cache)

    actual = torch.ops.vllm.qknorm_rope_prefill_attention_with_cache(
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
        1.0 / math.sqrt(HEAD_DIM),
    )
    expected_output = reference_attention(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_query_heads,
        num_kv_heads,
        1e-6,
    )
    torch.testing.assert_close(actual, expected_output, rtol=2e-2, atol=2e-2)

    _, raw_k, expected_v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    raw_k = raw_k.view(seq_len, num_kv_heads, HEAD_DIM).float()
    expected_k = (
        (raw_k * torch.rsqrt(raw_k.square().mean(dim=-1, keepdim=True) + 1e-6) * k_weight.float())
        .to(torch.bfloat16)
        .float()
    )
    cos, sin = cos_sin_cache[positions].float().chunk(2, dim=-1)
    first, second = expected_k.chunk(2, dim=-1)
    expected_k = torch.cat(
        (
            first * cos[:, None] - second * sin[:, None],
            second * cos[:, None] + first * sin[:, None],
        ),
        dim=-1,
    ).to(torch.bfloat16)
    expected_v = expected_v.view(seq_len, num_kv_heads, HEAD_DIM)

    flat_k = key_cache.view(-1, num_kv_heads, HEAD_DIM)
    flat_v = value_cache.view(-1, num_kv_heads, HEAD_DIM)
    torch.testing.assert_close(flat_k[slot_mapping.long()], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(flat_v[slot_mapping.long()], expected_v, rtol=0, atol=0)
    assert torch.count_nonzero(flat_k[seq_len:]).item() == 0
    assert torch.count_nonzero(flat_v[seq_len:]).item() == 0


def test_qknorm_rope_kv_matches_reference():
    torch.manual_seed(23)
    device = torch.device("npu")
    seq_len, num_query_heads, num_kv_heads = 17, 16, 8
    q_size = num_query_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(seq_len, q_size + 2 * kv_size, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    angles = torch.randn(seq_len, HEAD_DIM // 2, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)

    key, value = torch.ops.vllm.qknorm_rope_kv(
        qkv, k_weight, cos_sin_cache, positions, num_query_heads, num_kv_heads, 1e-6
    )
    _, raw_key, expected_value = qkv.split((q_size, kv_size, kv_size), dim=-1)
    raw_key = raw_key.view(seq_len, num_kv_heads, HEAD_DIM).float()
    expected_key = (
        (raw_key * torch.rsqrt(raw_key.square().mean(dim=-1, keepdim=True) + 1e-6) * k_weight.float())
        .to(torch.bfloat16)
        .float()
    )
    cos, sin = cos_sin_cache[positions].float().chunk(2, dim=-1)
    first, second = expected_key.chunk(2, dim=-1)
    expected_key = torch.cat(
        (first * cos[:, None] - second * sin[:, None], second * cos[:, None] + first * sin[:, None]),
        dim=-1,
    ).to(torch.bfloat16)

    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value.view_as(value), rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [8, 17])
def test_chunked_kv_cache_then_decode_matches_full_reference(chunk_size: int):
    """Chunked cache publication must preserve the following decode result."""
    torch.manual_seed(31)
    device = torch.device("npu")
    prompt_len, total_len = 33, 34
    num_query_heads, num_kv_heads = 16, 8
    q_size = num_query_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(total_len, q_size + 2 * kv_size, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.arange(total_len, dtype=torch.int64, device=device)
    angles = torch.randn(total_len, HEAD_DIM // 2, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    key_cache = torch.zeros(1, 128, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros_like(key_cache)
    slots = torch.arange(prompt_len, dtype=torch.int32, device=device)

    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        torch.ops.vllm.qknorm_rope_kv_cache(
            qkv[start:end],
            k_weight,
            cos_sin_cache,
            positions[start:end],
            key_cache,
            value_cache,
            slots[start:end],
            num_query_heads,
            num_kv_heads,
            1e-6,
        )

    decode_key, decode_value = torch.ops.vllm.qknorm_rope_kv(
        qkv[-1:],
        k_weight,
        cos_sin_cache,
        positions[-1:],
        num_query_heads,
        num_kv_heads,
        1e-6,
    )
    raw_q = qkv[-1:, :q_size].view(1, num_query_heads, HEAD_DIM).float()
    decode_q = (raw_q * torch.rsqrt(raw_q.square().mean(dim=-1, keepdim=True) + 1e-6) * q_weight.float()).to(
        torch.bfloat16
    )
    cos, sin = cos_sin_cache[positions[-1:]].float().chunk(2, dim=-1)
    first, second = decode_q.float().chunk(2, dim=-1)
    decode_q = torch.cat(
        (
            first * cos[:, None] - second * sin[:, None],
            second * cos[:, None] + first * sin[:, None],
        ),
        dim=-1,
    ).to(torch.bfloat16)
    cached_key = key_cache.view(-1, num_kv_heads, HEAD_DIM)[:prompt_len]
    cached_value = value_cache.view(-1, num_kv_heads, HEAD_DIM)[:prompt_len]
    actual = torch_npu.npu_fused_infer_attention_score(
        query=decode_q,
        key=torch.cat((cached_key, decode_key), dim=0),
        value=torch.cat((cached_value, decode_value), dim=0),
        input_layout="TND",
        actual_seq_lengths=[1],
        actual_seq_lengths_kv=[total_len],
        num_heads=num_query_heads,
        num_key_value_heads=num_kv_heads,
        sparse_mode=0,
        scale=1.0 / math.sqrt(HEAD_DIM),
    )[0]
    expected = reference_attention(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_query_heads,
        num_kv_heads,
        1e-6,
    )[-1:]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
