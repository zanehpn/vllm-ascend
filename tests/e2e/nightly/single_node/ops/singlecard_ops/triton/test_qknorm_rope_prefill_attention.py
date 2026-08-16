# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import vllm_ascend.ops  # noqa: F401

HEAD_DIM = 128


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
        raw_k * torch.rsqrt(raw_k.square().mean(dim=-1, keepdim=True) + 1e-6) * k_weight.float()
    ).to(torch.bfloat16).float()
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
