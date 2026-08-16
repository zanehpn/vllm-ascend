# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 fused QK-Norm/RoPE/cache publication followed by official FIA."""

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.device.device_op import DeviceOperator

SUPPORTED_HEAD_DIM = 128
SUPPORTED_QUERY_HEADS = (16, 32)
SUPPORTED_KV_HEADS = 8
SUPPORTED_MAX_SEQ_LEN = 128


def _cache_supports_contiguous_write(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    qkv: torch.Tensor,
) -> bool:
    return (
        key_cache.ndim == 4
        and value_cache.shape == key_cache.shape
        and key_cache.device == qkv.device
        and value_cache.device == qkv.device
        and key_cache.dtype == qkv.dtype
        and value_cache.dtype == qkv.dtype
        and key_cache.shape[1] == SUPPORTED_MAX_SEQ_LEN
        and key_cache.shape[2:] == (SUPPORTED_KV_HEADS, SUPPORTED_HEAD_DIM)
        and qkv.shape[0] <= key_cache.shape[1]
    )


def _get_attention_runtime(layer_name: str):
    forward_context = get_forward_context()
    attention_layer = forward_context.no_compile_layers[layer_name]
    metadata = forward_context.attn_metadata
    if isinstance(metadata, dict):
        metadata = metadata[layer_name]
    return attention_layer, metadata


def _can_use_fused_preprocess_fia(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    metadata,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_num_batched_tokens: int,
) -> bool:
    actual_seq_lengths_q = getattr(metadata, "actual_seq_lengths_q", None)
    slot_mapping = getattr(metadata, "slot_mapping", None)
    block_tables = getattr(metadata, "block_tables", None)
    return (
        metadata is not None
        and metadata.attn_state == AscendAttentionState.PrefillNoCache
        and metadata.causal
        and qkv.device.type == "npu"
        and qkv.dtype == torch.bfloat16
        and qkv.ndim == 2
        and qkv.shape[1] == (num_query_heads + 2 * num_kv_heads) * head_dim
        and positions.ndim == 1
        and positions.shape[0] == qkv.shape[0]
        and cos_sin_cache.device == qkv.device
        and cos_sin_cache.dtype == qkv.dtype
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[1] == head_dim
        and head_dim == SUPPORTED_HEAD_DIM
        and num_query_heads in SUPPORTED_QUERY_HEADS
        and num_kv_heads == SUPPORTED_KV_HEADS
        and isinstance(actual_seq_lengths_q, list)
        and len(actual_seq_lengths_q) == 1
        and actual_seq_lengths_q[0] == qkv.shape[0]
        and qkv.shape[0] <= SUPPORTED_MAX_SEQ_LEN
        and qkv.shape[0] < max_num_batched_tokens
        and isinstance(slot_mapping, torch.Tensor)
        and slot_mapping.shape[0] >= qkv.shape[0]
        and isinstance(block_tables, torch.Tensor)
        and block_tables.ndim == 2
        and block_tables.shape[0] >= len(actual_seq_lengths_q)
    )


def _paged_fia(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    num_blocks, block_size, _, _ = key_cache.shape
    key = key_cache.view(num_blocks, block_size, -1)
    value = value_cache.view(num_blocks, block_size, -1)
    batch_size = len(metadata.actual_seq_lengths_q)
    output, _ = DeviceOperator.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=metadata.attn_mask,
        block_table=metadata.block_tables[:batch_size],
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=metadata.actual_seq_lengths_q,
        actual_seq_lengths_kv=metadata.seq_lens_list[:batch_size],
        num_key_value_heads=num_kv_heads,
        num_heads=num_query_heads,
        head_size=head_dim,
        scale=scale,
        key_cache=key_cache,
        value_cache=value_cache,
        current_key=key,
        current_value=value,
        attn_metadata=metadata,
        is_prefill_no_cache=False,
        sparse_mode=3,
    )
    return output.view(query.shape[0], num_query_heads * head_dim)


def qwen3_qknorm_prefill_attention_impl(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_num_batched_tokens: int,
    eps: float,
    scale: float,
) -> torch.Tensor:
    attention_layer, metadata = _get_attention_runtime(layer_name)
    q_hidden_size = num_query_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim

    can_use_fused_preprocess = (
        _can_use_fused_preprocess_fia(
            qkv,
            positions,
            cos_sin_cache,
            metadata,
            num_query_heads,
            num_kv_heads,
            head_dim,
            max_num_batched_tokens,
        )
        and len(attention_layer.kv_cache) > 1
    )
    if can_use_fused_preprocess:
        key_cache = attention_layer.kv_cache[0]
        value_cache = attention_layer.kv_cache[1]
        can_use_fused_preprocess = _cache_supports_contiguous_write(
            key_cache,
            value_cache,
            qkv,
        )

    if can_use_fused_preprocess:
        if attention_layer.impl.key_cache is None:
            attention_layer.impl.key_cache = key_cache
            attention_layer.impl.value_cache = value_cache
        query = torch.ops.vllm.qkv_rmsnorm_rope_cache(
            qkv,
            cos_sin_cache,
            positions,
            q_weight,
            k_weight,
            key_cache,
            value_cache,
            metadata.slot_mapping[: qkv.shape[0]],
            q_hidden_size,
            kv_hidden_size,
            head_dim,
            eps,
            True,
        ).view(qkv.shape[0], num_query_heads, head_dim)
        notify_kv_cache_written(layer_name)
        return _paged_fia(
            query,
            key_cache,
            value_cache,
            metadata,
            num_query_heads,
            num_kv_heads,
            head_dim,
            scale,
        )

    query, key, value = DeviceOperator.split_qkv_rmsnorm_rope(
        input=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        eps=eps,
        q_bias=None,
        k_bias=None,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
    )
    return attention_layer(query, key, value)


def qwen3_qknorm_prefill_attention_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_num_batched_tokens: int,
    eps: float,
    scale: float,
) -> torch.Tensor:
    del q_weight, k_weight, cos_sin_cache, positions, layer_name
    del num_kv_heads, max_num_batched_tokens, eps, scale
    return torch.empty(
        (qkv.shape[0], num_query_heads * head_dim),
        dtype=qkv.dtype,
        device=qkv.device,
    )


direct_register_custom_op(
    op_name="qwen3_qknorm_prefill_attention",
    op_func=qwen3_qknorm_prefill_attention_impl,
    mutates_args=[],
    fake_impl=qwen3_qknorm_prefill_attention_fake,
    dispatch_key="PrivateUse1",
)
