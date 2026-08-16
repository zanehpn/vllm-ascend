# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime dispatch for experimental Qwen3 non-materializing prefill."""

import os

import torch
import torch.nn.functional as F
import torch_npu
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.device.device_op import DeviceOperator

SUPPORTED_HEAD_DIM = 128
SUPPORTED_QUERY_HEADS = (16, 32)
SUPPORTED_KV_HEADS = 8
SUPPORTED_MAX_SEQ_LEN = 128


def _print_first_layer_component_diagnostics(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fused_output: torch.Tensor,
    baseline_output: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    scale: float,
) -> None:
    """CPU/FP32 diagnostic decomposition; never used outside diagnostic mode."""
    tokens = qkv.shape[0]
    q_size = num_query_heads * head_dim
    kv_size = num_kv_heads * head_dim
    raw_q, raw_k, _ = qkv.detach().cpu().split((q_size, kv_size, kv_size), dim=-1)
    raw_q = raw_q.view(tokens, num_query_heads, head_dim).float()
    raw_k = raw_k.view(tokens, num_kv_heads, head_dim).float()
    weight_q = q_weight.detach().cpu().float()
    weight_k = k_weight.detach().cpu().float()
    cache = cos_sin_cache.detach().cpu().float()
    pos = positions.detach().cpu().long()
    cos, sin = cache[pos].chunk(2, dim=-1)

    def norm_rope(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        x = (x * weight).to(torch.bfloat16).float()
        first, second = x.chunk(2, dim=-1)
        return (
            torch.cat(
                (first * cos[:, None] - second * sin[:, None], second * cos[:, None] + first * sin[:, None]),
                dim=-1,
            )
            .to(torch.bfloat16)
            .float()
        )

    manual_q = norm_rope(raw_q, weight_q)
    manual_k = norm_rope(raw_k, weight_k)
    backend_q = query.detach().cpu().view_as(manual_q).float()
    backend_k = key.detach().cpu().view_as(manual_k).float()

    def error_line(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
        error = actual.float() - expected.float()
        return f"{name}_max={error.abs().max().item():.8f} {name}_mean={error.abs().mean().item():.8f}"

    print(
        "QKNORM_COMPONENT_ROPE " + error_line("q", manual_q, backend_q) + " " + error_line("k", manual_k, backend_k),
        flush=True,
    )

    group = num_query_heads // num_kv_heads
    k = backend_k.repeat_interleave(group, dim=1)
    v = value.detach().cpu().view(tokens, num_kv_heads, head_dim).float().repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", backend_q, k) * scale
    causal = torch.ones(tokens, tokens, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probability_fp32 = torch.softmax(scores, dim=-1)
    output_fp32 = torch.einsum("hts,shd->thd", probability_fp32, v)
    output_score_bf16 = torch.einsum("hts,shd->thd", torch.softmax(scores.to(torch.bfloat16).float(), dim=-1), v)
    output_prob_bf16 = torch.einsum("hts,shd->thd", probability_fp32.to(torch.bfloat16).float(), v)
    fia = baseline_output.detach().cpu().view_as(output_fp32).float()
    fused = fused_output.detach().cpu().view_as(output_fp32).float()
    print(
        "QKNORM_COMPONENT_ATTN "
        + error_line("fp32_vs_fia", output_fp32, fia)
        + " "
        + error_line("scorebf16_vs_fp32", output_score_bf16, output_fp32)
        + " "
        + error_line("probbf16_vs_fp32", output_prob_bf16, output_fp32)
        + " "
        + error_line("fused_vs_fp32", fused, output_fp32),
        flush=True,
    )


def _get_attention_runtime(layer_name: str):
    forward_context = get_forward_context()
    attention_layer = forward_context.no_compile_layers[layer_name]
    metadata = forward_context.attn_metadata
    if isinstance(metadata, dict):
        metadata = metadata[layer_name]
    return attention_layer, metadata


def _can_use_non_materializing_prefill(
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
    # A first full-sized chunk is reported as PrefillNoCache even when more
    # prompt chunks remain. This kernel currently handles a complete sequence,
    # so keep a full scheduler chunk on the established paged FIA path.
    return (
        metadata is not None
        and metadata.attn_state == AscendAttentionState.PrefillNoCache
        and metadata.causal
        and qkv.device.type == "npu"
        and qkv.dtype == torch.bfloat16
        and qkv.ndim == 2
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
    )


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

    if (
        _can_use_non_materializing_prefill(
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
    ):
        key_cache = attention_layer.kv_cache[0]
        value_cache = attention_layer.kv_cache[1]
        # The stock attention forward lazily binds these references during
        # prefill.  Bypassing that forward must preserve the same state for
        # subsequent decode steps.
        if attention_layer.impl.key_cache is None:
            attention_layer.impl.key_cache = key_cache
            attention_layer.impl.value_cache = value_cache
        isolate_cache_write = os.getenv("VLLM_ASCEND_QKNORM_SAFE_CACHE") == "1"
        if isolate_cache_write:
            fused_output = torch.ops.vllm.qknorm_rope_prefill_attention(
                qkv,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                num_query_heads,
                num_kv_heads,
                eps,
                scale,
            )
        else:
            fused_output = torch.ops.vllm.qknorm_rope_prefill_attention_with_cache(
                qkv,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                key_cache,
                value_cache,
                metadata.slot_mapping[: metadata.num_actual_tokens],
                num_query_heads,
                num_kv_heads,
                eps,
                scale,
            )
            # Keep the nested custom-op allocation alive across the outer
            # runtime custom-op boundary.  This also isolates any output/cache
            # aliasing introduced by PrivateUse1 custom-op functionalization.
            fused_output = fused_output.clone()
            notify_kv_cache_written(layer_name)
            if os.getenv("VLLM_ASCEND_QKNORM_DIRECT_CACHE_TINY_SYNC", "1") == "1":
                sync_key, sync_value = torch.ops.vllm.qknorm_rope_kv(
                    qkv[:1],
                    k_weight,
                    cos_sin_cache,
                    positions[:1],
                    num_query_heads,
                    num_kv_heads,
                    eps,
                )
                DeviceOperator.reshape_and_cache(
                    sync_key,
                    sync_value,
                    key_cache,
                    value_cache,
                    metadata.slot_mapping[:1],
                )
                tiny_q = torch.zeros((1, num_query_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
                tiny_kv = torch.zeros((1, num_kv_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
                torch_npu.npu_fused_infer_attention_score(
                    query=tiny_q,
                    key=tiny_kv,
                    value=tiny_kv,
                    input_layout="TND",
                    actual_seq_lengths=[1],
                    actual_seq_lengths_kv=[1],
                    num_key_value_heads=num_kv_heads,
                    num_heads=num_query_heads,
                    scale=scale,
                    sparse_mode=0,
                )
                return fused_output.view(qkv.shape[0], q_hidden_size)
        if envs.VLLM_ASCEND_QKNORM_PREFILL_DIAGNOSTIC or isolate_cache_write:
            fused_snapshot = fused_output.clone()
            torch.npu.synchronize()
            cache_slots = metadata.slot_mapping[: metadata.num_actual_tokens]
            direct_key_snapshot = None
            direct_value_snapshot = None
            if envs.VLLM_ASCEND_QKNORM_PREFILL_DIAGNOSTIC and layer_name == "model.layers.0.self_attn.attn":
                direct_key_snapshot = key_cache.view(-1, num_kv_heads, head_dim)[cache_slots].clone()
                direct_value_snapshot = value_cache.view(-1, num_kv_heads, head_dim)[cache_slots].clone()
            fia_probe = os.getenv("VLLM_ASCEND_QKNORM_FIA_PROBE", "kv_only_tiny_fia")
            if fia_probe == "kv_only_tiny_fia":
                key, value = torch.ops.vllm.qknorm_rope_kv(
                    qkv,
                    k_weight,
                    cos_sin_cache,
                    positions,
                    num_query_heads,
                    num_kv_heads,
                    eps,
                )
                attention_layer.impl.reshape_and_cache(
                    fused_snapshot,
                    key,
                    value,
                    attention_layer.kv_cache,
                    metadata,
                    torch.empty_like(fused_snapshot),
                )
                tiny_q = torch.zeros((1, num_query_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
                tiny_kv = torch.zeros((1, num_kv_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
                torch_npu.npu_fused_infer_attention_score(
                    query=tiny_q,
                    key=tiny_kv,
                    value=tiny_kv,
                    input_layout="TND",
                    actual_seq_lengths=[1],
                    actual_seq_lengths_kv=[1],
                    num_key_value_heads=num_kv_heads,
                    num_heads=num_query_heads,
                    scale=scale,
                    sparse_mode=0,
                )
                return fused_snapshot.view(qkv.shape[0], q_hidden_size)
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
            if fia_probe == "scatter_tiny_fia":
                attention_layer.impl.reshape_and_cache(
                    query.view(qkv.shape[0], num_query_heads, head_dim),
                    key.view(qkv.shape[0], num_kv_heads, head_dim),
                    value.view(qkv.shape[0], num_kv_heads, head_dim),
                    attention_layer.kv_cache,
                    metadata,
                    torch.empty_like(fused_snapshot),
                )
                tiny_q = torch.zeros((1, num_query_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
                tiny_kv = torch.zeros((1, num_kv_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
                torch_npu.npu_fused_infer_attention_score(
                    query=tiny_q,
                    key=tiny_kv,
                    value=tiny_kv,
                    input_layout="TND",
                    actual_seq_lengths=[1],
                    actual_seq_lengths_kv=[1],
                    num_key_value_heads=num_kv_heads,
                    num_heads=num_query_heads,
                    scale=scale,
                    sparse_mode=0,
                )
                return fused_snapshot.view(qkv.shape[0], q_hidden_size)
            if fia_probe == "scatter_fia_zero":
                attention_layer.impl.reshape_and_cache(
                    query.view(qkv.shape[0], num_query_heads, head_dim),
                    key.view(qkv.shape[0], num_kv_heads, head_dim),
                    value.view(qkv.shape[0], num_kv_heads, head_dim),
                    attention_layer.kv_cache,
                    metadata,
                    torch.empty_like(fused_snapshot),
                )
                attention_layer.impl.forward_fused_infer_attention(
                    torch.zeros_like(query).view(qkv.shape[0], num_query_heads, head_dim),
                    torch.zeros_like(key).view(qkv.shape[0], num_kv_heads, head_dim),
                    torch.zeros_like(value).view(qkv.shape[0], num_kv_heads, head_dim),
                    metadata,
                    torch.empty_like(fused_snapshot),
                    attention_layer.kv_cache,
                )
                return fused_snapshot.view(qkv.shape[0], q_hidden_size)
            if fia_probe == "fia_only":
                attention_layer.impl.forward_fused_infer_attention(
                    torch.zeros_like(query).view(qkv.shape[0], num_query_heads, head_dim),
                    torch.zeros_like(key).view(qkv.shape[0], num_kv_heads, head_dim),
                    torch.zeros_like(value).view(qkv.shape[0], num_kv_heads, head_dim),
                    metadata,
                    torch.empty_like(fused_snapshot),
                    attention_layer.kv_cache,
                )
                return fused_snapshot.view(qkv.shape[0], q_hidden_size)
            if fia_probe == "matmul":
                attention_layer.impl.reshape_and_cache(
                    query.view(qkv.shape[0], num_query_heads, head_dim),
                    key.view(qkv.shape[0], num_kv_heads, head_dim),
                    value.view(qkv.shape[0], num_kv_heads, head_dim),
                    attention_layer.kv_cache,
                    metadata,
                    torch.empty_like(fused_snapshot),
                )
                lhs = torch.zeros((16, head_dim), dtype=qkv.dtype, device=qkv.device)
                rhs = torch.zeros((head_dim, 16), dtype=qkv.dtype, device=qkv.device)
                torch.matmul(lhs, rhs)
                torch.npu.synchronize()
                return fused_snapshot.view(qkv.shape[0], q_hidden_size)
            if fia_probe == "clone":
                fia_query, fia_key, fia_value = query.clone(), key.clone(), value.clone()
            elif fia_probe == "zero":
                fia_query = torch.zeros_like(query)
                fia_key = torch.zeros_like(key)
                fia_value = torch.zeros_like(value)
            else:
                fia_query, fia_key, fia_value = query, key, value
            baseline_output = attention_layer(fia_query, fia_key, fia_value)
            if isolate_cache_write and not envs.VLLM_ASCEND_QKNORM_PREFILL_DIAGNOSTIC:
                return fused_snapshot.view(qkv.shape[0], q_hidden_size)
            if direct_key_snapshot is not None and direct_value_snapshot is not None:
                reference_key = key_cache.view(-1, num_kv_heads, head_dim)[cache_slots]
                reference_value = value_cache.view(-1, num_kv_heads, head_dim)[cache_slots]
                print(
                    "QKNORM_CACHE_ERROR "
                    f"key_max={(direct_key_snapshot.float() - reference_key.float()).abs().max().item():.8f} "
                    f"value_max={(direct_value_snapshot.float() - reference_value.float()).abs().max().item():.8f}",
                    flush=True,
                )
            fused_flat = fused_snapshot.view_as(baseline_output).float()
            baseline_fp32 = baseline_output.float()
            error = fused_flat - baseline_fp32
            print(
                "QKNORM_PREFILL_LAYER "
                f"layer={layer_name} tokens={qkv.shape[0]} "
                f"max_abs={error.abs().max().item():.8f} "
                f"mean_abs={error.abs().mean().item():.8f} "
                f"rmse={error.square().mean().sqrt().item():.8f} "
                f"cosine={F.cosine_similarity(fused_flat.flatten(), baseline_fp32.flatten(), dim=0).item():.8f} "
                f"alias={fused_snapshot.data_ptr() == baseline_output.data_ptr()}",
                flush=True,
            )
            if layer_name == "model.layers.0.self_attn.attn":
                _print_first_layer_component_diagnostics(
                    qkv,
                    q_weight,
                    k_weight,
                    cos_sin_cache,
                    positions,
                    query,
                    key,
                    value,
                    fused_snapshot,
                    baseline_output,
                    num_query_heads,
                    num_kv_heads,
                    head_dim,
                    eps,
                    scale,
                )
            return baseline_output
        return fused_output.view(qkv.shape[0], q_hidden_size)

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
