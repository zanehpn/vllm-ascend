# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime dispatch for experimental Qwen3 non-materializing prefill."""

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend import envs

SUPPORTED_HEAD_DIM = 128
SUPPORTED_QUERY_HEADS = (16, 32)
SUPPORTED_KV_HEADS = 8


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
        return torch.cat(
            (first * cos[:, None] - second * sin[:, None],
             second * cos[:, None] + first * sin[:, None]),
            dim=-1,
        ).to(torch.bfloat16).float()

    manual_q = norm_rope(raw_q, weight_q)
    manual_k = norm_rope(raw_k, weight_k)
    backend_q = query.detach().cpu().view_as(manual_q).float()
    backend_k = key.detach().cpu().view_as(manual_k).float()

    def error_line(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
        error = actual.float() - expected.float()
        return (
            f"{name}_max={error.abs().max().item():.8f} "
            f"{name}_mean={error.abs().mean().item():.8f}"
        )

    print(
        "QKNORM_COMPONENT_ROPE "
        + error_line("q", manual_q, backend_q)
        + " "
        + error_line("k", manual_k, backend_k),
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
    output_score_bf16 = torch.einsum(
        "hts,shd->thd", torch.softmax(scores.to(torch.bfloat16).float(), dim=-1), v
    )
    output_prob_bf16 = torch.einsum(
        "hts,shd->thd", probability_fp32.to(torch.bfloat16).float(), v
    )
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
) -> bool:
    actual_seq_lengths_q = getattr(metadata, "actual_seq_lengths_q", None)
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
    )


def qwen3_qknorm_prefill_attention_impl(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    scale: float,
) -> None:
    attention_layer, metadata = _get_attention_runtime(layer_name)
    q_hidden_size = num_query_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim

    if _can_use_non_materializing_prefill(
        qkv,
        positions,
        cos_sin_cache,
        metadata,
        num_query_heads,
        num_kv_heads,
        head_dim,
    ):
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
        # Correctness scaffold: compute the established fused QK-Norm/RoPE +
        # FIA path on the same inputs.  Until K/V cache writes are integrated
        # into the experimental kernel, this path owns the externally visible
        # output and cache side effects.  The experimental result is retained
        # for layer-by-layer diagnostics only.
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
        baseline_output = attention_layer(query, key, value)
        if envs.VLLM_ASCEND_QKNORM_PREFILL_DIAGNOSTIC:
            fused_flat = fused_output.view_as(baseline_output).float()
            baseline_fp32 = baseline_output.float()
            error = fused_flat - baseline_fp32
            print(
                "QKNORM_PREFILL_LAYER "
                f"layer={layer_name} tokens={qkv.shape[0]} "
                f"max_abs={error.abs().max().item():.8f} "
                f"mean_abs={error.abs().mean().item():.8f} "
                f"rmse={error.square().mean().sqrt().item():.8f} "
                f"cosine={F.cosine_similarity(fused_flat.flatten(), baseline_fp32.flatten(), dim=0).item():.8f}",
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
                    fused_output,
                    baseline_output,
                    num_query_heads,
                    num_kv_heads,
                    head_dim,
                    eps,
                    scale,
                )
        output.copy_(baseline_output.view_as(output))
        return

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
    output.copy_(attention_layer(query, key, value))


def qwen3_qknorm_prefill_attention_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    scale: float,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_qknorm_prefill_attention",
    op_func=qwen3_qknorm_prefill_attention_impl,
    mutates_args=["output"],
    fake_impl=qwen3_qknorm_prefill_attention_fake,
    dispatch_key="PrivateUse1",
)
