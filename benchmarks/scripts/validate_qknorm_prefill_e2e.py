# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate QK-norm prefill dispatch with a real vLLM engine.

Run this file in two fresh processes with the optimization disabled/enabled,
then use ``--compare``.  Besides token/logprob outputs, the runner records raw
logits and every written block in every layer's paged KV cache.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams

MAX_LOGIT_ABS_ERROR = 0.5
MAX_LOGPROB_ABS_ERROR = 0.5
MAX_CACHE_ABS_ERROR = 0.5
MAX_CACHE_MEAN_ERROR = 1e-3


def _prompts() -> dict[str, list[str]]:
    # More than one 128-token cache block, while still below the 256-token
    # scheduler budget, so the second request must exercise a real prefix hit.
    shared = "A reliable distributed system must handle partial failure. " * 16
    return {
        "chunked_prefill": [
            ("The quick brown fox jumps over the lazy dog. " * 40) + "Summarize the repeated sentence."
        ],
        "mixed_batch": [
            "Hi",
            "The capital of France is",
            "List three uses of matrix multiplication.",
            ("NPU inference benefits from batching. " * 18) + "Explain why.",
        ],
        "prefix_seed": [shared + "State the central requirement."],
        "prefix_hit": [shared + "Give one concrete example."],
        "multi_request": [f"Request {index}: give a deterministic fact about number {index}." for index in range(8)],
        # Keep the only dispatch-eligible request last so its cache block
        # cannot be overwritten before the per-layer cache snapshot.
        "full_prefill": ["Explain why the sky appears blue in two sentences."],
    }


def _runner(llm: LLM):
    core_client = llm.llm_engine.engine_core
    engine_core = getattr(core_client, "engine_core", core_client)
    worker_wrapper = engine_core.model_executor.driver_worker
    worker = getattr(worker_wrapper, "worker", worker_wrapper)
    return worker.model_runner


def _layer_caches(runner) -> dict[str, list[torch.Tensor]]:
    contexts = runner.compilation_config.static_forward_context
    caches: dict[str, list[torch.Tensor]] = {}
    for layer_name, layer in sorted(contexts.items()):
        layer_cache = getattr(layer, "kv_cache", None)
        if layer_cache is None:
            continue
        if isinstance(layer_cache, torch.Tensor):
            components = [layer_cache]
        else:
            components = [item for item in layer_cache if isinstance(item, torch.Tensor)]
        if components:
            caches[layer_name] = components
    return caches


def _zero_unique_caches(caches: dict[str, list[torch.Tensor]]) -> None:
    seen: set[int] = set()
    for components in caches.values():
        for tensor in components:
            pointer = tensor.data_ptr()
            if pointer not in seen:
                tensor.zero_()
                seen.add(pointer)
    torch.npu.synchronize()


def _written_cache_blocks(
    caches: dict[str, list[torch.Tensor]],
) -> dict[str, list[dict[str, torch.Tensor]]]:
    result: dict[str, list[dict[str, torch.Tensor]]] = {}
    for layer_name, components in caches.items():
        result[layer_name] = []
        for tensor in components:
            by_block = tensor.reshape(tensor.shape[0], -1)
            written = torch.any(by_block != 0, dim=1)
            indices = torch.nonzero(written, as_tuple=False).flatten()
            result[layer_name].append(
                {
                    "indices": indices.cpu(),
                    "data": tensor.index_select(0, indices).cpu(),
                }
            )
    return result


def _serialize_outputs(outputs) -> list[dict[str, Any]]:
    serialized = []
    for request in outputs:
        sample = request.outputs[0]
        steps = []
        for token_logprobs in sample.logprobs or []:
            steps.append({str(token_id): float(logprob.logprob) for token_id, logprob in token_logprobs.items()})
        serialized.append(
            {
                "prompt_token_ids": list(request.prompt_token_ids),
                "num_cached_tokens": request.num_cached_tokens,
                "token_ids": list(sample.token_ids),
                "text": sample.text,
                "logprobs": steps,
            }
        )
    return serialized


def run(args: argparse.Namespace) -> None:
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        gpu_memory_utilization=0.7,
    )
    runner = _runner(llm)
    caches = _layer_caches(runner)
    if not caches:
        raise RuntimeError("No layer KV caches were found")
    _zero_unique_caches(caches)

    logits: list[torch.Tensor] = []
    original_compute_logits = runner.model.compute_logits

    def capture_logits(*call_args, **call_kwargs):
        output = original_compute_logits(*call_args, **call_kwargs)
        logits.append(output.detach().cpu())
        return output

    runner.model.compute_logits = capture_logits
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True, logprobs=20)
    scenario_outputs = {}
    for scenario, prompts in _prompts().items():
        scenario_outputs[scenario] = _serialize_outputs(llm.generate(prompts, params, use_tqdm=False))
    if args.multimodal:
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model)
        image = Image.new("RGB", (224, 224), color=(32, 96, 160))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "Describe the dominant color in one sentence.",
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        scenario_outputs["multimodal_image"] = _serialize_outputs(
            llm.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    }
                ],
                params,
                use_tqdm=False,
            )
        )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(scenario_outputs, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(logits, output_prefix.with_name(output_prefix.name + "_logits.pt"))
    torch.save(
        _written_cache_blocks(caches),
        output_prefix.with_name(output_prefix.name + "_cache.pt"),
    )
    print(
        json.dumps(
            {
                "scenarios": len(scenario_outputs),
                "requests": sum(len(value) for value in scenario_outputs.values()),
                "logit_calls": len(logits),
                "cache_layers": len(caches),
            },
            sort_keys=True,
        )
    )
    llm.llm_engine.engine_core.shutdown()


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    if actual.shape != expected.shape:
        raise AssertionError(f"Tensor shapes differ: actual={actual.shape}, expected={expected.shape}")
    difference = (actual.float() - expected.float()).abs()
    return float(difference.max().item()), float(difference.mean().item())


def compare(args: argparse.Namespace) -> None:
    """Compare request outputs, raw logits, and every written paged-cache block."""
    baseline_prefix = Path(args.compare[0])
    optimized_prefix = Path(args.compare[1])
    baseline_outputs = json.loads(baseline_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    optimized_outputs = json.loads(optimized_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if baseline_outputs.keys() != optimized_outputs.keys():
        raise AssertionError("Scenario sets differ")

    token_mismatches = []
    top_logprob_set_mismatches = []
    max_logprob_error = 0.0
    for scenario, baseline_requests in baseline_outputs.items():
        if scenario not in optimized_outputs:
            raise AssertionError(f"Optimized output is missing scenario {scenario}")
        optimized_requests = optimized_outputs[scenario]
        if len(baseline_requests) != len(optimized_requests):
            raise AssertionError(f"Request count differs for scenario {scenario}")
        for index, (baseline, optimized) in enumerate(zip(baseline_requests, optimized_requests, strict=True)):
            if baseline["prompt_token_ids"] != optimized["prompt_token_ids"]:
                raise AssertionError(f"Prompt tokens differ for {scenario}[{index}]")
            if baseline.get("num_cached_tokens") != optimized.get("num_cached_tokens"):
                raise AssertionError(f"Prefix cache hit differs for {scenario}[{index}]")
            if baseline["token_ids"] != optimized["token_ids"]:
                token_mismatches.append(f"{scenario}[{index}]")
            for baseline_step, optimized_step in zip(baseline["logprobs"], optimized["logprobs"], strict=True):
                if baseline_step.keys() != optimized_step.keys():
                    top_logprob_set_mismatches.append(f"{scenario}[{index}]")
                common_tokens = baseline_step.keys() & optimized_step.keys()
                for token_id in common_tokens:
                    max_logprob_error = max(
                        max_logprob_error,
                        abs(baseline_step[token_id] - optimized_step[token_id]),
                    )

    baseline_logits = torch.load(
        baseline_prefix.with_name(baseline_prefix.name + "_logits.pt"),
        map_location="cpu",
        weights_only=True,
    )
    optimized_logits = torch.load(
        optimized_prefix.with_name(optimized_prefix.name + "_logits.pt"),
        map_location="cpu",
        weights_only=True,
    )
    if len(baseline_logits) != len(optimized_logits):
        raise AssertionError("The number of captured logit calls differs")
    logit_errors = [
        _tensor_error(optimized, baseline)
        for baseline, optimized in zip(baseline_logits, optimized_logits, strict=True)
    ]

    baseline_cache = torch.load(
        baseline_prefix.with_name(baseline_prefix.name + "_cache.pt"),
        map_location="cpu",
        weights_only=True,
    )
    optimized_cache = torch.load(
        optimized_prefix.with_name(optimized_prefix.name + "_cache.pt"),
        map_location="cpu",
        weights_only=True,
    )
    cache_errors = {}
    for layer_name, baseline_components in baseline_cache.items():
        optimized_components = optimized_cache[layer_name]
        component_errors = []
        for baseline, optimized in zip(baseline_components, optimized_components, strict=True):
            if not torch.equal(baseline["indices"], optimized["indices"]):
                raise AssertionError(f"Written cache blocks differ for {layer_name}")
            component_errors.append(_tensor_error(optimized["data"], baseline["data"]))
        cache_errors[layer_name] = component_errors

    report = {
        "scenarios": {
            scenario: {
                "prompt_lengths": [len(request["prompt_token_ids"]) for request in requests],
                "cached_tokens": [request.get("num_cached_tokens") for request in requests],
                "generated_tokens_per_request": [len(request["token_ids"]) for request in requests],
            }
            for scenario, requests in baseline_outputs.items()
        },
        "token_mismatches": token_mismatches,
        "top_logprob_set_mismatches": sorted(set(top_logprob_set_mismatches)),
        "max_logprob_error": max_logprob_error,
        "logit_calls": len(logit_errors),
        "max_logit_error": max(error[0] for error in logit_errors),
        "mean_logit_error": max(error[1] for error in logit_errors),
        "cache_layers": len(cache_errors),
        "max_cache_error": max(error[0] for components in cache_errors.values() for error in components),
        "mean_cache_error": max(error[1] for components in cache_errors.values() for error in components),
        "per_layer_cache_error": cache_errors,
    }
    report["passed"] = (
        not token_mismatches
        and report["max_logit_error"] <= MAX_LOGIT_ABS_ERROR
        and max_logprob_error <= MAX_LOGPROB_ABS_ERROR
        and report["max_cache_error"] <= MAX_CACHE_ABS_ERROR
        and report["mean_cache_error"] <= MAX_CACHE_MEAN_ERROR
    )
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if token_mismatches:
        raise AssertionError(f"Generated tokens differ: {token_mismatches}")
    if not report["passed"]:
        raise AssertionError("Logit, logprob, or per-layer cache error exceeded tolerance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/tmp/models/Qwen3-0.6B")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--output-prefix")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "OPTIMIZED"))
    parser.add_argument("--report", default="qknorm_e2e_report.json")
    args = parser.parse_args()
    if args.compare:
        compare(args)
    elif args.output_prefix:
        run(args)
    else:
        parser.error("either --output-prefix or --compare is required")


if __name__ == "__main__":
    main()
