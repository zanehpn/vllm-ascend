# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.ops.triton.batch_invariant.matmul import linear_batch_invariant


def test_batch_invariant_linear_supports_vision_encoder_shape():
    torch.manual_seed(41)
    input_tensor = torch.randn((2, 3, 256), dtype=torch.bfloat16, device="npu")
    weight = torch.randn((128, 256), dtype=torch.bfloat16, device="npu")
    bias = torch.randn((128,), dtype=torch.bfloat16, device="npu")

    actual = linear_batch_invariant(input_tensor, weight, bias)
    expected = torch.nn.functional.linear(input_tensor, weight, bias)

    assert actual.shape == (2, 3, 128)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
