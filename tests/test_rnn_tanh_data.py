# Copyright 2026 FlagOS Contributors
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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

_RNN_ACCELERATOR_AVAILABLE = not cfg.TO_CPU and (
    (flag_gems.device == "cuda" and torch.cuda.is_available())
    or (
        flag_gems.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available()
    )
)


@pytest.fixture(autouse=True)
def _ieee_reference():
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def _reference_rnn(*args):
    def to_reference(arg):
        if isinstance(arg, torch.Tensor):
            return utils.to_reference(arg)
        if isinstance(arg, (tuple, list)):
            return type(arg)(to_reference(value) for value in arg)
        return arg

    return torch.ops.aten.rnn_tanh.data(*(to_reference(arg) for arg in args))


def _assert_rnn_close(actual, expected):
    utils.gems_assert_close(actual[0], expected[0], torch.float32, atol=1e-4)
    utils.gems_assert_close(actual[1], expected[1], torch.float32, atol=1e-4)


@pytest.mark.rnn_tanh_data
@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_rnn_tanh_data(num_layers, bidirectional):
    dtype = torch.float32
    input_size, hidden_size = 16, 16
    padded = torch.randn(5, 4, input_size, dtype=dtype)
    lengths = torch.tensor([5, 4, 2, 1], dtype=torch.int64)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded, lengths, enforce_sorted=True
    )
    packed_data = packed.data.to(flag_gems.device)
    directions = 2 if bidirectional else 1
    hx = torch.randn(
        num_layers * directions,
        4,
        hidden_size,
        device=flag_gems.device,
        dtype=dtype,
    )
    module = torch.nn.RNN(
        input_size,
        hidden_size,
        num_layers,
        bidirectional=bidirectional,
    ).to(device=flag_gems.device, dtype=dtype)
    params = tuple(module._flat_weights)
    args = (
        packed_data,
        packed.batch_sizes,
        hx,
        params,
        True,
        num_layers,
        0.0,
        True,
        bidirectional,
    )

    reference = _reference_rnn(*args)
    actual = flag_gems.rnn_tanh_data(*args)
    _assert_rnn_close(actual, reference)


@pytest.mark.rnn_tanh_data
@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_data_dispatch():
    padded = torch.randn(4, 3, 16)
    lengths = torch.tensor([4, 3, 1], dtype=torch.int64)
    packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths)
    packed_data = packed.data.to(flag_gems.device)
    hx = torch.randn(1, 3, 16, device=flag_gems.device)
    module = torch.nn.RNN(16, 16).to(flag_gems.device)
    params = tuple(module._flat_weights)
    args = (
        packed_data,
        packed.batch_sizes,
        hx,
        params,
        True,
        1,
        0.0,
        False,
        False,
    )

    reference = _reference_rnn(*args)
    actual = flag_gems.rnn_tanh_data(*args)
    _assert_rnn_close(actual, reference)
