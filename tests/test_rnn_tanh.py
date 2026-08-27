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

from unittest import mock

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.rnn_tanh

# T-Head's ACDNN currently rejects tanh RNN mode.  Disable that unavailable
# fast path so Torch's generic implementation remains usable as the reference.
if flag_gems.vendor_name == "thead":
    torch.backends.cudnn.enabled = False

_RNN_ACCELERATOR_AVAILABLE = not cfg.TO_CPU and (
    (flag_gems.device == "cuda" and torch.cuda.is_available())
    or (
        flag_gems.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available()
    )
)


def _make_case(
    seq_len,
    batch_size,
    input_size,
    hidden_size,
    dtype,
    num_layers=1,
    has_biases=True,
    bidirectional=False,
    batch_first=False,
):
    shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    directions = 2 if bidirectional else 1
    hx = torch.randn(
        num_layers * directions,
        batch_size,
        hidden_size,
        dtype=dtype,
        device=flag_gems.device,
    )
    rnn = torch.nn.RNN(
        input_size,
        hidden_size,
        num_layers,
        nonlinearity="tanh",
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(dtype=dtype, device=flag_gems.device)
    return inp, hx, tuple(rnn._flat_weights)


def _assert_rnn_close(actual, expected, dtype):
    atol = {
        torch.float32: 3e-3,
        torch.float16: 6e-3,
        torch.bfloat16: 4e-2,
    }[dtype]
    utils.gems_assert_close(actual[0], expected[0], dtype, atol=atol)
    utils.gems_assert_close(actual[1], expected[1], dtype, atol=atol)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "num_layers,has_biases,bidirectional,batch_first",
    [
        (1, True, False, False),
        (1, False, False, True),
        (1, True, True, False),
        (2, True, False, True),
        (2, False, True, False),
    ],
)
def test_rnn_tanh_forward_modes(
    dtype, num_layers, has_biases, bidirectional, batch_first
):
    inp, hx, params = _make_case(
        5,
        3,
        16,
        24,
        dtype,
        num_layers,
        has_biases,
        bidirectional,
        batch_first,
    )
    reference = torch.rnn_tanh(
        inp,
        hx,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )
    with flag_gems.use_gems():
        actual = torch.rnn_tanh(
            inp,
            hx,
            params,
            has_biases,
            num_layers,
            0.0,
            False,
            bidirectional,
            batch_first,
        )
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_bfloat16_medium_hidden():
    """Cover the NVIDIA bf16 tensor-core selector and its Ascend peer shape."""
    dtype = torch.bfloat16
    inp, hx, params = _make_case(3, 2, 64, 128, dtype)
    reference = torch.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    with flag_gems.use_gems():
        actual = torch.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("seq_len", [17, 20])
def test_rnn_tanh_bfloat16_long_bidirectional(seq_len):
    """Cover full/remainder recurrent chunks across both directions and layers."""
    dtype = torch.bfloat16
    inp, hx, params = _make_case(
        seq_len, 3, 64, 128, dtype, num_layers=2, bidirectional=True
    )
    reference = torch.rnn_tanh(inp, hx, params, True, 2, 0.0, False, True, False)
    with flag_gems.use_gems():
        actual = torch.rnn_tanh(inp, hx, params, True, 2, 0.0, False, True, False)
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [(16, 4, 32), (32, 8, 64), (64, 16, 128)],
)
def test_rnn_tanh_comprehensive_forward_shapes(dtype, shape):
    """Cover comprehensive shapes, including optimized selectors where enabled."""
    seq_len, batch_size, hidden_size = shape
    inp, hx, params = _make_case(
        seq_len,
        batch_size,
        hidden_size,
        hidden_size,
        dtype,
    )
    reference = torch.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    with flag_gems.use_gems():
        actual = torch.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_backward():
    from flag_gems.ops.rnn_tanh import rnn_tanh as gems_rnn_tanh

    dtype = torch.float32
    inp, hx, params = _make_case(4, 2, 16, 16, dtype, num_layers=2, bidirectional=True)

    ref_input = inp.detach().clone().requires_grad_(True)
    ref_hx = hx.detach().clone().requires_grad_(True)
    ref_params = tuple(p.detach().clone().requires_grad_(True) for p in params)
    ref_output, ref_hidden = torch.rnn_tanh(
        ref_input, ref_hx, ref_params, True, 2, 0.0, True, True, False
    )
    (ref_output.sum() + ref_hidden.sum()).backward()

    gems_input = inp.detach().clone().requires_grad_(True)
    gems_hx = hx.detach().clone().requires_grad_(True)
    gems_params = tuple(p.detach().clone().requires_grad_(True) for p in params)
    output, hidden = gems_rnn_tanh(
        gems_input, gems_hx, gems_params, True, 2, 0.0, True, True, False
    )
    (output.sum() + hidden.sum()).backward()

    utils.gems_assert_close(gems_input.grad, ref_input.grad, dtype, atol=5e-3)
    utils.gems_assert_close(gems_hx.grad, ref_hx.grad, dtype, atol=5e-3)
    for actual, expected in zip(gems_params, ref_params):
        # Weight gradients accumulate over both time and batch dimensions, so
        # tensor-core rounding is amplified compared with forward activations.
        utils.gems_assert_close(actual.grad, expected.grad, dtype, atol=3e-2)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_launcher_has_no_torch_composition():
    """Guard the pure-Triton requirement against accidental Torch fallback."""
    from flag_gems.ops.rnn_tanh import rnn_tanh as gems_rnn_tanh

    inp, hx, params = _make_case(3, 2, 16, 16, torch.float32)
    forbidden = ["mm", "matmul", "addmm", "stack", "cat", "tanh"]
    patches = [
        mock.patch.object(torch, name, side_effect=AssertionError(name))
        for name in forbidden
    ]
    for patch in patches:
        patch.start()
    try:
        gems_rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    finally:
        for patch in patches:
            patch.stop()


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_large_hidden_fallback():
    from flag_gems.ops.rnn_tanh import rnn_tanh as gems_rnn_tanh

    inp, hx, params = _make_case(3, 2, 32, 320, torch.float32)
    reference = torch.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    actual = gems_rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    _assert_rnn_close(actual, reference, torch.float32)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_dropout_one():
    from flag_gems.ops.rnn_tanh import rnn_tanh as gems_rnn_tanh

    inp, hx, params = _make_case(5, 3, 16, 16, torch.float32, num_layers=2)
    reference = torch.rnn_tanh(inp, hx, params, True, 2, 1.0, True, False, False)
    actual = gems_rnn_tanh(inp, hx, params, True, 2, 1.0, True, False, False)
    _assert_rnn_close(actual, reference, torch.float32)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_dropout_respects_manual_seed():
    from flag_gems.ops.rnn_tanh import rnn_tanh as gems_rnn_tanh

    inp, hx, params = _make_case(5, 3, 16, 16, torch.float32, num_layers=2)
    torch.manual_seed(2026)
    first = gems_rnn_tanh(inp, hx, params, True, 2, 0.25, True, False, False)
    torch.manual_seed(2026)
    second = gems_rnn_tanh(inp, hx, params, True, 2, 0.25, True, False, False)

    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_rnn_tanh_packed_forward_and_backward(num_layers, bidirectional):
    from flag_gems.ops.rnn_tanh import rnn_tanh_data

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

    ref_data = packed_data.detach().clone().requires_grad_(True)
    ref_hx = hx.detach().clone().requires_grad_(True)
    ref_params = tuple(p.detach().clone().requires_grad_(True) for p in params)
    ref_output, ref_hidden = torch.rnn_tanh(
        ref_data,
        packed.batch_sizes,
        ref_hx,
        ref_params,
        True,
        num_layers,
        0.0,
        True,
        bidirectional,
    )

    actual_data = packed_data.detach().clone().requires_grad_(True)
    actual_hx = hx.detach().clone().requires_grad_(True)
    actual_params = tuple(p.detach().clone().requires_grad_(True) for p in params)
    actual_output, actual_hidden = rnn_tanh_data(
        actual_data,
        packed.batch_sizes,
        actual_hx,
        actual_params,
        True,
        num_layers,
        0.0,
        True,
        bidirectional,
    )
    _assert_rnn_close(
        (actual_output, actual_hidden),
        (ref_output, ref_hidden),
        dtype,
    )
    (ref_output.sum() + ref_hidden.sum()).backward()
    (actual_output.sum() + actual_hidden.sum()).backward()
    utils.gems_assert_close(actual_data.grad, ref_data.grad, dtype, atol=1e-2)
    utils.gems_assert_close(actual_hx.grad, ref_hx.grad, dtype, atol=1e-2)
    for actual, expected in zip(actual_params, ref_params):
        utils.gems_assert_close(actual.grad, expected.grad, dtype, atol=3e-2)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_packed_dispatch():
    padded = torch.randn(4, 3, 16)
    lengths = torch.tensor([4, 3, 1], dtype=torch.int64)
    packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths)
    packed_data = packed.data.to(flag_gems.device)
    hx = torch.randn(1, 3, 16, device=flag_gems.device)
    module = torch.nn.RNN(16, 16).to(flag_gems.device)
    params = tuple(module._flat_weights)
    reference = torch.rnn_tanh(
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
    with flag_gems.use_gems():
        actual = torch.rnn_tanh(
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
    _assert_rnn_close(actual, reference, torch.float32)
