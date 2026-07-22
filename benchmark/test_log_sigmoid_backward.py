from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


def _has_native_ascend_kernel() -> bool:
    if flag_gems.vendor_name != "ascend":
        return True
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::log_sigmoid_backward", "PrivateUse1"
        )
    except (AttributeError, RuntimeError):
        return False


_HAS_NATIVE_ASCEND_KERNEL = _has_native_ascend_kernel()
_DEFAULT_STARTUP_OVERHEAD_SHAPE = (64, 64)
_ASCEND_STARTUP_OVERHEAD_SHAPE = (3584, 3584)
_CORE_SHAPE_REPLACEMENTS = {
    (1024 * 1024 * 1024,): (32 * 1024 * 1024,),
    (1024, 1024, 1024): (128, 512, 512),
}


def torch_log_sigmoid_backward(grad_output, inp, buffer):
    if _HAS_NATIVE_ASCEND_KERNEL:
        return torch.ops.aten.log_sigmoid_backward(grad_output, inp, buffer)

    # Some Ascend PyTorch builds do not provide this ATen C kernel. In that
    # case compare against the equivalent composition of native device ops.
    return grad_output * torch.sigmoid(-inp)


class LogSigmoidBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # The generic unary defaults contain the same billion-element workload
        # twice. Four FP32 tensors make either case consume about 16 GiB, which
        # is not a stable core benchmark on shared devices. Keep both the 1D and
        # 3D throughput coverage with bounded-memory equivalents.
        self.shapes = [
            _CORE_SHAPE_REPLACEMENTS.get(shape, shape) for shape in self.shapes
        ]
        if flag_gems.vendor_name == "ascend":
            # The native 64x64 kernel completes near the device timing floor, so
            # it does not provide a meaningful comparison with a Triton launch.
            # On 910B, 3584x3584 remains in the Triton launch-latency plateau
            # while being large enough to produce stable native measurements.
            self.shapes = [
                (
                    _ASCEND_STARTUP_OVERHEAD_SHAPE
                    if shape == _DEFAULT_STARTUP_OVERHEAD_SHAPE
                    else shape
                )
                for shape in self.shapes
            ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            grad_output = base.generate_tensor_input(shape, cur_dtype, self.device)
            buffer = torch.exp(-torch.abs(inp))
            yield grad_output, inp, buffer


@pytest.mark.log_sigmoid_backward
def test_log_sigmoid_backward():
    bench = LogSigmoidBackwardBenchmark(
        op_name="log_sigmoid_backward",
        torch_op=torch_log_sigmoid_backward,
        gems_op=flag_gems.log_sigmoid_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
