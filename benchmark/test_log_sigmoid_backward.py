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
_CORE_THROUGHPUT_SHAPES = [
    (16 * 1024 * 1024,),
    (32 * 1024 * 1024,),
    (4096, 4096),
    (64, 512, 512),
    (128, 512, 512),
]
_COMPREHENSIVE_THROUGHPUT_SHAPES = [
    *_CORE_THROUGHPUT_SHAPES[:2],
    (1024, 16384),
    (1024, 24576),
    (1024, 32768),
    (1024, 49152),
    (1024, 65536),
    *_CORE_THROUGHPUT_SHAPES[2:3],
    (64, 64, 4096),
    (64, 64, 6144),
    (64, 64, 8192),
    (64, 64, 12288),
    *_CORE_THROUGHPUT_SHAPES[3:],
]


def torch_log_sigmoid_backward(grad_output, inp, buffer):
    if _HAS_NATIVE_ASCEND_KERNEL:
        return torch.ops.aten.log_sigmoid_backward(grad_output, inp, buffer)

    # Some Ascend PyTorch builds do not provide this ATen C kernel. In that
    # case compare against the equivalent composition of native device ops.
    return grad_output * torch.sigmoid(-inp)


class LogSigmoidBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Use the same throughput-oriented 1D, 2D, and 3D workloads on every
        # backend. Tiny shapes mostly measure framework and kernel-launch floors,
        # while the generic billion-element cases are unstable on shared devices.
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = _COMPREHENSIVE_THROUGHPUT_SHAPES
        else:
            self.shapes = _CORE_THROUGHPUT_SHAPES

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
