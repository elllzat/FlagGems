from typing import Generator

import pytest
import torch

from . import base, consts


class LogSigmoidBackwardBenchmark(base.UnaryPointwiseBenchmark):
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
        torch_op=torch.ops.aten.log_sigmoid_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
