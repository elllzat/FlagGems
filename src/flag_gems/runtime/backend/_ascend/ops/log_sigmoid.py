import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    self_fp32 = self.to(tl.float32)
    z = tl.exp(-tl.abs(self_fp32))
    derivative = tl.where(self_fp32 < 0.0, 1.0 / (1.0 + z), z / (1.0 + z))
    return grad_output * derivative


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_ASCEND LOG_SIGMOID BACKWARD")

    # The device forward path may return an empty buffer, as CUDA does.  The
    # stable recomputation keeps the operator independent of buffer storage.
    del buffer
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_ASCEND LOG_SIGMOID BACKWARD OUT")

    del buffer
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
