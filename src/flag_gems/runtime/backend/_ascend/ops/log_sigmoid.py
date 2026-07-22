import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    self_fp32 = self.to(tl.float32)
    z = tl.exp(-tl.abs(self_fp32))
    derivative = tl.where(self_fp32 < 0.0, 1.0 / (1.0 + z), z / (1.0 + z))
    return grad_output * derivative


@libentry()
@triton.jit
def log_sigmoid_backward_contiguous_kernel(
    grad_output,
    self,
    grad_input,
    n_elements,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = ext.program_id(0)
    program_count = ext.num_programs(0)
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    for tile in tl.range(0, TILES_PER_PROGRAM):
        offsets = (program_id + tile * program_count) * BLOCK_SIZE + lane_offsets
        mask = offsets < n_elements
        grad = tl.load(grad_output + offsets, mask=mask)
        inp = tl.load(self + offsets, mask=mask).to(tl.float32)
        result = grad * tl.sigmoid(-inp)
        tl.store(grad_input + offsets, result, mask=mask)


def _can_use_contiguous_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (grad_input is None or grad_input.is_contiguous())
    )


def _device_guard(tensor):
    device_index = tensor.device.index
    if device_index is None or device_index == torch_device_fn.current_device():
        return nullcontext()
    return torch_device_fn.device(tensor.device)


def _launch_contiguous_kernel(grad_output, self, grad_input=None):
    if grad_input is None:
        grad_input = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input

    block_size = 8192
    tile_count = triton.cdiv(n_elements, block_size)
    grid_size = min(tile_count, 65535)
    tiles_per_program = triton.cdiv(tile_count, grid_size)
    with _device_guard(self):
        log_sigmoid_backward_contiguous_kernel[(grid_size,)](
            grad_output,
            self,
            grad_input,
            n_elements,
            TILES_PER_PROGRAM=tiles_per_program,
            BLOCK_SIZE=block_size,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_ASCEND LOG_SIGMOID BACKWARD")

    # The device forward path may return an empty buffer, as CUDA does.  The
    # stable recomputation keeps the operator independent of buffer storage.
    del buffer
    if _can_use_contiguous_kernel(grad_output, self):
        return _launch_contiguous_kernel(grad_output, self)
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_ASCEND LOG_SIGMOID BACKWARD OUT")

    del buffer
    if _can_use_contiguous_kernel(grad_output, self, grad_input):
        return _launch_contiguous_kernel(grad_output, self, grad_input)
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
