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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _vecdot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_batch,
    vdim: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    rows = ext.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < n_batch
    acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)

    for start in range(0, vdim, BLOCK_COLS):
        cols = start + tl.arange(0, BLOCK_COLS)
        mask = row_mask[:, None] & (cols[None, :] < vdim)
        offsets = rows[:, None] * vdim + cols[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += x * y

    tl.store(out_ptr + rows, tl.sum(acc, axis=1), mask=row_mask)


def _block_config(n_batch, vdim):
    block_cols = min(1024, max(32, triton.next_power_of_2(vdim)))
    max_rows = max(1, 8192 // block_cols)
    block_rows = min(max_rows, triton.next_power_of_2(n_batch))
    return block_rows, block_cols


def _launch_vecdot(x, y, out):
    vdim = x.shape[-1]
    n_batch = x.numel() // vdim
    block_rows, block_cols = _block_config(n_batch, vdim)
    grid = (triton.cdiv(n_batch, block_rows),)
    with torch_device_fn.device(x.device):
        _vecdot_kernel[grid](
            x,
            y,
            out,
            n_batch,
            vdim,
            BLOCK_ROWS=block_rows,
            BLOCK_COLS=block_cols,
        )


def _prepare_inputs(x, y, dim):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match")
    ndim = x.dim()
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )

    if dim != ndim - 1:
        x = x.movedim(dim, -1)
        y = y.movedim(dim, -1)
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    return x, y


def _linalg_vecdot_impl(x, y, dim, out=None):
    x, y = _prepare_inputs(x, y, dim)
    out_shape = x.shape[:-1]

    direct_out = out is not None and out.is_contiguous() and out.shape == out_shape
    if direct_out:
        result = out
    else:
        result = torch.empty(
            out_shape if out_shape else (), dtype=x.dtype, device=x.device
        )

    _launch_vecdot(x, y, result)
    if out is not None and not direct_out:
        out.copy_(result)
        return out
    return result


def linalg_vecdot(x, y, dim=-1):
    logger.debug("GEMS_ASCEND LINALG_VECDOT")
    return _linalg_vecdot_impl(x, y, dim)


def linalg_vecdot_out(x, y, dim=-1, out=None):
    logger.debug("GEMS_ASCEND LINALG_VECDOT_OUT")
    if out is None:
        return _linalg_vecdot_impl(x, y, dim)
    return _linalg_vecdot_impl(x, y, dim, out=out)
