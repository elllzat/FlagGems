import logging

import torch

logger = logging.getLogger(__name__)

_BACKEND_SELECT_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.BackendSelect)


def empty(
    *size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """Allocate an uninitialized tensor through the native device allocator."""
    logger.debug("GEMS EMPTY")
    if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
        size = tuple(size[0])
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        import flag_gems.runtime as _rt

        device = torch.device(_rt.device.name)
    return torch.ops.aten.empty.memory_format.redispatch(
        _BACKEND_SELECT_KEYSET,
        list(size),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
