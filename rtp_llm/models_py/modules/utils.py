from __future__ import annotations

import torch

from rtp_llm.ops.compute_ops import DeviceType, get_device


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))  # type: ignore


def is_cuda():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.Cuda:
        return True
    else:
        return False


def is_hip():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.ROCm:
        return True
    else:
        return False


def cudagraph_debug_kernel(
    data: torch.Tensor,
    info_id: int = 1,
    m: int = 0,
    n: int = 0,
    row_len: int = 0,
    name: str = "cudagraph_debug_kernel",
):
    print(f"{name} shape is {data.shape}")
    if data.dim() == 1:
        data = data.unsqueeze(0)
    data = data.contiguous().to(torch.float32)
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    row_len = data.size(1) if row_len == 0 else row_len
    n = data.size(1) if n == 0 else n
    m = data.size(0) if m == 0 else m
    m = m if m < 10 else 10
    debug_op = rtp_llm_ops.DebugKernelOp()
    debug_op.forward(
        data=data,
        start_row=0,
        start_col=0,
        m=1,
        n=n,
        row_len=row_len,  # 每行的长度
        info_id=info_id,
    )
