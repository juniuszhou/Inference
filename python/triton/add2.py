# basic example of triton

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    # 2D grid：把 (pid0, pid1) 压成一维线性编号，两个维度都参与算地址
    block_id: int = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    block_start: int = block_id * BLOCK_SIZE
    offsets: tl.tensor = block_start + tl.arange(0, BLOCK_SIZE)
    mask: tl.tensor = offsets < n_elements
    x: tl.tensor = tl.load(x_ptr + offsets, mask=mask)
    y: tl.tensor = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output: torch.Tensor = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    # size of the tensor
    n_elements: int = x.numel()
    # 每个 program 处理 BLOCK_SIZE 个元素，所以 grid 是 1D 的 cdiv
    BLOCK_SIZE: int = 1024
    grid: tuple[int] = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


# TRITON_PRINT_AUTOTUNING=1 uv run python basic.py
if __name__ == "__main__":
    print("Using device:", DEVICE)
    torch.manual_seed(0)
    x: torch.Tensor = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
    y: torch.Tensor = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
    out: torch.Tensor = add(x, y)

    host_x: torch.Tensor = out.cpu()
    torch.testing.assert_close(out, x + y)
    print("result:", host_x)
