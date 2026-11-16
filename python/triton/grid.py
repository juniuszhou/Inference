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
    block_id: int = tl.program_id(0)
    thread_id = tl.num_programs(0)
    block_start: int = block_id * BLOCK_SIZE
    offsets: tl.tensor = block_start + tl.arange(0, BLOCK_SIZE)
    mask: tl.tensor = offsets < n_elements
    x: tl.tensor = tl.load(x_ptr + offsets, mask=mask)
    y: tl.tensor = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, grid: tuple[int, ...]) -> torch.Tensor:
    output: torch.Tensor = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    # size of the tensor
    n_elements: int = x.numel()
    # split according to the block size
    # grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # print("grid:", grid())

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    print("Using device:", DEVICE)
    torch.manual_seed(0)
    x: torch.Tensor = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
    y: torch.Tensor = torch.randn(2**16, device=DEVICE, dtype=torch.float32)
    out: torch.Tensor = add(x, y, grid=(1, 2**16 // 1024))

    host_x: torch.Tensor = out.cpu()
    torch.testing.assert_close(out, x + y)
    print("result:", host_x)
