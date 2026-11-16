# basic example of triton
from collections.abc import Callable

import torch
import triton
import triton.language as tl


@triton.autotune(
    # looks like each warp hangle size as 128 elements
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    # the two above configs will be evaluated anytime
    # the value of n_elements changes
    key=["n_elements"],
    cache_results=True,
)
@triton.jit
def add_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    block_id: int = tl.program_id(0)
    block_start: int = block_id * BLOCK_SIZE
    offsets: tl.tensor = block_start + tl.arange(0, BLOCK_SIZE)
    mask: tl.tensor = offsets < n_elements
    x: tl.tensor = tl.load(x_ptr + offsets, mask=mask)
    y: tl.tensor = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output: torch.Tensor = torch.empty_like(x)
    n_elements: int = x.numel()
    # 注意：grid 必须从 meta 里取 BLOCK_SIZE，因为它是 tuner 选出来的
    grid: Callable[[dict[str, int]], tuple[int]] = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )
    add_kernel[grid](x, y, output, n_elements)
    return output


# TRITON_PRINT_AUTOTUNING=1 uv run python basic.py
if __name__ == "__main__":
    torch.manual_seed(0)
    x: torch.Tensor = torch.randn(2**16, device="cuda", dtype=torch.float32)
    y: torch.Tensor = torch.randn(2**16, device="cuda", dtype=torch.float32)
    out: torch.Tensor = add(x, y)
    torch.testing.assert_close(out, x + y)
    print("best config:", add_kernel.best_config)
