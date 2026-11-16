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
    block_start: int = block_id * BLOCK_SIZE
    offsets: tl.tensor = block_start + tl.arange(0, BLOCK_SIZE)
    mask: tl.tensor = offsets < n_elements
    x: tl.tensor = tl.load(x_ptr + offsets, mask=mask)
    y: tl.tensor = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@triton.jit
def add_kernel_2(
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


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

    # benchmark.run(print_data=True, show_plots=True)
