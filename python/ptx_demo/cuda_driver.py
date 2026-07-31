"""pycuda-based wrapper to load PTX and run the vecadd kernel with timing."""

import numpy as np
import pycuda.driver as cuda

_state = {"init": False, "ctx": None, "mod": None}


def run_vecadd(
    block_size: int,
    num_elements: int,
    ptx_path: str,
    warmup: bool = True,
    repeats: int = 5,
) -> float:
    if not _state["init"]:
        cuda.init()
        ctx = cuda.Device(0).make_context()
        mod = cuda.module_from_file(ptx_path)
        _state.update(init=True, ctx=ctx, mod=mod)
        ctx.pop()

    ctx = _state["ctx"]
    mod = _state["mod"]
    ctx.push()
    try:
        func = mod.get_function("vecadd")

        rng = np.random.default_rng(42)
        host_a = rng.standard_normal(num_elements).astype(np.float32)
        host_b = rng.standard_normal(num_elements).astype(np.float32)

        dev_a = cuda.to_device(host_a)
        dev_b = cuda.to_device(host_b)
        dev_c = cuda.mem_alloc(host_a.nbytes)

        start = cuda.Event()
        stop = cuda.Event()

        grid_size = (num_elements + block_size - 1) // block_size
        grid = (grid_size, 1)
        block = (block_size, 1, 1)

        if warmup:
            func(dev_a, dev_b, dev_c, np.uint32(num_elements), block=block, grid=grid)

        times = []
        for _ in range(repeats):
            start.record()
            func(dev_a, dev_b, dev_c, np.uint32(num_elements), block=block, grid=grid)
            stop.record()
            stop.synchronize()
            times.append(start.time_till(stop))

    finally:
        ctx.pop()

    return float(np.median(times))
