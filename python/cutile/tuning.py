import cuda.tile as ct
import cupy as cp
import numpy as np


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def main():
    N = 2**12
    rng = cp.random.default_rng(0)
    a = rng.random(N, dtype=cp.float32)
    b = rng.random(N, dtype=cp.float32)
    stream = cp.cuda.get_current_stream()

    space = [{"tile": t} for t in (64, 128, 256)]

    def grid_fn(cfg):
        return (ct.cdiv(N, cfg["tile"]), 1, 1)

    def args_fn(cfg):
        return (a, b, cp.empty_like(a), cfg["tile"])

    res = ct.tune.exhaustive_search(
        space, stream, grid_fn, vector_add, args_fn, quiet=True
    )
    print(res.best.config)
    print(res.best.mean_us)

    best_tile = res.best.config["tile"]
    c = cp.empty_like(a)
    ct.launch(stream, grid_fn(res.best.config), vector_add, (a, b, c, best_tile))
    np.testing.assert_array_almost_equal(cp.asnumpy(c), cp.asnumpy(a) + cp.asnumpy(b))
    print("verified with tile =", best_tile)


if __name__ == "__main__":
    main()
