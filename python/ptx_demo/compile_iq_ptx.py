"""
Demo: Use CompileIQ to find the optimal CUDA launch configuration
for the vecadd PTX kernel in this directory.

The "object" is the real PTX file (vecadd.ptx).  CompileIQ searches
over block_size and num_elements to minimise kernel execution time.
"""

# All imports the objective needs — declared at module level so they are
# available inside the worker subprocess that CompileIQ spawns.
import os
import sys

import compileiq.search_spaces.base as ss
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration

# Ensure the parent directory (containing `ptx_demo`) is on sys.path so the
# worker subprocess can import `ptx_demo.cuda_driver`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx_demo.cuda_driver import run_vecadd

# Absolute path to the PTX file (must be visible from worker subprocesses)
_PTX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vecadd.ptx",
)


def objective(config):
    """
    Objective function executed by CompileIQ workers.

    Receives a *config* dict with keys ``block_size`` and ``num_elements``
    (both chosen by the search algorithm), launches the vecadd kernel
    with those parameters, and returns the median execution time in
    milliseconds.  CompileIQ minimises this score.
    """
    block_size = config["block_size"]
    num_elements = config["num_elements"]

    elapsed_ms = run_vecadd(
        block_size=block_size,
        num_elements=num_elements,
        ptx_path=_PTX_PATH,
    )
    return elapsed_ms


def main():
    # Search space: common CUDA block sizes × problem sizes
    dna_config = {
        "block_size": ss.choice([32, 64, 128, 256, 512, 1024]),
        "num_elements": ss.choice([1024, 4096, 16384, 65536, 262144, 1048576]),
    }

    search_config = SearchConfiguration(
        generations=5,
        problem_type="min",
        num_objectives=1,
    )

    tuner = Search(
        objective_function=objective,
        search_space=dna_config,
        search_config=search_config,
    )

    results = tuner.start()
    print(f"\nEntire Results Dataframe:\n{results.get_results()}")
    print(f"\nBest Result: {results.get_best_result()}")


if __name__ == "__main__":
    main()
