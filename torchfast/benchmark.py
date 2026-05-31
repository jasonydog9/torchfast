import time
import torch


def benchmark(fn, *args, runs: int = 500, warmup: int = 20) -> float:
    """
    Benchmark *fn(*args)* and return mean runtime in milliseconds.

    Parameters
    ----------
    fn      : callable to benchmark
    *args   : positional arguments forwarded to fn
    runs    : number of timed iterations
    warmup  : number of un-timed warmup iterations

    Returns
    -------
    float   : mean wall-clock time per call in milliseconds
    """
    for _ in range(warmup):
        fn(*args)

    start = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    end = time.perf_counter()

    return (end - start) / runs * 1000.0
