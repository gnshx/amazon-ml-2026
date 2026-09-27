#!/usr/bin/env python3
"""Benchmark Multi-Core Speed for Test Submission Generator."""

import multiprocessing as mp
import time

def worker_task(chunk):
    # Dummy worker simulation
    total = sum(x * x for x in chunk)
    return total

def main():
    print(f"Available CPU count: {mp.cpu_count()}")
    start = time.time()
    chunks = [list(range(i * 100000, (i + 1) * 100000)) for i in range(16)]
    with mp.Pool(processes=16) as pool:
        res = pool.map(worker_task, chunks)
    print(f"1.6 million items processed across 16 cores in {time.time() - start:.3f}s. Result sum: {sum(res)}")

if __name__ == "__main__":
    main()
