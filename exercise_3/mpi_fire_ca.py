"""MPI forest-fire cellular automaton with 1-D row decomposition.

Run with:
    mpiexec -n 4 python exercise_3/mpi_fire_ca.py --grid 400 --steps 80
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mpi4py is required for this script. Install mpi4py and an MPI runtime.") from exc


def local_step(local_with_halo: np.ndarray, intensity: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = local_with_halo[1:-1]
    burning = local_with_halo == 2
    neighbors = np.zeros_like(center, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbors += burning[dy : dy + center.shape[0], dx : dx + center.shape[1]]
    probability = np.minimum(0.08 + 0.16 * neighbors + 0.15 * intensity, 0.92)
    ignites = (center == 1) & (neighbors > 0) & (rng.random(center.shape) < probability)
    out = center.copy()
    out[center == 2] = 3
    out[ignites] = 2
    return out


def counts_displs(n: int, size: int) -> tuple[list[int], list[int]]:
    counts = [n // size + (1 if r < n % size else 0) for r in range(size)]
    displs = [0]
    for count in counts[:-1]:
        displs.append(displs[-1] + count)
    return counts, displs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=400)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260422)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    row_counts, row_displs = counts_displs(args.grid, size)
    sendcounts = [rows * args.grid for rows in row_counts]
    displs = [row * args.grid for row in row_displs]

    if rank == 0:
        rng = np.random.default_rng(args.seed)
        state = np.ones((args.grid, args.grid), dtype=np.uint8)
        state[rng.random((args.grid, args.grid)) < 0.08] = 0
        state[rng.integers(0, args.grid, 80), rng.integers(0, args.grid, 80)] = 2
        intensity = rng.random((args.grid, args.grid)) * 0.4
    else:
        state = None
        intensity = None

    local = np.empty((row_counts[rank], args.grid), dtype=np.uint8)
    local_intensity = np.empty((row_counts[rank], args.grid), dtype=np.float64)
    comm.Scatterv([state, sendcounts, displs, MPI.UNSIGNED_CHAR], local, root=0)
    comm.Scatterv([intensity, sendcounts, displs, MPI.DOUBLE], local_intensity, root=0)

    comm.Barrier()
    start = time.perf_counter()
    for t in range(args.steps):
        top = np.zeros((1, args.grid), dtype=np.uint8)
        bottom = np.zeros((1, args.grid), dtype=np.uint8)
        if rank > 0:
            comm.Sendrecv(local[:1], dest=rank - 1, sendtag=10, recvbuf=top, source=rank - 1, recvtag=11)
        if rank < size - 1:
            comm.Sendrecv(local[-1:], dest=rank + 1, sendtag=11, recvbuf=bottom, source=rank + 1, recvtag=10)
        local_with_halo = np.vstack([top, local, bottom])
        local = local_step(local_with_halo, local_intensity, args.seed + t * 100 + rank)
    elapsed = time.perf_counter() - start

    if rank == 0:
        final = np.empty((args.grid, args.grid), dtype=np.uint8)
    else:
        final = None
    comm.Gatherv(local, [final, sendcounts, displs, MPI.UNSIGNED_CHAR], root=0)
    if rank == 0:
        print(
            {
                "grid": args.grid,
                "steps": args.steps,
                "processes": size,
                "time_s": elapsed,
                "burned_cells": int(np.sum(final == 3)),
                "decomposition": "1-D row slabs with top/bottom halo exchange",
            }
        )


if __name__ == "__main__":
    main()
