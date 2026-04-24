"""MPI row-decomposed dense matrix multiplication.

Run with:
    mpiexec -n 4 python exercise_1/mpi_matrix_mult.py --n 512
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mpi4py is required for this script. Install mpi4py and an MPI runtime.") from exc


def split_counts(n: int, size: int) -> tuple[list[int], list[int]]:
    counts = [n // size + (1 if rank < n % size else 0) for rank in range(size)]
    displacements = [0]
    for count in counts[:-1]:
        displacements.append(displacements[-1] + count)
    return counts, displacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260422)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rng = np.random.default_rng(args.seed)
        a = rng.normal(size=(args.n, args.n))
        b = rng.normal(size=(args.n, args.n))
    else:
        a = None
        b = np.empty((args.n, args.n), dtype=np.float64)

    row_counts, row_displs = split_counts(args.n, size)
    sendcounts = [rows * args.n for rows in row_counts]
    displs = [row * args.n for row in row_displs]
    local_a = np.empty((row_counts[rank], args.n), dtype=np.float64)

    comm.Barrier()
    start = time.perf_counter()
    comm.Scatterv([a, sendcounts, displs, MPI.DOUBLE], local_a, root=0)
    comm.Bcast(b, root=0)
    local_c = local_a @ b
    if rank == 0:
        c = np.empty((args.n, args.n), dtype=np.float64)
    else:
        c = None
    comm.Gatherv(local_c, [c, sendcounts, displs, MPI.DOUBLE], root=0)
    elapsed = time.perf_counter() - start

    if rank == 0:
        print(
            {
                "n": args.n,
                "processes": size,
                "time_s": elapsed,
                "distribution": "row blocks with Scatterv/Bcast/Gatherv",
                "checksum": float(np.sum(c)),
            }
        )


if __name__ == "__main__":
    main()
