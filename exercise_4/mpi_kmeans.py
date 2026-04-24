"""MPI K-means using collective aggregation of partial cluster statistics.

Run with:
    mpiexec -n 4 python exercise_4/mpi_kmeans.py --samples 50000 --clusters 7
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mpi4py is required for this script. Install mpi4py and an MPI runtime.") from exc


def load_real_covtype(path: Path, samples: int, features: int) -> np.ndarray:
    frame = pd.read_csv(path, header=None, nrows=samples)
    x = frame.iloc[:, :features].to_numpy(dtype=np.float64)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x


def counts_displs(n: int, size: int) -> tuple[list[int], list[int]]:
    counts = [n // size + (1 if r < n % size else 0) for r in range(size)]
    displs = [0]
    for count in counts[:-1]:
        displs.append(displs[-1] + count)
    return counts, displs


def assign(x: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, distances[np.arange(x.shape[0]), labels]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--features", type=int, default=54)
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent / "data" / "covtype.data.gz"),
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    row_counts, row_displs = counts_displs(args.samples, size)
    sendcounts = [rows * args.features for rows in row_counts]
    displs = [row * args.features for row in row_displs]

    if rank == 0:
        x = load_real_covtype(Path(args.data), args.samples, args.features)
        rng = np.random.default_rng(args.seed + args.clusters)
        centroids = x[rng.choice(args.samples, args.clusters, replace=False)].copy()
    else:
        x = None
        centroids = np.empty((args.clusters, args.features), dtype=np.float64)

    local = np.empty((row_counts[rank], args.features), dtype=np.float64)
    comm.Scatterv([x, sendcounts, displs, MPI.DOUBLE], local, root=0)
    comm.Bcast(centroids, root=0)

    comm.Barrier()
    start = time.perf_counter()
    inertia = 0.0
    for iteration in range(args.max_iter):
        labels, distances = assign(local, centroids)
        local_sums = np.zeros((args.clusters, args.features), dtype=np.float64)
        local_counts = np.bincount(labels, minlength=args.clusters).astype(np.float64)
        for cluster in range(args.clusters):
            if local_counts[cluster]:
                local_sums[cluster] = local[labels == cluster].sum(axis=0)
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        old = centroids.copy()
        for cluster in range(args.clusters):
            if global_counts[cluster]:
                centroids[cluster] = global_sums[cluster] / global_counts[cluster]
        local_inertia = np.array(float(distances.sum()))
        inertia_arr = np.array(0.0)
        comm.Allreduce(local_inertia, inertia_arr, op=MPI.SUM)
        inertia = float(inertia_arr)
        if np.linalg.norm(centroids - old) < 1e-4:
            break
    elapsed = time.perf_counter() - start

    if rank == 0:
        print(
            {
                "samples": args.samples,
                "features": args.features,
                "clusters": args.clusters,
                "processes": size,
                "iterations": iteration + 1,
                "time_s": elapsed,
                "inertia": inertia,
                "communication": "Allreduce of kxd sums and k counts each iteration",
            }
        )


if __name__ == "__main__":
    main()
