"""Serial, multiprocessing, and MPI-ready K-means for Exercise 4."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
from multiprocessing import get_context, shared_memory
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 20260422
_SHM = None
_SHAPE = None
_DTYPE = None


def count_gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt") as fh:
        return sum(1 for _ in fh)


def load_covtype_real(
    data_dir: Path, benchmark_samples: int = 150000, features: int = 54
) -> tuple[np.ndarray, dict]:
    gz_path = data_dir / "covtype.data.gz"
    info_path = data_dir / "covtype.info"
    if not gz_path.exists():
        raise FileNotFoundError(f"Expected real Covertype data at {gz_path}")
    frame = pd.read_csv(gz_path, header=None, nrows=benchmark_samples)
    x = frame.iloc[:, :features].to_numpy(dtype=np.float32)
    y = frame.iloc[:, features].to_numpy(dtype=np.int32)
    meta = {
        "benchmark_samples": int(len(x)),
        "features": int(features),
        "label_min": int(y.min()),
        "label_max": int(y.max()),
        "full_dataset_rows": count_gzip_rows(gz_path),
        "info_excerpt": info_path.read_text(encoding="utf-8", errors="replace").splitlines()[:8]
        if info_path.exists()
        else [],
    }
    return x, meta


def standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return ((x - mean) / scale).astype(np.float32)


def init_centroids(x: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed + k)
    indices = rng.choice(x.shape[0], size=k, replace=False)
    return x[indices].copy()


def assign(x: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(x.shape[0]), labels]
    return labels, min_distances


def serial_kmeans(
    x: np.ndarray, k: int, max_iter: int = 10, tol: float = 1e-4, seed: int = SEED
) -> dict:
    centroids = init_centroids(x, k, seed)
    history = []
    total_start = time.perf_counter()
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for iteration in range(max_iter):
        iter_start = time.perf_counter()
        labels, distances = assign(x, centroids)
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        for cluster in range(k):
            if counts[cluster]:
                new_centroids[cluster] = x[labels == cluster].mean(axis=0)
            else:
                new_centroids[cluster] = centroids[cluster]
        shift = float(np.linalg.norm(new_centroids - centroids))
        inertia = float(distances.sum())
        centroids = new_centroids
        elapsed = time.perf_counter() - iter_start
        history.append({"iteration": iteration, "time_s": elapsed, "shift": shift, "inertia": inertia})
        if shift < tol:
            break
    return {
        "centroids": centroids,
        "labels": labels,
        "history": pd.DataFrame(history),
        "total_time_s": time.perf_counter() - total_start,
        "inertia": history[-1]["inertia"],
        "iterations": len(history),
    }


def _init_shared(name: str, shape: tuple[int, int], dtype_str: str) -> None:
    global _SHM, _SHAPE, _DTYPE
    _SHM = shared_memory.SharedMemory(name=name)
    _SHAPE = shape
    _DTYPE = np.dtype(dtype_str)


def _partial_stats_shared(args: tuple[int, int, np.ndarray, int]) -> tuple[np.ndarray, np.ndarray, float]:
    start, end, centroids, chunk_size = args
    x = np.ndarray(_SHAPE, dtype=_DTYPE, buffer=_SHM.buf)
    block = x[start:end]
    k, features = centroids.shape
    sums = np.zeros((k, features), dtype=np.float64)
    counts = np.zeros(k, dtype=np.float64)
    inertia = 0.0
    for batch_start in range(0, len(block), chunk_size):
        batch = block[batch_start : batch_start + chunk_size]
        labels, distances = assign(batch, centroids)
        counts += np.bincount(labels, minlength=k)
        for cluster in range(k):
            mask = labels == cluster
            if np.any(mask):
                sums[cluster] += batch[mask].sum(axis=0)
        inertia += float(distances.sum())
    return sums, counts, inertia


def parallel_kmeans_mp(
    x: np.ndarray,
    k: int,
    workers: int,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int = SEED,
    chunk_size: int = 5000,
) -> dict:
    centroids = init_centroids(x, k, seed)
    shm = shared_memory.SharedMemory(create=True, size=x.nbytes)
    shm_array = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf)
    shm_array[:] = x
    edges = np.linspace(0, len(x), workers + 1, dtype=int)
    blocks = [(int(edges[i]), int(edges[i + 1]), centroids, chunk_size) for i in range(workers) if edges[i] < edges[i + 1]]
    history = []
    total_start = time.perf_counter()
    try:
        with get_context("spawn").Pool(
            processes=workers,
            initializer=_init_shared,
            initargs=(shm.name, x.shape, x.dtype.str),
        ) as pool:
            for iteration in range(max_iter):
                iter_start = time.perf_counter()
                tasks = [(start, end, centroids, chunk_size) for start, end, _, _ in blocks]
                partials = pool.map(_partial_stats_shared, tasks)
                sums = np.sum([item[0] for item in partials], axis=0)
                counts = np.sum([item[1] for item in partials], axis=0)
                inertia = float(np.sum([item[2] for item in partials]))
                new_centroids = centroids.copy()
                for cluster in range(k):
                    if counts[cluster]:
                        new_centroids[cluster] = sums[cluster] / counts[cluster]
                shift = float(np.linalg.norm(new_centroids - centroids))
                centroids = new_centroids.astype(np.float32)
                elapsed = time.perf_counter() - iter_start
                history.append({"iteration": iteration, "time_s": elapsed, "shift": shift, "inertia": inertia})
                if shift < tol:
                    break
        labels, _ = assign(x, centroids)
        return {
            "centroids": centroids,
            "labels": labels,
            "history": pd.DataFrame(history),
            "total_time_s": time.perf_counter() - total_start,
            "inertia": history[-1]["inertia"],
            "iterations": len(history),
        }
    finally:
        shm.close()
        shm.unlink()


def benchmark(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).resolve().parent / "data"
    raw_x, data_meta = load_covtype_real(data_dir, benchmark_samples=150000, features=54)
    x = standardize(raw_x)
    records = []
    histories = []
    for k in [4, 7]:
        serial = serial_kmeans(x, k, max_iter=6)
        serial["history"].assign(method="serial", workers=1, clusters=k).to_csv(
            output_dir / f"history_serial_k{k}.csv", index=False
        )
        histories.append(serial["history"].assign(method="serial", workers=1, clusters=k))
        records.append(
            {
                "dataset": "Covertype_real_subset",
                "samples": x.shape[0],
                "features": x.shape[1],
                "clusters": k,
                "method": "serial",
                "workers": 1,
                "iterations": serial["iterations"],
                "time_s": serial["total_time_s"],
                "runtime_per_iteration_s": serial["total_time_s"] / serial["iterations"],
                "inertia": serial["inertia"],
                "speedup": 1.0,
                "efficiency": 1.0,
            }
        )
        for workers in [2, 4]:
            parallel = parallel_kmeans_mp(x, k, workers, max_iter=6)
            parallel["history"].assign(method="parallel_partial_stats_mp", workers=workers, clusters=k).to_csv(
                output_dir / f"history_parallel_k{k}_w{workers}.csv", index=False
            )
            histories.append(
                parallel["history"].assign(method="parallel_partial_stats_mp", workers=workers, clusters=k)
            )
            records.append(
                {
                    "dataset": "Covertype_real_subset",
                    "samples": x.shape[0],
                    "features": x.shape[1],
                    "clusters": k,
                    "method": "parallel_partial_stats_mp",
                    "workers": workers,
                    "iterations": parallel["iterations"],
                    "time_s": parallel["total_time_s"],
                    "runtime_per_iteration_s": parallel["total_time_s"] / parallel["iterations"],
                    "inertia": parallel["inertia"],
                    "speedup": serial["total_time_s"] / parallel["total_time_s"],
                    "efficiency": (serial["total_time_s"] / parallel["total_time_s"]) / workers,
                }
            )
    pd.DataFrame(records).to_csv(output_dir / "benchmark_results.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output_dir / "iteration_history.csv", index=False)
    metadata = {
        "source": "Real Covertype dataset from exc4_covertype.zip using a deterministic benchmark subset.",
        "full_dataset_rows": data_meta["full_dataset_rows"],
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "cover_type_label_range": [data_meta["label_min"], data_meta["label_max"]],
        "preprocessing": "feature standardization to zero mean and unit variance",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": SEED,
        "pid": os.getpid(),
    }
    (output_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args(argv)
    benchmark(Path(args.output_dir))


if __name__ == "__main__":
    main()
