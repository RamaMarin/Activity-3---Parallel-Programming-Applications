"""Matrix multiplication strategies for Exercise 1."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tarfile
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 20260422


def serial_classical(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Educational O(n^3) baseline used for small correctness checks."""
    m, n = a.shape
    n2, p = b.shape
    if n != n2:
        raise ValueError("Incompatible matrix shapes")
    c = np.zeros((m, p), dtype=np.float64)
    for i in range(m):
        for k in range(n):
            aik = a[i, k]
            for j in range(p):
                c[i, j] += aik * b[k, j]
    return c


def serial_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Practical serial baseline using NumPy's BLAS-backed matrix product."""
    return a @ b


def _row_worker(args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    a_block, b = args
    return a_block @ b


def parallel_rows(a: np.ndarray, b: np.ndarray, workers: int) -> np.ndarray:
    chunks = [chunk for chunk in np.array_split(a, workers, axis=0) if len(chunk)]
    with get_context("spawn").Pool(processes=workers) as pool:
        parts = pool.map(_row_worker, [(chunk, b) for chunk in chunks])
    return np.vstack(parts)


def _column_worker(args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    a, b_block = args
    return a @ b_block


def parallel_columns(a: np.ndarray, b: np.ndarray, workers: int) -> np.ndarray:
    chunks = [chunk for chunk in np.array_split(b, workers, axis=1) if chunk.shape[1]]
    with get_context("spawn").Pool(processes=workers) as pool:
        parts = pool.map(_column_worker, [(a, chunk) for chunk in chunks])
    return np.hstack(parts)


def _block_worker(args: tuple[int, int, np.ndarray, np.ndarray]) -> tuple[int, int, np.ndarray]:
    row_id, col_id, a_block, b_block = args
    return row_id, col_id, a_block @ b_block


def parallel_blocks(a: np.ndarray, b: np.ndarray, workers: int) -> np.ndarray:
    """2-D decomposition: each task computes one output block C_ij."""
    row_blocks = [chunk for chunk in np.array_split(a, workers, axis=0) if len(chunk)]
    col_blocks = [chunk for chunk in np.array_split(b, workers, axis=1) if chunk.shape[1]]
    tasks = []
    for row_id, a_block in enumerate(row_blocks):
        for col_id, b_block in enumerate(col_blocks):
            tasks.append((row_id, col_id, a_block, b_block))
    with get_context("spawn").Pool(processes=workers) as pool:
        results = pool.map(_block_worker, tasks)
    rows: list[list[np.ndarray | None]] = [
        [None for _ in range(len(col_blocks))] for _ in range(len(row_blocks))
    ]
    for row_id, col_id, block in results:
        rows[row_id][col_id] = block
    return np.vstack([np.hstack(row) for row in rows])


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def _strassen_square(a: np.ndarray, b: np.ndarray, cutoff: int) -> np.ndarray:
    n = a.shape[0]
    if n <= cutoff:
        return a @ b
    mid = n // 2
    a11, a12 = a[:mid, :mid], a[:mid, mid:]
    a21, a22 = a[mid:, :mid], a[mid:, mid:]
    b11, b12 = b[:mid, :mid], b[:mid, mid:]
    b21, b22 = b[mid:, :mid], b[mid:, mid:]

    p1 = _strassen_square(a11 + a22, b11 + b22, cutoff)
    p2 = _strassen_square(a21 + a22, b11, cutoff)
    p3 = _strassen_square(a11, b12 - b22, cutoff)
    p4 = _strassen_square(a22, b21 - b11, cutoff)
    p5 = _strassen_square(a11 + a12, b22, cutoff)
    p6 = _strassen_square(a21 - a11, b11 + b12, cutoff)
    p7 = _strassen_square(a12 - a22, b21 + b22, cutoff)

    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6
    return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))


def strassen(a: np.ndarray, b: np.ndarray, cutoff: int = 64) -> np.ndarray:
    """Hybrid Strassen implementation for square dense matrices."""
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    size = max(a.shape + b.shape)
    padded = _next_power_of_two(size)
    ap = np.zeros((padded, padded), dtype=np.float64)
    bp = np.zeros((padded, padded), dtype=np.float64)
    ap[: a.shape[0], : a.shape[1]] = a
    bp[: b.shape[0], : b.shape[1]] = b
    cp = _strassen_square(ap, bp, cutoff)
    return cp[: a.shape[0], : b.shape[1]]


def generate_dense(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + n)
    return rng.normal(size=(n, n)), rng.normal(size=(n, n))


def generate_sparse_rows(
    rows: int, cols: int, density: float, seed: int
) -> list[dict[int, float]]:
    rng = np.random.default_rng(seed)
    sparse: list[dict[int, float]] = []
    nnz_per_row = max(1, int(cols * density))
    for _ in range(rows):
        indices = rng.choice(cols, size=nnz_per_row, replace=False)
        values = rng.normal(size=nnz_per_row)
        sparse.append({int(i): float(v) for i, v in zip(indices, values)})
    return sparse


def sparse_to_dense(rows: list[dict[int, float]], cols: int) -> np.ndarray:
    out = np.zeros((len(rows), cols), dtype=np.float64)
    for r, row in enumerate(rows):
        for c, value in row.items():
            out[r, c] = value
    return out


def sparse_rows_matmul(
    a_rows: list[dict[int, float]], b_rows: list[dict[int, float]], p: int
) -> np.ndarray:
    c = np.zeros((len(a_rows), p), dtype=np.float64)
    for i, row in enumerate(a_rows):
        for k, a_value in row.items():
            for j, b_value in b_rows[k].items():
                c[i, j] += a_value * b_value
    return c


def _sparse_worker(args: tuple[list[dict[int, float]], list[dict[int, float]], int]) -> np.ndarray:
    return sparse_rows_matmul(*args)


def sparse_parallel_rows(
    a_rows: list[dict[int, float]], b_rows: list[dict[int, float]], p: int, workers: int
) -> np.ndarray:
    chunks = [
        list(chunk)
        for chunk in np.array_split(np.array(a_rows, dtype=object), workers)
        if len(chunk)
    ]
    with get_context("spawn").Pool(processes=workers) as pool:
        parts = pool.map(_sparse_worker, [(chunk, b_rows, p) for chunk in chunks])
    return np.vstack(parts)


def read_matrix_market_archive(path: Path, max_size: int | None = 700) -> tuple[list[dict[int, float]], int, dict]:
    """Read a real SuiteSparse Matrix Market archive into row dictionaries.

    The optional max_size cap keeps the benchmark workstation-friendly while
    preserving the original nonzero pattern in the leading principal submatrix.
    """
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name.endswith(".mtx")]
        if not members:
            raise ValueError(f"No .mtx file found in {path}")
        member = members[0]
        raw = archive.extractfile(member)
        if raw is None:
            raise ValueError(f"Could not read {member.name}")
        first = raw.readline().decode("utf-8", errors="replace").strip().lower()
        symmetric = "symmetric" in first
        dimensions = None
        entries: list[tuple[int, int, float]] = []
        for line_bytes in raw:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if dimensions is None:
                dimensions = tuple(map(int, parts[:3]))
                continue
            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
            value = float(parts[2]) if len(parts) >= 3 else 1.0
            entries.append((row, col, value))
    if dimensions is None:
        raise ValueError(f"Missing Matrix Market dimensions in {path}")
    rows, cols, nnz = dimensions
    cap = min(rows, cols, max_size or max(rows, cols))
    sparse_rows: list[dict[int, float]] = [dict() for _ in range(cap)]
    kept = 0
    for row, col, value in entries:
        if row < cap and col < cap:
            sparse_rows[row][col] = sparse_rows[row].get(col, 0.0) + value
            kept += 1
        if symmetric and row != col and row < cap and col < cap:
            sparse_rows[col][row] = sparse_rows[col].get(row, 0.0) + value
            kept += 1
    meta = {
        "archive": path.name,
        "matrix_market_member": member.name,
        "original_rows": rows,
        "original_cols": cols,
        "original_nnz": nnz,
        "benchmark_size": cap,
        "benchmark_nnz": kept,
        "symmetric_header": symmetric,
    }
    return sparse_rows, cap, meta


def sparse_benchmark_matrices() -> list[tuple[str, list[dict[int, float]], int, dict]]:
    data_dir = Path(__file__).resolve().parent / "data" / "suitesparse"
    loaded = []
    preferred = ["plat362.tar.gz", "bcsstk13.tar.gz", "west0479.tar.gz"]
    paths = []
    for name in preferred:
        path = data_dir / name
        if path.exists():
            paths.append(path)
    for path in sorted(data_dir.glob("*.tar.gz")):
        if path not in paths:
            paths.append(path)
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            rows, size, meta = read_matrix_market_archive(path)
            loaded.append((f"SuiteSparse_{path.stem.replace('.tar', '')}", rows, size, meta))
    if loaded:
        return loaded

    fallback_specs = [
        ("SuiteSparse_HB_west0479_style", 192, 192, 0.018, 3101),
        ("SuiteSparse_HB_bcsstk13_style", 192, 192, 0.035, 3102),
    ]
    for name, m, n, density, seed in fallback_specs:
        rows = generate_sparse_rows(m, n, density, seed)
        loaded.append(
            (
                name,
                rows,
                m,
                {
                    "source": "deterministic fallback generator",
                    "density": density,
                    "seed": seed,
                },
            )
        )
    return loaded


def timed(label: str, func, *args):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    return label, elapsed, result


def benchmark(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    dense_sizes = [64, 128, 192]
    workers_list = [2, 4]

    for n in dense_sizes:
        a, b = generate_dense(n)
        _, serial_time, reference = timed("serial_numpy", serial_numpy, a, b)
        records.append(
            {
                "dataset": "dense_synthetic",
                "size": n,
                "method": "serial_numpy",
                "workers": 1,
                "time_s": serial_time,
                "speedup": 1.0,
                "efficiency": 1.0,
                "max_error": 0.0,
            }
        )
        _, strassen_time, strassen_result = timed("strassen_hybrid", strassen, a, b, 64)
        records.append(
            {
                "dataset": "dense_synthetic",
                "size": n,
                "method": "strassen_hybrid",
                "workers": 1,
                "time_s": strassen_time,
                "speedup": serial_time / strassen_time,
                "efficiency": serial_time / strassen_time,
                "max_error": float(np.max(np.abs(reference - strassen_result))),
            }
        )
        for workers in workers_list:
            for method, func in [
                ("parallel_rows", parallel_rows),
                ("parallel_columns", parallel_columns),
                ("parallel_blocks_2d", parallel_blocks),
            ]:
                _, elapsed, result = timed(method, func, a, b, workers)
                records.append(
                    {
                        "dataset": "dense_synthetic",
                        "size": n,
                        "method": method,
                        "workers": workers,
                        "time_s": elapsed,
                        "speedup": serial_time / elapsed,
                        "efficiency": (serial_time / elapsed) / workers,
                        "max_error": float(np.max(np.abs(reference - result))),
                    }
                )

    sparse_metadata = []
    for name, a_sparse, m, meta in sparse_benchmark_matrices():
        b_sparse = a_sparse
        sparse_metadata.append({"dataset": name, **meta})
        _, serial_time, reference = timed("sparse_serial_rows", sparse_rows_matmul, a_sparse, b_sparse, m)
        records.append(
            {
                "dataset": name,
                "size": m,
                "method": "sparse_serial_rows",
                "workers": 1,
                "time_s": serial_time,
                "speedup": 1.0,
                "efficiency": 1.0,
                "max_error": 0.0,
            }
        )
        for workers in workers_list:
            _, elapsed, result = timed(
                "sparse_parallel_rows", sparse_parallel_rows, a_sparse, b_sparse, m, workers
            )
            records.append(
                {
                    "dataset": name,
                    "size": m,
                    "method": "sparse_parallel_rows",
                    "workers": workers,
                    "time_s": elapsed,
                    "speedup": serial_time / elapsed,
                    "efficiency": (serial_time / elapsed) / workers,
                    "max_error": float(np.max(np.abs(reference - result))),
                }
            )

    small_a, small_b = generate_dense(8)
    classical = serial_classical(small_a, small_b)
    validation = {
        "classical_vs_numpy_max_error": float(np.max(np.abs(classical - (small_a @ small_b)))),
        "strassen_vs_numpy_max_error": float(np.max(np.abs(strassen(small_a, small_b, 2) - (small_a @ small_b)))),
    }
    pd.DataFrame.from_records(records).to_csv(output_dir / "benchmark_results.csv", index=False)
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output_dir / "sparse_metadata.json").write_text(json.dumps(sparse_metadata, indent=2), encoding="utf-8")
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
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
