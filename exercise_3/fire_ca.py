"""Forest-fire cellular automaton driven by real MODIS Mexico 2024 hotspots."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


SEED = 20260422
COLORS = np.array(
    [
        [35, 35, 40],      # non-burnable/outside
        [42, 120, 62],     # vegetation
        [245, 116, 32],    # burning
        [65, 56, 48],      # burned
    ],
    dtype=np.uint8,
)


def load_hotspots(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Expected real MODIS CSV at {data_path}")
    hotspots = pd.read_csv(data_path, parse_dates=["acq_date"])
    hotspots["confidence"] = pd.to_numeric(hotspots["confidence"], errors="coerce")
    hotspots["frp"] = pd.to_numeric(hotspots["frp"], errors="coerce")
    hotspots = hotspots.dropna(subset=["latitude", "longitude", "frp", "confidence", "acq_date"])
    return hotspots


def filter_hotspots(
    hotspots: pd.DataFrame,
    start_date: str = "2024-03-01",
    end_date: str = "2024-05-31",
    min_confidence: int = 70,
) -> pd.DataFrame:
    filtered = hotspots[
        (hotspots["acq_date"] >= start_date)
        & (hotspots["acq_date"] <= end_date)
        & (hotspots["confidence"] >= min_confidence)
    ].copy()
    filtered = filtered.sort_values(["acq_date", "frp"], ascending=[True, False]).reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No hotspots available after filtering")
    return filtered


def region_from_hotspots(hotspots: pd.DataFrame) -> dict[str, float]:
    return {
        "lat_min": float(hotspots["latitude"].min()),
        "lat_max": float(hotspots["latitude"].max()),
        "lon_min": float(hotspots["longitude"].min()),
        "lon_max": float(hotspots["longitude"].max()),
    }


def build_schedule(hotspots: pd.DataFrame, grid_size: int, steps: int) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    region = region_from_hotspots(hotspots)
    lat_norm = (hotspots["latitude"] - region["lat_min"]) / (region["lat_max"] - region["lat_min"])
    lon_norm = (hotspots["longitude"] - region["lon_min"]) / (region["lon_max"] - region["lon_min"])
    rows = np.clip(((1 - lat_norm) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    cols = np.clip((lon_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)
    day_index = (hotspots["acq_date"] - hotspots["acq_date"].min()).dt.days.to_numpy()
    horizon_days = max(int(day_index.max()) + 1, 1)
    step_ids = np.clip((day_index * steps) // horizon_days, 0, steps - 1)
    frp = hotspots["frp"].to_numpy(dtype=np.float32)
    max_frp = max(float(frp.max()), 1.0)
    ignite_schedule = [np.zeros((grid_size, grid_size), dtype=bool) for _ in range(steps)]
    intensity_schedule = [np.zeros((grid_size, grid_size), dtype=np.float32) for _ in range(steps)]
    for step_id, row, col, frp_value in zip(step_ids, rows, cols, frp):
        ignite_schedule[int(step_id)][int(row), int(col)] = True
        intensity_schedule[int(step_id)][int(row), int(col)] = max(
            intensity_schedule[int(step_id)][int(row), int(col)],
            float(frp_value / max_frp),
        )
    meta = {
        "region": region,
        "date_min": str(hotspots["acq_date"].min().date()),
        "date_max": str(hotspots["acq_date"].max().date()),
        "hotspots": int(len(hotspots)),
        "max_frp": max_frp,
    }
    return ignite_schedule, intensity_schedule, meta


def initial_state(grid_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + grid_size)
    state = np.ones((grid_size, grid_size), dtype=np.uint8)
    non_burnable = rng.random((grid_size, grid_size)) < 0.08
    state[non_burnable] = 0
    return state


def burning_neighbors(state: np.ndarray) -> np.ndarray:
    burning = state == 2
    padded = np.pad(burning, 1, mode="constant", constant_values=False)
    total = np.zeros_like(state, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += padded[dy : dy + state.shape[0], dx : dx + state.shape[1]]
    return total


def step_with_random(
    state: np.ndarray,
    local_intensity: np.ndarray,
    external_ignite: np.ndarray,
    random_grid: np.ndarray,
) -> np.ndarray:
    next_state = state.copy()
    neighbors = burning_neighbors(state)
    susceptible = state == 1
    probability = np.minimum(0.04 + 0.14 * neighbors + 0.25 * local_intensity, 0.95)
    ignites = susceptible & (((neighbors > 0) & (random_grid < probability)) | external_ignite)
    next_state[state == 2] = 3
    next_state[ignites] = 2
    return next_state


def step(state: np.ndarray, intensity: np.ndarray, external_ignite: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return step_with_random(state, intensity, external_ignite, rng.random(state.shape))


def simulate_serial(
    grid_size: int,
    steps: int,
    ignite_schedule: list[np.ndarray],
    intensity_schedule: list[np.ndarray],
    seed: int = SEED,
) -> tuple[np.ndarray, list[np.ndarray]]:
    state = initial_state(grid_size, seed)
    rng = np.random.default_rng(seed + 11)
    state[ignite_schedule[0] & (state == 1)] = 2
    snapshots = [state.copy()]
    for t in range(steps):
        state = step(state, intensity_schedule[t], ignite_schedule[t], rng)
        if t in {steps // 4, steps // 2, steps - 1}:
            snapshots.append(state.copy())
    return state, snapshots


def _step_block(args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    block_with_halo, intensity_block, ignite_block, random_block = args
    updated = step_with_random(
        block_with_halo,
        np.pad(intensity_block, ((1, 1), (0, 0)), mode="edge"),
        np.pad(ignite_block, ((1, 1), (0, 0)), mode="constant", constant_values=False),
        np.pad(random_block, ((1, 1), (0, 0)), mode="edge"),
    )
    return updated[1:-1, :]


def simulate_parallel_mp(
    grid_size: int,
    steps: int,
    ignite_schedule: list[np.ndarray],
    intensity_schedule: list[np.ndarray],
    workers: int,
    seed: int = SEED,
) -> np.ndarray:
    state = initial_state(grid_size, seed)
    state[ignite_schedule[0] & (state == 1)] = 2
    row_edges = np.linspace(0, grid_size, workers + 1, dtype=int)
    rng = np.random.default_rng(seed + 11)
    with get_context("spawn").Pool(processes=workers) as pool:
        for t in range(steps):
            random_grid = rng.random(state.shape)
            tasks = []
            for part in range(workers):
                start, end = row_edges[part], row_edges[part + 1]
                halo_start = max(0, start - 1)
                halo_end = min(grid_size, end + 1)
                block = state[halo_start:halo_end]
                if start == 0:
                    block = np.vstack([np.zeros((1, grid_size), dtype=state.dtype), block])
                if end == grid_size:
                    block = np.vstack([block, np.zeros((1, grid_size), dtype=state.dtype)])
                tasks.append(
                    (
                        block,
                        intensity_schedule[t][start:end],
                        ignite_schedule[t][start:end],
                        random_grid[start:end],
                    )
                )
            parts = pool.map(_step_block, tasks)
            state = np.vstack(parts)
    return state


def save_snapshot(state: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = COLORS[state]
    Image.fromarray(img, mode="RGB").resize((512, 512), resample=Image.Resampling.NEAREST).save(path)


def benchmark(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hotspots = load_hotspots(Path(__file__).resolve().parent / "data" / "firms_modis_2024_mexico.csv")
    filtered = filter_hotspots(hotspots)
    timings = []
    schedule_meta = None
    for grid_size, steps in [(240, 60), (420, 90)]:
        ignite_schedule, intensity_schedule, schedule_meta = build_schedule(filtered, grid_size, steps)
        start = time.perf_counter()
        final_state, snapshots = simulate_serial(grid_size, steps, ignite_schedule, intensity_schedule)
        serial_time = time.perf_counter() - start
        burned = int(np.sum(final_state == 3))
        timings.append(
            {
                "dataset": "MODIS_2024_Mexico_filtered",
                "grid_size": grid_size,
                "steps": steps,
                "method": "serial",
                "workers": 1,
                "time_s": serial_time,
                "speedup": 1.0,
                "efficiency": 1.0,
                "burned_cells": burned,
            }
        )
        if grid_size == 240:
            for idx, snapshot in enumerate(snapshots):
                save_snapshot(snapshot, output_dir / "snapshots" / f"fire_t{idx}.png")
        for workers in [2, 4]:
            start = time.perf_counter()
            parallel_final = simulate_parallel_mp(grid_size, steps, ignite_schedule, intensity_schedule, workers)
            elapsed = time.perf_counter() - start
            timings.append(
                {
                    "dataset": "MODIS_2024_Mexico_filtered",
                    "grid_size": grid_size,
                    "steps": steps,
                    "method": "parallel_domain_mp",
                    "workers": workers,
                    "time_s": elapsed,
                    "speedup": serial_time / elapsed,
                    "efficiency": (serial_time / elapsed) / workers,
                    "burned_cells": int(np.sum(parallel_final == 3)),
                }
            )
    pd.DataFrame(timings).to_csv(output_dir / "benchmark_results.csv", index=False)
    metadata = {
        "source": "Real MODIS 2024 Mexico hotspot CSV provided by the user.",
        "total_hotspots": int(len(hotspots)),
        "filtered_hotspots": int(len(filtered)),
        "filtering": {
            "start_date": "2024-03-01",
            "end_date": "2024-05-31",
            "min_confidence": 70,
        },
        "region": schedule_meta["region"] if schedule_meta else {},
        "date_min": schedule_meta["date_min"] if schedule_meta else None,
        "date_max": schedule_meta["date_max"] if schedule_meta else None,
        "states": {"0": "non-burnable", "1": "susceptible vegetation", "2": "burning", "3": "burned"},
        "neighborhood": "Moore 8-neighborhood",
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
