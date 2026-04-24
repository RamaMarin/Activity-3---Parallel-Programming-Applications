"""Cell image processing and morphology for Exercise 2."""

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
from PIL import Image, ImageDraw


SEED = 20260422


def dataset_pairs(data_root: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for sequence in ["01", "02"]:
        image_dir = data_root / sequence
        mask_dir = data_root / f"{sequence}_ST" / "SEG"
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("t*.tif")):
            frame = image_path.stem[1:]
            mask_path = mask_dir / f"man_seg{frame}.tif"
            if mask_path.exists():
                pairs.append((sequence, image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(
            "Real DIC-C2DH-HeLa data not found. Expected extracted files under exercise_2/data/DIC-C2DH-HeLa."
        )
    return pairs


def labeled_components(mask: np.ndarray, min_area: int = 20) -> list[tuple[int, np.ndarray]]:
    components = []
    for label in np.unique(mask):
        if label == 0:
            continue
        coords = np.argwhere(mask == label).astype(np.float64)
        if len(coords) >= min_area:
            components.append((int(label), coords))
    return components


def component_measurements(coords: np.ndarray, object_id: int) -> dict[str, float | int]:
    y = coords[:, 0]
    x = coords[:, 1]
    min_y, max_y = int(y.min()), int(y.max())
    min_x, max_x = int(x.min()), int(x.max())
    centered = np.column_stack((x - x.mean(), y - y.mean()))
    if len(coords) > 2:
        cov = np.cov(centered, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        minor, major = np.sqrt(eigvals) * 4.0
    else:
        major = minor = 0.0
    return {
        "object_id": object_id,
        "bbox_min_x": min_x,
        "bbox_min_y": min_y,
        "bbox_max_x": max_x,
        "bbox_max_y": max_y,
        "bbox_width": max_x - min_x + 1,
        "bbox_height": max_y - min_y + 1,
        "area_px": int(len(coords)),
        "major_axis_px": float(major),
        "minor_axis_px": float(minor),
    }


def process_image(
    image_path: str | Path,
    mask_path: str | Path,
    sequence: str,
    overlay_dir: str | Path | None = None,
) -> tuple[list[dict], dict]:
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    image = np.asarray(Image.open(image_path).convert("L"))
    mask = np.asarray(Image.open(mask_path))
    components = labeled_components(mask)
    rows = []
    overlay = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for object_id, (label_id, coords) in enumerate(components, start=1):
        record = component_measurements(coords, object_id)
        record["sequence"] = sequence
        record["mask_label"] = label_id
        record["image"] = image_path.name
        record["mask"] = mask_path.name
        rows.append(record)
        draw.rectangle(
            [
                record["bbox_min_x"],
                record["bbox_min_y"],
                record["bbox_max_x"],
                record["bbox_max_y"],
            ],
            outline=(255, 40, 40),
            width=1,
        )
    if overlay_dir is not None:
        overlay_path = Path(overlay_dir)
        overlay_path.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path / f"{sequence}_{image_path.stem}_overlay.png")
    if rows:
        frame = pd.DataFrame(rows)
        summary = {
            "sequence": sequence,
            "image": image_path.name,
            "detected_cells": len(rows),
            "avg_width_px": float(frame["bbox_width"].mean()),
            "std_width_px": float(frame["bbox_width"].std(ddof=0)),
            "avg_length_px": float(frame["major_axis_px"].mean()),
            "std_length_px": float(frame["major_axis_px"].std(ddof=0)),
        }
    else:
        summary = {
            "sequence": sequence,
            "image": image_path.name,
            "detected_cells": 0,
            "avg_width_px": 0.0,
            "std_width_px": 0.0,
            "avg_length_px": 0.0,
            "std_length_px": 0.0,
        }
    return rows, summary


def run_serial(pairs: list[tuple[str, Path, Path]], overlay_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    objects = []
    summaries = []
    for sequence, image_path, mask_path in pairs:
        rows, summary = process_image(image_path, mask_path, sequence, overlay_dir)
        objects.extend(rows)
        summaries.append(summary)
    return pd.DataFrame(objects), pd.DataFrame(summaries)


def _process_for_pool(args: tuple[str, str, str, str | None]) -> tuple[list[dict], dict]:
    return process_image(args[0], args[1], args[2], args[3])


def run_parallel(
    pairs: list[tuple[str, Path, Path]], workers: int, overlay_dir: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlay_text = str(overlay_dir) if overlay_dir else None
    with get_context("spawn").Pool(processes=workers) as pool:
        results = pool.map(
            _process_for_pool,
            [(str(image_path), str(mask_path), sequence, overlay_text) for sequence, image_path, mask_path in pairs],
        )
    objects = []
    summaries = []
    for rows, summary in results:
        objects.extend(rows)
        summaries.append(summary)
    return pd.DataFrame(objects), pd.DataFrame(summaries)


def benchmark(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(__file__).resolve().parent / "data" / "DIC-C2DH-HeLa"
    pairs = dataset_pairs(data_root)

    timings = []
    start = time.perf_counter()
    objects, summary = run_serial(pairs, output_dir / "overlays_serial")
    serial_time = time.perf_counter() - start
    objects.to_csv(output_dir / "objects_serial.csv", index=False)
    summary.to_csv(output_dir / "summary_serial.csv", index=False)
    timings.append(
        {
            "dataset": "DIC-C2DH-HeLa_ST_masks",
            "images": len(pairs),
            "method": "serial_morphology",
            "workers": 1,
            "time_s": serial_time,
            "speedup": 1.0,
            "efficiency": 1.0,
            "detected_cells": int(len(objects)),
        }
    )

    for workers in [2, 4]:
        start = time.perf_counter()
        p_objects, p_summary = run_parallel(pairs, workers, output_dir / f"overlays_parallel_{workers}")
        elapsed = time.perf_counter() - start
        p_objects.to_csv(output_dir / f"objects_parallel_{workers}.csv", index=False)
        p_summary.to_csv(output_dir / f"summary_parallel_{workers}.csv", index=False)
        timings.append(
            {
                "dataset": "DIC-C2DH-HeLa_ST_masks",
                "images": len(pairs),
                "method": "parallel_by_image",
                "workers": workers,
                "time_s": elapsed,
                "speedup": serial_time / elapsed,
                "efficiency": (serial_time / elapsed) / workers,
                "detected_cells": int(len(p_objects)),
            }
        )

    pd.DataFrame(timings).to_csv(output_dir / "benchmark_results.csv", index=False)
    sequence_summary = summary.groupby("sequence", as_index=False).agg(
        images=("image", "count"),
        detected_cells=("detected_cells", "sum"),
        avg_cells_per_image=("detected_cells", "mean"),
        avg_width_px=("avg_width_px", "mean"),
        avg_length_px=("avg_length_px", "mean"),
    )
    sequence_summary.to_csv(output_dir / "summary_by_sequence.csv", index=False)
    sample_image = np.asarray(Image.open(pairs[0][1]).convert("L"))
    metadata = {
        "source": "Real DIC-C2DH-HeLa Cell Tracking Challenge frames with silver-truth masks from *_ST/SEG.",
        "image_format": "8-bit grayscale TIFF",
        "image_size_px": list(sample_image.shape[::-1]),
        "sequences": sorted({sequence for sequence, _, _ in pairs}),
        "annotated_frames": len(pairs),
        "segmentation_source": "Provided silver-truth labeled masks",
        "measurements": "pixels",
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
