from pathlib import Path

from image_pipeline import benchmark


if __name__ == "__main__":
    benchmark(Path(__file__).resolve().parent / "results")
