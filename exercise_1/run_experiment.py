from pathlib import Path

from matrix_mult import benchmark


if __name__ == "__main__":
    benchmark(Path(__file__).resolve().parent / "results")
