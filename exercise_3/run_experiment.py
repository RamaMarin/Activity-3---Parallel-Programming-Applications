from pathlib import Path

from fire_ca import benchmark


if __name__ == "__main__":
    benchmark(Path(__file__).resolve().parent / "results")
