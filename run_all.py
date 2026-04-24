"""Run all reproducible local experiments and generate docs/report.pdf."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: Path) -> None:
    print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main() -> None:
    scripts = [
        ROOT / "exercise_1" / "run_experiment.py",
        ROOT / "exercise_2" / "run_experiment.py",
        ROOT / "exercise_3" / "run_experiment.py",
        ROOT / "exercise_4" / "run_experiment.py",
        ROOT / "docs" / "generate_report.py",
    ]
    for script in scripts:
        run(script)
    print("\nAll outputs generated. Open docs/report.pdf for the report.")


if __name__ == "__main__":
    main()
