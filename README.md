# High Performance Computing Assignment

This repository contains four reproducible high performance computing exercises
covering matrix multiplication, image processing, cellular automata, and
distributed-style K-means clustering.

Each exercise includes a serial baseline and at least one parallel
implementation. The MPI scripts are included for the exercises that require
distributed-memory designs; the local benchmark runner uses Python
`multiprocessing` so the project can be reproduced on a standard workstation.

## Repository Structure

- `exercise_1/`: dense and sparse matrix multiplication, row/column/block
  multiprocessing decompositions, Strassen implementation, and MPI row
  decomposition script.
- `exercise_2/`: cell-image segmentation and morphological characterization
  pipeline with serial and multiprocessing image-level execution.
- `exercise_3/`: NASA FIRMS-inspired forest-fire cellular automaton with serial,
  multiprocessing domain decomposition, visualization snapshots, and an MPI
  halo-exchange script.
- `exercise_4/`: Covertype-style K-means clustering with serial,
  multiprocessing partial-statistics aggregation, and an MPI implementation.
- `docs/`: generated PDF report and report assets.
- `run_all.py`: runs the reproducible benchmark suite and builds the report.
- `requirements.txt`: Python package requirements for full reproduction.

## Software Requirements

Minimum local benchmark requirements:

- Python 3.10+
- `numpy`
- `pandas`
- `Pillow`
- `reportlab`

Full optional requirements:

- `mpi4py`
- An MPI runtime such as Microsoft MPI, MPICH, or Open MPI
- Optional scientific packages such as `scipy`, `scikit-image`, and `cellpose`
  if real microscopy segmentation is desired.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Reproduce All Local Experiments

From the repository root:

```bash
python run_all.py
```

This command uses the real datasets stored in the repository, executes serial
and parallel benchmarks, exports CSV logs and visual outputs, and writes:

```text
docs/report.pdf
```

The experiments fix random seeds and write their environment metadata next to
the results.

## Exercise Commands

Run each exercise independently:

```bash
python exercise_1/run_experiment.py
python exercise_2/run_experiment.py
python exercise_3/run_experiment.py
python exercise_4/run_experiment.py
python docs/generate_report.py
```

## MPI Commands

The MPI scripts require `mpi4py` and an MPI launcher. Example commands:

```bash
mpiexec -n 4 python exercise_1/mpi_matrix_mult.py --n 512
mpiexec -n 4 python exercise_3/mpi_fire_ca.py --grid 400 --steps 80
mpiexec -n 4 python exercise_4/mpi_kmeans.py --samples 50000 --clusters 7
```

## Data Notes

The assignment statement references external public datasets. This repository
now contains the real files used for the experiments:

- Exercise 1: `exercise_1/data/suitesparse/plat362.tar.gz` from SuiteSparse,
  plus additional real SuiteSparse archives already present in the repo.
- Exercise 2: `exercise_2/data/DIC-C2DH-HeLa/` extracted from the provided zip.
- Exercise 3: `exercise_3/data/firms_modis_2024_mexico.csv`.
- Exercise 4: `exercise_4/data/covtype.data.gz` and `covtype.info`.

Notes by exercise:

- Exercise 1 benchmarks dense synthetic matrices plus real sparse matrices from
  SuiteSparse.
- Exercise 2 uses the real DIC-C2DH-HeLa raw frames together with the
  silver-truth segmentation masks in `*_ST/SEG`.
- Exercise 3 uses the real MODIS Mexico 2024 hotspot CSV and documents the date
  and confidence filter used in the report.
- Exercise 4 uses a deterministic benchmark subset from the real Covertype
  dataset to keep repeated serial/parallel experiments tractable on a local
  workstation.

SuiteSparse source pages:

- <https://sparse.tamu.edu/HB/west0479>
- <https://sparse.tamu.edu/HB/bcsstk13>
