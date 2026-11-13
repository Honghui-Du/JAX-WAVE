# Using JAX-WAVE

Run commands from the repository root with Python 3.9 or later.

## Installation

For result analysis and numerical checks:

```bash
python -m pip install -r requirements-analysis.txt
```

For the neural model and numerical solver:

```bash
python -m pip install -r requirements-model.txt
```

The model requirements install the CPU version of JAX. For an accelerator, install
the matching JAX build for your device. The model and prediction interfaces are
the same on CPU and GPU.

## Data and checkpoints

Download the nine archive parts listed in `artifacts.json` from the
[data archive](https://drive.google.com/drive/folders/1YucW1H7JgKLcU9j8-GwHiVQ3OwmoBRg5).
Then run:

```bash
python scripts/reproduce.py install --archive /path/to/downloaded-parts
```

The installer checks the archive and extracts the training pairs, validation
pairs and three selected checkpoints into `artifacts/`. It also accepts a
complete archive ZIP. Use `--output PATH` to choose another directory.

Input NPZ files contain `target` and `c`, each with shape `(count, n, n)`.
The grid is square, with `n` divisible by eight. Targets are nonzero, and wave
speeds are positive. The published models use `n=64`.

## Prediction

```bash
python -m jaxwave.predict \
  --input artifacts/validation.npz \
  --checkpoint artifacts/selected_seed0.msgpack \
  --width 64 \
  --output outputs/predictions.npz
```

The output contains `initial_pressure`, `terminal_field` and per-case
`relative_l2`. The terminal field uses the same wave model as training, with an
independent NumPy evaluation in double precision.

## Training

```bash
python -m jaxwave.train \
  --train artifacts/train.npz \
  --validation artifacts/validation.npz \
  --width 64 --seed 0 --updates 60000 \
  --output outputs/model
```

Training minimizes the propagated field mismatch plus the control penalty.
The output directory contains configuration, validation history, and the best
and last parameter checkpoints. The best checkpoint minimizes validation mean
squared relative field error. Repeat with `--seed 1` or `--seed 2` for additional
initializations. Existing output directories are not overwritten.

## Numerical inversion

```python
from jaxwave.data import load_pairs
from jaxwave.numerical import solve, evaluate

target, speed = load_pairs("artifacts/validation.npz")
control = solve(target[:1], speed[:1], iterations=256)
metrics = evaluate(control, target[:1], speed[:1])
print(metrics["relative_l2"])
```

The solver uses projected FISTA with backtracking and the same pressure bounds
and regularized objective as the neural model. Set `restart=True` for adaptive
momentum restart.

## Result analysis

The included measurements support CPU reanalysis without downloading checkpoints:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/analyze_budget_benchmark.py
OPENBLAS_NUM_THREADS=1 python scripts/plot_budget_benchmark.py
python scripts/cluster_sensitivity.py
```

Outputs go to `outputs/`. The benchmark analysis repeats timing-only iteration
selection and the paired bootstrap, and checks agreement with the reported
statistics. The plotting command redraws the three online-budget figures.

## Numerical checks

```bash
python scripts/reproduce.py verify
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py math
```

These verify file checksums and test the discrete propagator, weighted adjoint,
gradient, sign identity and objective-gap certificate on small grids.
