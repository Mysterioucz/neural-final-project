# neural-final-project

SVM kernel comparison on the Wisconsin Breast Cancer dataset.

## Setup

Requires Python 3.12+.

**With uv (recommended):**

```bash
uv sync
```

**With standard venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install cvxopt matplotlib numpy pandas
```

## Running the scripts

### Kernel comparison

Trains Linear, RBF, and Polynomial SVMs on all features and prints an accuracy/precision/recall table. Results are saved to `results/kernel_comparison.csv`.

```bash
# uv
uv run python scripts/compare_kernels.py

# venv
python3 scripts/compare_kernels.py
```

### Decision boundary visualization

Trains the same three kernels on two features (`radius_mean`, `texture_mean`) and saves side-by-side decision boundary plots to `results/svm_kernels_comparison.png`.

```bash
# uv
uv run python scratch/visualize_svm.py

# venv
python3 scratch/visualize_svm.py
```

Run both scripts from the repo root.
