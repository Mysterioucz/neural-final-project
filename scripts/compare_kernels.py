"""
Kernel Comparison Script.

Trains Linear, RBF, and Polynomial SVMs on the Wisconsin Breast Cancer dataset
and compares their Accuracy, Precision, Recall, and Training Time.

Results are printed to the console as a formatted table and exported to:
    results/kernel_comparison.csv

Usage
-----
    uv run python scripts/compare_kernels.py
    # or:
    python3 scripts/compare_kernels.py
"""

import csv
import pathlib
import sys
import time

import numpy as np

# Allow running directly from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_loader import load_and_clean_data
from src.data_utils import train_test_split
from src.label_encoder import encode_labels
from src.metrics import accuracy_score, precision_score, recall_score
from src.model import SVM
from src.preprocessing import ManualStandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "data.csv")
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
CSV_PATH = RESULTS_DIR / "kernel_comparison.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42
C = 1.0

# Kernel configurations to evaluate
KERNEL_CONFIGS = [
    {
        "name": "Linear",
        "kernel": "linear",
        "C": C,
        "gamma": "scale",
        "degree": 3,
        "coef0": 0.0,
    },
    {
        "name": "RBF",
        "kernel": "rbf",
        "C": C,
        "gamma": "scale",
        "degree": 3,
        "coef0": 0.0,
    },
    {
        "name": "Polynomial (d=3)",
        "kernel": "poly",
        "C": C,
        "gamma": "scale",
        "degree": 3,
        "coef0": 1.0,
    },
]

# CSV column order
CSV_FIELDNAMES = [
    "Kernel",
    "Accuracy",
    "Precision",
    "Recall",
    "Training_Time_s",
    "Support_Vectors",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_width(header: str, value: str, min_width: int = 0) -> int:
    return max(len(header), len(value), min_width)


def _print_table(rows: list[dict]) -> None:
    """Print results as a formatted ASCII table."""
    headers = {
        "Kernel": "Kernel",
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "Training_Time_s": "Train Time (s)",
        "Support_Vectors": "SVs",
    }

    # Format values for display
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Kernel": row["Kernel"],
                "Accuracy": f"{float(row['Accuracy']) * 100:.2f}%",
                "Precision": f"{float(row['Precision']) * 100:.2f}%",
                "Recall": f"{float(row['Recall']) * 100:.2f}%",
                "Training_Time_s": f"{float(row['Training_Time_s']):.4f}",
                "Support_Vectors": str(row["Support_Vectors"]),
            }
        )

    # Compute column widths
    widths = {k: len(v) for k, v in headers.items()}
    for dr in display_rows:
        for k in widths:
            widths[k] = max(widths[k], len(dr[k]))

    # Build separator and header
    sep = "+-" + "-+-".join("-" * widths[k] for k in headers) + "-+"
    header_line = (
        "| "
        + " | ".join(headers[k].ljust(widths[k]) for k in headers)
        + " |"
    )

    print(sep)
    print(header_line)
    print(sep)
    for dr in display_rows:
        print(
            "| "
            + " | ".join(dr[k].ljust(widths[k]) for k in headers)
            + " |"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    """Run kernel comparison. Returns 0 on success, 1 on failure."""
    try:
        print("=" * 65)
        print("  Kernel Comparison — Wisconsin Breast Cancer Dataset")
        print("=" * 65)

        # ---- Load and preprocess data ----------------------------------------
        X_raw, y_raw = load_and_clean_data(DATA_PATH)
        print(f"\nDataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

        y = encode_labels(y_raw)
        X_np = X_raw.to_numpy()

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_np, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Split  : {X_train_raw.shape[0]} train / {X_test_raw.shape[0]} test")

        scaler = ManualStandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # ---- Train and evaluate each kernel -----------------------------------
        print()
        results: list[dict] = []

        for cfg in KERNEL_CONFIGS:
            name = cfg["name"]
            print(f"Training {name} SVM (C={cfg['C']}) ...", end=" ", flush=True)

            svm = SVM(
                C=cfg["C"],
                kernel=cfg["kernel"],
                gamma=cfg["gamma"],
                degree=cfg["degree"],
                coef0=cfg["coef0"],
            )

            t_start = time.perf_counter()
            svm.fit(X_train, y_train)
            t_elapsed = time.perf_counter() - t_start

            y_pred = svm.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1.0)
            rec = recall_score(y_test, y_pred, pos_label=1.0)
            n_sv = len(svm.alphas_)

            print(f"done in {t_elapsed:.4f}s — Accuracy: {acc * 100:.2f}%")

            results.append(
                {
                    "Kernel": name,
                    "Accuracy": round(acc, 6),
                    "Precision": round(prec, 6),
                    "Recall": round(rec, 6),
                    "Training_Time_s": round(t_elapsed, 6),
                    "Support_Vectors": n_sv,
                }
            )

        # ---- Print results table ----------------------------------------------
        print()
        _print_table(results)

        # ---- Determine best kernel (by accuracy) ------------------------------
        best = max(results, key=lambda r: float(r["Accuracy"]))
        print(
            f"\nBest Kernel : {best['Kernel']} "
            f"(Accuracy: {float(best['Accuracy']) * 100:.2f}%)"
        )

        # ---- Export to CSV ----------------------------------------------------
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with CSV_PATH.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)

        try:
            display_path = CSV_PATH.relative_to(pathlib.Path.cwd())
        except ValueError:
            display_path = CSV_PATH
        print(f"\nResults saved to: {display_path}")
        print()
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
