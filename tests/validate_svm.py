"""
End-to-end validation of the SVM implementation on the Wisconsin Breast Cancer dataset.

Trains an SVM with C=1.0 (linear kernel) on standardised data and reports
test-set accuracy together with the full confusion matrix and precision/recall.

Expected accuracy: > 90% (typically > 95% for a linear SVM on this dataset).

Usage
-----
    uv run python tests/validate_svm.py
    # or:
    python3 tests/validate_svm.py
"""

import sys
import pathlib

# Allow running directly from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.data_loader import load_and_clean_data
from src.label_encoder import encode_labels
from src.preprocessing import ManualStandardScaler
from src.data_utils import train_test_split
from src.model import SVM
from src.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "data.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
C = 1.0
ACCURACY_THRESHOLD = 0.90   # Minimum acceptable test accuracy


def main() -> int:
    """Run the validation pipeline.  Returns exit code (0=pass, 1=fail)."""
    print("=" * 60)
    print("SVM Validation — Wisconsin Breast Cancer Dataset")
    print("=" * 60)

    # ---- Load data --------------------------------------------------------
    X_raw, y_raw = load_and_clean_data(DATA_PATH)
    print(f"Dataset loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

    # ---- Encode labels ----------------------------------------------------
    y = encode_labels(y_raw)   # M -> +1.0, B -> -1.0
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == -1.0))
    print(f"Labels: {n_pos} Malignant (+1), {n_neg} Benign (-1)")

    # ---- Train/test split -------------------------------------------------
    X_np = X_raw.to_numpy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_np, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Split: {X_train_raw.shape[0]} train / {X_test_raw.shape[0]} test")

    # ---- Standardise features (fit on train only) -------------------------
    scaler = ManualStandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ---- Train SVM --------------------------------------------------------
    print(f"\nTraining SVM (C={C}, kernel='linear') …")
    svm = SVM(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    n_sv = len(svm.alphas_)
    print(f"Training complete — {n_sv} support vectors found")

    # ---- Evaluate ---------------------------------------------------------
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1.0)
    rec = recall_score(y_test, y_pred, pos_label=1.0)
    cm = confusion_matrix(y_test, y_pred, pos_label=1.0)

    print("\n--- Results ---")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%  (Malignant)")
    print(f"Recall    : {rec * 100:.2f}%  (Malignant)")
    print("\nConfusion Matrix  (rows=actual, cols=predicted):")
    print("                Pred -1   Pred +1")
    print(f"  Actual -1 :   {cm[0, 0]:5d}     {cm[0, 1]:5d}")
    print(f"  Actual +1 :   {cm[1, 0]:5d}     {cm[1, 1]:5d}")

    # ---- Pass / Fail gate ------------------------------------------------
    print("\n--- Threshold Check ---")
    passed = acc >= ACCURACY_THRESHOLD
    status = "PASS" if passed else "FAIL"
    print(f"Accuracy {acc * 100:.2f}% >= {ACCURACY_THRESHOLD * 100:.0f}% threshold: {status}")

    if not passed:
        print(
            f"\n[ERROR] Test accuracy {acc * 100:.2f}% is below the required "
            f"{ACCURACY_THRESHOLD * 100:.0f}% threshold.",
            file=sys.stderr,
        )
        return 1

    print("\nValidation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
