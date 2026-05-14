---
phase: 01-foundation-linear-svm
reviewed: 2026-05-14T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - src/data_loader.py
  - src/label_encoder.py
  - src/preprocessing.py
  - src/data_utils.py
  - tests/test_data_pipeline.py
  - src/kernels.py
  - src/model.py
  - src/metrics.py
  - tests/validate_svm.py
  - pyproject.toml
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-05-14
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Reviewed a Dual Lagrangian SVM implementation over the Wisconsin Breast Cancer dataset. The QP matrix construction is correct and was verified end-to-end with cvxopt. The validate_svm.py pipeline correctly fits the scaler on train data only. However, two critical bugs were found: `predict()` silently returns `NaN` predictions when the QP solver finds no support vectors above the threshold (degenerate case), and the test fixture leaks all 569 samples into the scaler before any train/test split, invalidating the standardisation tests as a correctness check. Five warnings cover missing input validation, a redundant condition, boundary behaviour of `np.sign`, and a missing F1 metric that matters for the class-imbalanced dataset.

---

## Critical Issues

### CR-01: Silent NaN predictions when no support vectors survive the alpha threshold

**File:** `src/model.py:158-186`

**Issue:** After solving the QP, `sv_mask = alphas_all > 1e-5` is applied. If the QP returns an optimal solution where all Lagrange multipliers happen to fall below `1e-5` (possible with very small C or a nearly-linear-separable set with a numeric outlier), `support_vectors_` becomes an empty array and both branches of the bias computation reduce to `np.mean([])`, which returns `NaN` with a `RuntimeWarning` — not an exception. `predict()` subsequently returns `NaN` for every sample because `w_` is the zero vector and `b_` is `NaN`. The `RuntimeError` guard in `predict()` (line 211) does not catch this because `w_` and `b_` are not `None`; they are numerically invalid.

**Proof of behaviour:**
```python
import numpy as np
alphas_sv = np.array([])
labels_sv = np.array([])
b = float(np.mean(labels_sv - np.zeros((0, 2)) @ np.zeros(2)))
# RuntimeWarning: Mean of empty slice -> b = nan
```

**Fix:** After extracting support vectors, raise a `RuntimeError` with a diagnostic message if no support vectors are found:
```python
sv_mask = alphas_all > self._ALPHA_THRESHOLD
if not np.any(sv_mask):
    raise RuntimeError(
        "No support vectors found (all alphas below threshold). "
        "Try increasing C or reducing _ALPHA_THRESHOLD."
    )
self.support_vectors_ = X[sv_mask]
# ... rest of fit()
```

---

### CR-02: Test fixture fits scaler on entire dataset before split — leaks test statistics into scaler

**File:** `tests/test_data_pipeline.py:29-35`

**Issue:** The `pipeline_data` fixture calls `scaler.fit_transform(X)` on all 569 samples. The resulting `X_scaled` is used to test that `mean ≈ 0` and `std ≈ 1` (lines 105-121). These tests pass trivially because the scaler was fit on the very data being measured — this is a tautology, not a correctness test. More importantly, this pattern normalises test-set statistics into the scaler, which is data leakage. The production path in `validate_svm.py` (lines 65-67) correctly fits on train only, but the test suite validates a broken workflow instead of the correct one.

**Additionally**, `TestTrainTestSplit.test_split_shapes_on_actual_data` (lines 165-174) repeats the same pattern: it creates a fresh scaler, calls `fit_transform(X)` on the full feature matrix, and only then splits — again encoding the wrong (leaky) pipeline.

**Fix:** Rewrite `pipeline_data` to split first, then fit on train only:
```python
@pytest.fixture(scope="module")
def pipeline_data(raw_data):
    X, y = raw_data
    y_encoded = encode_labels(y)
    X_np = np.asarray(X, dtype=np.float64)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_np, y_encoded, test_size=0.2, random_state=42
    )
    scaler = ManualStandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, y_train, y_test
```
Update the `TestManualStandardScaler` tests to assert mean/std on `X_train` (where they should be ≈ 0/1) and separately assert that `X_test` mean/std differ (proving the scaler was not re-fit on test data).

---

## Warnings

### WR-01: `predict()` maps decision-boundary samples to 0.0, not a valid class label

**File:** `src/model.py:216`

**Issue:** `np.sign(0.0)` returns `0.0`, not `+1.0` or `-1.0`. A sample whose decision value is exactly zero is classified as neither class. All downstream metrics (`accuracy_score`, `precision_score`, `confusion_matrix`) treat `0.0` as the negative class silently, corrupting metric counts. While this is measure-zero in practice, it is a correctness gap that can surface with synthetic unit-test data constructed exactly on the margin.

**Fix:**
```python
raw = np.sign(decision)
# Replace exactly-zero entries with +1 (arbitrary tie-break, document it)
return np.where(raw == 0.0, 1.0, raw)
```

---

### WR-02: `train_test_split` does not validate that `X` and `y` have the same number of samples

**File:** `src/data_utils.py:45`

**Issue:** `n_samples = X_arr.shape[0]` uses only `X`'s row count. If `len(y) < n_samples`, the function generates indices up to `n_samples - 1` and then attempts `y_arr[train_idx]`, raising an opaque `IndexError` with no indication that the mismatch between `X` and `y` is the cause.

**Fix:** Add an explicit check at the top of the function:
```python
if X_arr.shape[0] != y_arr.shape[0]:
    raise ValueError(
        f"X and y must have the same number of samples, "
        f"got X: {X_arr.shape[0]}, y: {y_arr.shape[0]}."
    )
```

---

### WR-03: `SVM.fit()` accepts single-class input but produces a degenerate model

**File:** `src/model.py:140`

**Issue:** The label validation `set(np.unique(y)) - {1.0, -1.0}` only checks for unknown values; it does not require both classes to be present. Passing a single-class target (all `+1.0`) satisfies the check, but the QP equality constraint `sum(alpha_i * y_i) = 0` with all `y_i = +1` forces all `alpha_i = 0`. This triggers the no-support-vectors path described in CR-01.

**Fix:** Add a both-classes check:
```python
unique_labels = set(np.unique(y))
if unknown := unique_labels - {1.0, -1.0}:
    raise ValueError(f"Unknown labels: {unknown}. Expected +1 and -1.")
if not ({1.0, -1.0} <= unique_labels):
    raise ValueError("Training data must contain both classes (+1 and -1).")
```

---

### WR-04: Redundant condition in `free_mask` (first clause is always `True`)

**File:** `src/model.py:173-175`

**Issue:** `self.alphas_` is constructed by filtering `alphas_all` with `sv_mask = alphas_all > _ALPHA_THRESHOLD`. Therefore every element of `self.alphas_` already satisfies `> _ALPHA_THRESHOLD`. The first clause of the `free_mask` conjunction — `self.alphas_ > self._ALPHA_THRESHOLD` — is always `True` and contributes nothing. This is dead code that implies the author may have intended a stricter lower bound but did not implement it.

**Fix:** Remove the redundant clause to make the intent explicit:
```python
free_mask = self.alphas_ < self.C - self._ALPHA_THRESHOLD
```

---

### WR-05: `test_size` boundary rounding silently produces an empty test or train set

**File:** `src/data_utils.py:46`

**Issue:** `n_test = int(np.round(n_samples * test_size))` can round to `0` or `n_samples` for extreme-but-technically-valid values of `test_size` (e.g. `test_size=0.001` on 10 samples gives `n_test=0`). The docstring says `test_size` must be in `(0, 1)` but does not guarantee non-empty splits. A zero-length test set causes `accuracy_score` and other metrics to return `NaN` via `np.mean` on an empty array.

**Fix:** After computing `n_test` and `n_train`, add:
```python
if n_test == 0 or n_train == 0:
    raise ValueError(
        f"test_size={test_size} with {n_samples} samples produces an empty "
        f"train or test set (n_train={n_train}, n_test={n_test})."
    )
```

---

## Info

### IN-01: `metrics.py` is missing an F1-score function

**File:** `src/metrics.py`

**Issue:** The dataset is class-imbalanced (357 Benign vs. 212 Malignant, roughly 63/37). `validate_svm.py` reports precision and recall but not F1, which is the standard summary metric for imbalanced binary classification. A model that predicts all-Benign achieves 63% accuracy and 0% recall — the absence of F1 makes this failure mode harder to detect from the reported metrics alone.

**Fix:** Add:
```python
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: float = 1.0) -> float:
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    denom = prec + rec
    return (2 * prec * rec) / denom if denom > 0 else 0.0
```

---

### IN-02: `pyproject.toml` has no pytest configuration and no test path specification

**File:** `pyproject.toml`

**Issue:** There is no `[tool.pytest.ini_options]` section. Running `pytest` from the project root may not discover `tests/` automatically depending on the pytest version and directory layout. The `testpaths`, `pythonpath`, and `addopts` settings are absent. Reproducibility is reduced.

**Fix:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v"
```

---

### IN-03: `linear_kernel` silently mishandles mixed 1-D/2-D input with incompatible shapes

**File:** `src/kernels.py:49-53`

**Issue:** When `x1` is 1-D (shape `(d,)`) and `x2` is 2-D (shape `(m, d2)`) with `d != d2`, the function falls into the matrix branch and attempts `x1 @ x2.T`, which raises a `ValueError` with a shape message that does not mention the kernel. Similarly, when both are 2-D with incompatible inner dimensions, the error message gives no context. This is a documentation/usability gap rather than a bug, but misleading for callers.

**Fix:** Add a shape assertion before the matrix multiply:
```python
if x1.shape[-1] != x2.shape[-1]:
    raise ValueError(
        f"linear_kernel: feature dimension mismatch — "
        f"x1 has {x1.shape[-1]} features, x2 has {x2.shape[-1]}."
    )
```

---

_Reviewed: 2026-05-14_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
