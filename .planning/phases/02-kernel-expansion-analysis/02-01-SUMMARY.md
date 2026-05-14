---
phase: "02"
plan: "01"
subsystem: kernel-expansion
tags: [svm, kernels, rbf, polynomial, dual-formulation, kernel-trick, evaluation]
dependency_graph:
  requires: [01-01, 01-02]
  provides: [rbf-kernel, polynomial-kernel, kernel-expansion-predict, kernel-comparison-report]
  affects: [src/kernels.py, src/model.py, scripts/compare_kernels.py]
tech_stack:
  added: []
  patterns:
    - Kernel factory pattern with get_kernel() returning callables
    - Vectorized RBF using ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> identity
    - Non-linear bias via kernel expansion over free support vectors
    - Internal predict() branching for O(d) linear vs kernel-expansion paths
key_files:
  created:
    - scripts/compare_kernels.py
    - results/kernel_comparison.csv
  modified:
    - src/kernels.py
    - src/model.py
decisions:
  - D-01: Kernel parameters (gamma, degree, coef0) added to SVM.__init__ mirroring scikit-learn API
  - D-02: Centralized kernel factory get_kernel() in src/kernels.py keeps model.py clean
  - D-03: UserWarning issued for irrelevant kernel parameters (e.g., degree for linear/rbf)
  - D-05: Internal predict() branch — O(d) w@x for linear, kernel expansion for non-linear
  - D-06: w_=None for non-linear kernels prevents misuse of weight vector
  - D-07: Bias b computed as mean over free SVs (0<alpha<C); fallback to all SVs if none
  - D-08: All kernel evaluations vectorized with NumPy broadcasting (n_sv, n_test) Gram matrices
  - D-09: Default gamma='scale' (1/(n_features*X.var())), gamma='auto' (1/n_features) also supported
  - D-10: compare_kernels.py outputs ASCII table to console and saves CSV to results/
  - D-11: Best Kernel recommendation printed based on highest test accuracy
metrics:
  duration: "4 min"
  completed: "2026-05-15"
  tasks_completed: 3
  tasks_total: 3
---

# Phase 02 Plan 01: Kernel Expansion & Analysis Summary

## One-liner

RBF and Polynomial kernels implemented via factory pattern with vectorized NumPy, non-linear SVM predict via kernel expansion, comparative analysis script achieving 99.12% accuracy across all three kernels on breast cancer dataset.

## What Was Built

### Task 1: Kernel implementations in `src/kernels.py` (commit `12c6f8f`)

- `rbf_kernel(gamma)` — returns a callable using the identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> for vectorized computation; numerical clamp prevents tiny negative values from floating-point error.
- `polynomial_kernel(gamma, degree, coef0)` — returns a callable computing K(x,y)=(gamma*<x,y>+coef0)^degree using fully vectorized matrix multiplication.
- `get_kernel(kernel, gamma, degree, coef0)` — factory that returns the correct callable for 'linear', 'rbf', or 'poly'. Issues `UserWarning` for irrelevant parameters. Raises `ValueError` for unsupported kernel names.
- Existing `linear_kernel` retained unchanged for backward compatibility.

### Task 2: Non-linear SVM refactor in `src/model.py` (commit `528c27b`)

- `SVM.__init__` now accepts `gamma`, `degree`, `coef0` with defaults mirroring scikit-learn (`gamma='scale'`, `degree=3`, `coef0=0.0`).
- `_resolve_gamma(X)` resolves 'scale' → `1/(n_features * X.var())`, 'auto' → `1/n_features`, or uses the float directly.
- `_build_qp_matrices` uses `get_kernel()` instead of the hardcoded `linear_kernel` call.
- `fit()` computes non-linear bias via kernel expansion over free support vectors; falls back to all SVs if none are free.
- `predict()` uses the O(d) linear path for `kernel='linear'` and the kernel expansion `(alpha*y) @ K(X_sv, X_test) + b` for non-linear kernels.
- `w_` set to `None` for non-linear kernels (D-06).
- All 23 existing tests pass; linear SVM validation achieves 99.12% accuracy.

### Task 3: `scripts/compare_kernels.py` (commit `eaf37bf`)

- Trains Linear, RBF, and Polynomial (d=3, coef0=1.0) SVMs with identical preprocessing (ManualStandardScaler, 80/20 split, seed=42).
- Measures training time with `time.perf_counter`.
- Computes Accuracy, Precision, Recall via `src.metrics`.
- Prints formatted ASCII table with all metrics.
- Exports raw results to `results/kernel_comparison.csv`.
- Prints "Best Kernel" recommendation based on highest accuracy.

## Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| `rbf_kernel` in src/kernels.py | PASS |
| `polynomial_kernel` in src/kernels.py | PASS |
| `get_kernel` returns correct function for 'linear', 'rbf', 'poly' | PASS |
| Vectorized RBF uses ||x-y||^2 identity | PASS |
| `SVM(kernel='rbf')` fits without errors | PASS |
| `predict()` branch for `self.kernel == 'linear'` | MUST match |
| Bias calculated as mean over free support vectors | PASS |
| `w_` is None for non-linear kernels | PASS |
| compare_kernels.py runs to completion | PASS |
| results/kernel_comparison.csv produced | PASS |
| "Best Kernel" recommendation in console output | PASS |

## Key Results

All three kernels achieve **99.12% accuracy** on the breast cancer test set:

| Kernel           | Accuracy | Precision | Recall   | SVs |
|------------------|----------|-----------|----------|-----|
| Linear           | 99.12%   | 100.00%   | 97.62%   | 39  |
| RBF              | 99.12%   | 97.67%    | 100.00%  | 117 |
| Polynomial (d=3) | 99.12%   | 97.67%    | 100.00%  | 61  |

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None — all kernel implementations are fully wired with real data and produce live results.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced. The CSV export writes to a local `results/` directory with no external exposure.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| src/kernels.py exists | FOUND |
| src/model.py exists | FOUND |
| scripts/compare_kernels.py exists | FOUND |
| results/kernel_comparison.csv exists | FOUND |
| 02-01-SUMMARY.md exists | FOUND |
| Commit 12c6f8f (Task 1) | FOUND |
| Commit 528c27b (Task 2) | FOUND |
| Commit eaf37bf (Task 3) | FOUND |
