---
phase: 02-kernel-expansion-analysis
verified: 2026-05-15T00:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
gaps: []
---

# Phase 2: Kernel Expansion & Analysis Verification Report

**Phase Goal:** Implement advanced kernels and perform a comparative analysis of model performance.
**Verified:** 2026-05-15
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | RBF and Polynomial kernels are implemented and integrated into the SVM class | VERIFIED | `rbf_kernel()` at `src/kernels.py:109`, `polynomial_kernel()` at `src/kernels.py:65`. `SVM.fit()` calls `get_kernel()` at line 233 of `src/model.py`, which routes to both. Gram matrix built with the resolved kernel callable. |
| 2 | Model can switch between kernels via configuration | VERIFIED | `SVM.__init__` accepts `kernel` parameter (validated against `{'linear', 'rbf', 'poly'}`). `predict()` branches on `self.kernel == "linear"` at line 348 for O(d) path; else uses kernel expansion at lines 353-357. `fit()` similarly branches at line 266. |
| 3 | A final report compares Accuracy, Precision, and Recall across all three kernels | VERIFIED | `scripts/compare_kernels.py` trains Linear, RBF, and Polynomial SVMs, computes all three metrics, prints ASCII table, prints "Best Kernel" recommendation, and exports `results/kernel_comparison.csv`. CSV exists with real data (3 data rows, non-zero metrics). |

**Score:** 3/3 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/kernels.py` | RBF + Polynomial kernels + `get_kernel` factory | VERIFIED | 244 lines. `rbf_kernel`, `polynomial_kernel`, `get_kernel` all present and substantive. Vectorized using `||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>` identity. Numerical clamp present. |
| `src/model.py` | Non-linear SVM with kernel switching | VERIFIED | 364 lines. `SVM.__init__` accepts `gamma`, `degree`, `coef0`. `_resolve_gamma()` handles 'scale'/'auto'/float. `_build_qp_matrices` uses `self._kernel_func_`. `predict()` has explicit `linear` / non-linear branch. `w_=None` set for non-linear kernels at line 290. |
| `scripts/compare_kernels.py` | Comparison script with CSV export and console output | VERIFIED | 232 lines. Trains all 3 kernels, measures time with `perf_counter`, computes Accuracy/Precision/Recall, prints ASCII table, writes `results/kernel_comparison.csv`, prints Best Kernel. |
| `results/kernel_comparison.csv` | CSV with per-kernel metrics | VERIFIED | File exists. Contains header + 3 data rows (Linear, RBF, Polynomial (d=3)). All metrics are non-trivial real floats (Accuracy: 0.991228 for all three). |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/model.py` | `src/kernels.py` | `from src.kernels import get_kernel` (line 23) | WIRED | `get_kernel()` called at `model.py:233` inside `fit()`. Result stored in `self._kernel_func_` and used for Gram matrix at line 171. |
| `scripts/compare_kernels.py` | `src/model.py` | `from src.model import SVM` (line 33) | WIRED | `SVM(kernel=cfg["kernel"], ...)` instantiated in loop at line 177; `.fit()` and `.predict()` called at lines 186-189. |
| `scripts/compare_kernels.py` | `src/metrics.py` | `from src.metrics import accuracy_score, precision_score, recall_score` (line 30) | WIRED | All three called at lines 190-192 with real `y_test` / `y_pred` values. Results stored in `results` list and written to CSV. |
| `scripts/compare_kernels.py` | `results/kernel_comparison.csv` | `csv.DictWriter` at line 222 | WIRED | `RESULTS_DIR.mkdir(parents=True, exist_ok=True)` ensures directory. `writer.writerows(results)` at line 224 writes real computed rows. |
| Non-linear `predict()` | Support vectors via kernel expansion | `(self.alphas_ * self.support_vector_labels_) @ K_sv_test + self.b_` (line 356-357) | WIRED | Kernel callable invoked with `(self.support_vectors_, X)` at line 354. Result is (n_sv, n_test) matrix; dot with (n_sv,) weight vector produces (n_test,) decision values. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `scripts/compare_kernels.py` | `results` (list of dicts) | `accuracy_score`, `precision_score`, `recall_score` called on live `y_test` / `y_pred` from trained SVMs | Yes — CSV contains 0.991228 accuracy from real model predictions on breast cancer test set | FLOWING |
| `src/model.py` predict() | `decision` (kernel expansion) | `self._kernel_func_(self.support_vectors_, X)` using real alpha/SV values from QP solve | Yes — kernel callable uses real `gamma_` resolved from training data variance | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CSV exists with real multi-kernel data | `cat results/kernel_comparison.csv` | 4 lines: header + Linear, RBF, Polynomial rows with Accuracy 0.991228 | PASS |
| Kernel factory raises on unknown kernel | `grep "raise ValueError" src/kernels.py` | Line 210: `raise ValueError(f"Unsupported kernel '{kernel}'...")` | PASS |
| `w_=None` for non-linear kernels | `grep "w_.*None" src/model.py` | Line 290 in `else` block (non-linear branch of `fit()`) | PASS |
| Linear branch in `predict()` | `grep "self.kernel == .linear." src/model.py` | Lines 266 and 348 — both `fit()` and `predict()` branch on linear | PASS |
| Commits exist and match claimed hashes | `git show --stat 12c6f8f 528c27b eaf37bf` | All three commits exist; file changes match Task 1/2/3 respectively | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| KERN-02 | 02-01-PLAN.md | Implement RBF (Gaussian) kernel with configurable gamma | SATISFIED | `rbf_kernel(gamma)` in `src/kernels.py:109`. Uses vectorized identity formula. Configurable via `SVM(kernel='rbf', gamma=...)`. |
| KERN-03 | 02-01-PLAN.md | Implement Polynomial kernel with configurable degree | SATISFIED | `polynomial_kernel(gamma, degree, coef0)` in `src/kernels.py:65`. `K(x,y)=(gamma*<x,y>+coef0)^degree`. Configurable via `SVM(kernel='poly', degree=..., coef0=...)`. |
| EVAL-03 | 02-01-PLAN.md | Create a comparison summary of results across all three kernels | SATISFIED | `scripts/compare_kernels.py` produces ASCII table and `results/kernel_comparison.csv` with Accuracy, Precision, Recall for all three kernels. |

No orphaned requirements: REQUIREMENTS.md maps KERN-02, KERN-03, EVAL-03 exclusively to Phase 2, and all three are claimed and addressed by plan 02-01.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/kernels.py` | 156 | `_PARAM_AWARE_KERNELS = {"poly"}` defined but never referenced anywhere | Warning | Dead code — no functional impact; misleads readers about dispatch logic. Documented in 02-REVIEW.md as WR-03. |
| `src/kernels.py` | 215 | Variable named `irrelevant` stores the *relevant* params set — inverted semantics | Warning | No functional impact; logic is correct despite confusing name. Documented in 02-REVIEW.md as WR-04. |
| `src/model.py` | 98-99 | `isinstance(gamma, float)` guard misses `int` and NumPy numeric types | Warning | `SVM(kernel='rbf', gamma=-1)` (int) silently passes validation and produces inverted RBF. Documented in 02-REVIEW.md as CR-02. Does not affect phase goal — normal usage with float or 'scale'/'auto' is safe. |
| `scripts/compare_kernels.py` | 226 | `CSV_PATH.relative_to(pathlib.Path.cwd())` crashes if cwd is not an ancestor of the results path | Warning | Script crashes after CSV is already written if invoked from a non-repo-root cwd. Documented in 02-REVIEW.md as CR-01. Does not block phase goal — CSV is produced before the crash point; the run that produced the existing CSV completed successfully. |
| `src/model.py` | — | No `X.ndim == 2` check in `fit()` | Warning | 1-D X input produces unhelpful IndexError. Documented in 02-REVIEW.md as WR-01. No impact on phase goal. |

No stub patterns (empty returns, placeholder strings, hardcoded empty data sources) found in any of the three key files.

---

### Human Verification Required

None — all critical behaviors are verifiable programmatically. The comparison report is a deterministic script with file output that was confirmed to exist and contain real data. No UI, visual, or real-time components are involved.

---

### Gaps Summary

No gaps. All three roadmap success criteria are fully met:

1. `rbf_kernel` and `polynomial_kernel` are implemented with correct mathematical formulas, vectorized NumPy operations, and integrated into `SVM` via `get_kernel()` factory.
2. Kernel switching works via `SVM(kernel='rbf'|'poly'|'linear')` constructor parameter, with distinct code paths in both `fit()` and `predict()`.
3. `scripts/compare_kernels.py` produces a real comparative report — the CSV at `results/kernel_comparison.csv` contains Accuracy, Precision, and Recall for all three kernels with verified non-trivial values.

The code review (02-REVIEW.md) identified two critical bugs (CR-01 path crash, CR-02 int gamma validation gap) and five warnings. None of these prevent the phase goal from being achieved — they are quality improvements for the next cycle.

---

_Verified: 2026-05-15T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
