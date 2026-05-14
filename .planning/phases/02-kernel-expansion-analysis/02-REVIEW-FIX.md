---
phase: 02-kernel-expansion-analysis
fixed_at: 2026-05-15T00:00:00Z
review_path: .planning/phases/02-kernel-expansion-analysis/02-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-05-15  
**Source review:** .planning/phases/02-kernel-expansion-analysis/02-REVIEW.md  
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (2 Critical, 5 Warning; 2 Info findings excluded per fix_scope)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: `compare_kernels.py` crashes after CSV write when cwd is not the repo root

**Files modified:** `scripts/compare_kernels.py`  
**Commit:** 80eba89  
**Applied fix:** Wrapped `CSV_PATH.relative_to(pathlib.Path.cwd())` in a `try/except ValueError` block. On failure (cwd is not an ancestor of CSV_PATH), falls back to printing the absolute `CSV_PATH`. The CSV file is already written before this point so data is never lost.

---

### CR-02: Negative/zero integer gamma bypasses validation in `SVM.__init__`

**Files modified:** `src/model.py`  
**Commit:** 0bb44b1  
**Applied fix:** Replaced the `isinstance(gamma, float)` guard with a `not isinstance(gamma, str)` branch that converts via `float()`, catching Python `int`, `np.int64`, `np.float32`, and any other numeric type. Raises `ValueError` if the numeric value is <= 0, preventing silent production of a non-PSD RBF kernel.

---

### WR-01: No validation that `X` is 2-D in `SVM.fit()`

**Files modified:** `src/model.py`  
**Commit:** c72d571  
**Applied fix:** Added `if X.ndim != 2: raise ValueError(...)` immediately after the `np.asarray` call in `fit()`. The error message includes the actual shape, replacing the confusing `IndexError: tuple index out of range` that previously surfaced inside `_resolve_gamma`.

---

### WR-02: No validation that `degree >= 1` for polynomial kernel

**Files modified:** `src/model.py`  
**Commit:** 8f0de32  
**Applied fix:** Added `if degree < 1: raise ValueError(...)` in `SVM.__init__` directly after the `C <= 0` guard, before the kernel is selected or constructed. Blocks `degree=0` (constant/degenerate kernel) and `degree=-1` (non-PSD inverse kernel) from silently producing invalid models.

---

### WR-03: `_PARAM_AWARE_KERNELS` constant is defined but never used

**Files modified:** `src/kernels.py`  
**Commit:** 7186aa9  
**Applied fix:** Removed the `_PARAM_AWARE_KERNELS = {"poly"}` constant and its associated comment entirely. The `_KERNEL_PARAMS` dict already encodes all necessary information for the warning logic in `get_kernel`.

---

### WR-04: Variable `irrelevant` in `get_kernel` stores *relevant* params — inverted semantics

**Files modified:** `src/kernels.py`  
**Commit:** b799969  
**Applied fix:** Renamed local variable `irrelevant` to `kernel_params` and updated the accompanying comment to "Warn about parameters that are irrelevant to the selected kernel". Both `if "degree" not in ...` and `if "coef0" not in ...` checks were updated to use the new name, removing the double-negative logic.

---

### WR-05: `main()` in `compare_kernels.py` never returns `1` despite claiming it can

**Files modified:** `scripts/compare_kernels.py`  
**Commit:** 1194df2  
**Applied fix:** Wrapped the entire `main()` function body in a `try/except Exception` block. On any exception, prints `Error: {exc}` to stderr and returns `1`. Updated the docstring from "Returns exit code (0 = success, 1 = failure)" to "Returns 0 on success, 1 on failure." to be consistent with the now-correct behavior.

---

_Fixed: 2026-05-15_  
_Fixer: Claude (gsd-code-fixer)_  
_Iteration: 1_
