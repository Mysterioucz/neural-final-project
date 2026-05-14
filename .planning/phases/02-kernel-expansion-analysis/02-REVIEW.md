---
phase: 02-kernel-expansion-analysis
reviewed: 2026-05-15T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/kernels.py
  - src/model.py
  - scripts/compare_kernels.py
  - results/kernel_comparison.csv
findings:
  critical: 2
  warning: 5
  info: 2
  total: 9
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-15  
**Depth:** standard  
**Files Reviewed:** 4  
**Status:** issues_found

## Summary

Reviewed `src/kernels.py`, `src/model.py`, `scripts/compare_kernels.py`, and `results/kernel_comparison.csv`. The kernel implementations (linear, RBF, polynomial) and QP-based SVM dual solver are structurally sound, and the numerical math is correct for the happy path. However, two blockers were found: a runtime crash in the comparison script when the working directory is not the repository root, and a validation gap in `SVM.__init__` that silently accepts negative or zero integer gamma values, producing a numerically wrong (but non-crashing) kernel. Five additional warnings cover missing input validation, dead code, and a misleading variable name that inverts the intent of the warning logic.

---

## Critical Issues

### CR-01: `compare_kernels.py` crashes after CSV write when cwd is not the repo root

**File:** `scripts/compare_kernels.py:226`  
**Issue:** `CSV_PATH.relative_to(pathlib.Path.cwd())` raises `ValueError` whenever the script is invoked from a directory that is not an ancestor of `CSV_PATH`. The CSV file has already been written successfully at this point, so data is not lost, but the script exits with an unhandled exception rather than exit code 0. Callers that check the exit code (CI scripts, `sys.exit(main())`) see a failure despite a successful result.

Reproduction: `python3 /abs/path/to/scripts/compare_kernels.py` run from `/tmp`.

```python
# ValueError: '/abs/path/to/results/kernel_comparison.csv' is not in the subpath of '/tmp'
```

**Fix:** Replace the `relative_to` call with a safe fallback:
```python
try:
    display_path = CSV_PATH.relative_to(pathlib.Path.cwd())
except ValueError:
    display_path = CSV_PATH
print(f"\nResults saved to: {display_path}")
```
Or, since `CSV_PATH` is already constructed from the resolved script directory, just print it directly:
```python
print(f"\nResults saved to: {CSV_PATH}")
```

---

### CR-02: Negative/zero integer gamma bypasses validation in `SVM.__init__`

**File:** `src/model.py:98-99`  
**Issue:** The guard for invalid numeric gamma uses `isinstance(gamma, float)`, which is `False` for Python `int` and all NumPy numeric types (`np.int64`, `np.float32`, etc.). A user passing `gamma=-1` or `gamma=0` (as an `int`) silently passes validation. `_resolve_gamma` then returns `float(-1)` or `float(0)`. For the RBF kernel this produces `exp(-(-1) * ||x-y||^2) = exp(||x-y||^2)`, which grows with distance — the opposite of the intended similarity measure and not a positive-semi-definite kernel. The QP solver will still converge, yielding a silently wrong model.

```python
# Current — misses int, numpy ints, numpy floats
if isinstance(gamma, float) and gamma <= 0:
    raise ValueError(...)

# Demonstration
gamma = -1          # int
isinstance(gamma, float)  # False -> check skipped -> -1.0 reaches rbf_kernel
```

**Fix:** Broaden the numeric check:
```python
if not isinstance(gamma, str):
    try:
        gamma_f = float(gamma)
    except (TypeError, ValueError):
        raise ValueError(f"gamma must be a float or 'scale'/'auto', got {gamma!r}.")
    if gamma_f <= 0:
        raise ValueError(f"gamma must be positive when given as a number, got {gamma}.")
```

---

## Warnings

### WR-01: No validation that `X` is 2-D in `SVM.fit()`

**File:** `src/model.py:220-232`  
**Issue:** `fit()` calls `np.asarray(X, dtype=np.float64)` but never checks `X.ndim == 2`. If `X` is 1-D (shape `(n_features,)`), the first use of `X.shape[1]` inside `_resolve_gamma` raises `IndexError: tuple index out of range` — a confusing, low-signal error message with no indication that the input shape is wrong.

**Fix:** Add an explicit shape check after the `np.asarray` call:
```python
X = np.asarray(X, dtype=np.float64)
if X.ndim != 2:
    raise ValueError(
        f"X must be a 2-D array of shape (n_samples, n_features), got shape {X.shape}."
    )
```

---

### WR-02: No validation that `degree >= 1` for polynomial kernel

**File:** `src/model.py:76-131` and `src/kernels.py:65-106`  
**Issue:** Neither `SVM.__init__` nor `polynomial_kernel` validates that `degree` is a positive integer. `degree=0` produces `K(x,y) = 1` for all pairs (constant kernel — the QP is degenerate). `degree=-1` produces an inverse function that is not positive-semi-definite. The solver may still converge, yielding a silently invalid model.

**Fix:** Add a degree guard in `SVM.__init__` (near line 83) and optionally inside `polynomial_kernel`:
```python
if degree < 1:
    raise ValueError(f"degree must be a positive integer, got {degree}.")
```

---

### WR-03: `_PARAM_AWARE_KERNELS` constant is defined but never used

**File:** `src/kernels.py:156`  
**Issue:** `_PARAM_AWARE_KERNELS = {"poly"}` is defined as a module-level constant but is never referenced anywhere in the file or in any other module. It is dead code that creates a false impression that the set is used by some dispatch or validation logic.

**Fix:** Remove the constant entirely, or use it to drive the `_KERNEL_PARAMS` lookup so the two data structures remain consistent:
```python
# Remove line 156:
_PARAM_AWARE_KERNELS = {"poly"}   # DELETE — never used
```

---

### WR-04: Variable `irrelevant` in `get_kernel` stores *relevant* params — inverted semantics

**File:** `src/kernels.py:215-228`  
**Issue:** The local variable `irrelevant` is assigned `_KERNEL_PARAMS.get(kernel, set())`, which stores the parameters that *are* meaningful to the given kernel (e.g., `{'degree', 'coef0'}` for `'poly'`). The warning logic then reads `if 'degree' not in irrelevant`, which is correct in effect but inverted in intent: the variable name says "irrelevant" while the set contains the relevant ones. This makes the logic read as a double-negative and will mislead any future maintainer.

**Fix:** Rename to reflect true semantics:
```python
# Before
irrelevant = _KERNEL_PARAMS.get(kernel, set())
if "degree" not in irrelevant and degree != 3:

# After
kernel_params = _KERNEL_PARAMS.get(kernel, set())
if "degree" not in kernel_params and degree != 3:
```

---

### WR-05: `main()` in `compare_kernels.py` never returns `1` despite claiming it can

**File:** `scripts/compare_kernels.py:148`  
**Issue:** The docstring says "Returns exit code (0 = success, 1 = failure)", but the function body returns `0` unconditionally (line 228) and handles all failure conditions by raising exceptions, not by returning `1`. `sys.exit(main())` will propagate the exception as a non-zero OS exit, but the contract documented in the docstring is never fulfilled. Callers that parse `main()`'s return value will always see `0` or an exception — never `1`.

**Fix:** Either correct the docstring to remove the "1 = failure" claim, or introduce explicit error handling that returns `1`:
```python
def main() -> int:
    """Run kernel comparison. Returns 0 on success, 1 on failure."""
    try:
        ...
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
```

---

## Info

### IN-01: `linear_kernel` returns `np.float64` (not `np.ndarray`) for 1-D inputs

**File:** `src/kernels.py:58-59`  
**Issue:** When both inputs are 1-D, `np.dot(x1, x2)` returns `np.float64`, which is not a subclass of `np.ndarray` in NumPy 2.x. The return-type annotation `-> np.ndarray` is therefore inaccurate for this branch, and callers relying on `.shape` or array methods on the result will get a 0-D generic scalar rather than a proper array object.

This function is not called with 1-D inputs by `get_kernel` (which wraps with `atleast_2d`), so there is no current breakage, but the annotation misleads users of the raw function.

**Fix:** Either update the annotation to `Union[np.floating, np.ndarray]` / `np.ndarray | np.floating`, or use `np.atleast_2d` on both inputs unconditionally and return a 2-D Gram matrix in all cases. The latter aligns with all other kernel functions in the module.

---

### IN-02: Inconsistent capitalisation of QP matrix variable `H` vs `h`

**File:** `src/model.py:190-199`  
**Issue:** Inside `_build_qp_matrices`, the inequality upper-bound vector is built into `h_np`, then stored in `H = cvxopt.matrix(h_np, tc="d")` (uppercase), while all other matrices use lowercase (`P`, `q`, `G`, `A`, `b_qp`). The return statement (line 199) returns `H`, which the caller (line 241) unpacks as `h` (lowercase). The capital `H` is inconsistent with the rest of the naming scheme and with standard QP notation where `h` is the bound vector.

**Fix:** Rename to lowercase throughout `_build_qp_matrices`:
```python
h = cvxopt.matrix(h_np, tc="d")
...
return P, q, G, h, A, b_qp
```

---

_Reviewed: 2026-05-15_  
_Reviewer: Claude (gsd-code-reviewer)_  
_Depth: standard_
