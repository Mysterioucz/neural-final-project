---
phase: 01-foundation-linear-svm
plan: "02"
subsystem: model
tags: [svm, cvxopt, quadratic-programming, dual-lagrangian, linear-kernel, numpy, metrics]

requires:
  - phase: 01-01
    provides: data_loader, label_encoder, ManualStandardScaler, train_test_split

provides:
  - linear_kernel function (Gram matrix or dot product)
  - SVM class with fit() via cvxopt QP solver and predict() via w,b decision function
  - accuracy_score, precision_score, recall_score, confusion_matrix metrics
  - tests/validate_svm.py end-to-end validation (99.12% accuracy on breast cancer)

affects:
  - phase-02 (kernel extensions — RBF, Polynomial will add kernel= options to SVM)
  - any future evaluation or reporting phase

tech-stack:
  added: [cvxopt==1.3.3]
  patterns:
    - QP matrix mapping (P,q,G,h,A,b) for soft-margin SVM dual
    - Support vector threshold alpha > 1e-5
    - Free support vector bias computation (0 < alpha < C)
    - 1e-8 ridge added to P for positive semi-definiteness

key-files:
  created:
    - src/kernels.py
    - src/model.py
    - src/metrics.py
    - tests/validate_svm.py
  modified: []

key-decisions:
  - "1e-8 ridge added to P matrix to guarantee positive semi-definiteness for cvxopt"
  - "Bias computed from free support vectors (0 < alpha < C); falls back to all SVs if none free"
  - "cvxopt.solvers.options['show_progress'] = False suppresses solver output globally at import"
  - "predict() uses linear w,b path (X @ w + b) rather than kernel expansion for O(d) inference"

patterns-established:
  - "SVM follows sklearn-compatible fit(X,y)/predict(X) API"
  - "All metrics are manual NumPy — no scikit-learn dependency"
  - "Validation scripts live in tests/ and exit with non-zero code on failure"

requirements-completed: [SVM-01, SVM-02, SVM-03, SVM-04, KERN-01, EVAL-01, EVAL-02]

duration: 2min
completed: 2026-05-14
---

# Phase 01 Plan 02: SVM Dual Formulation and cvxopt Integration Summary

**Soft-margin linear SVM via cvxopt QP solver achieving 99.12% accuracy on Wisconsin Breast Cancer with manual NumPy metrics**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-05-14T16:34:01Z
- **Completed:** 2026-05-14T16:36:41Z
- **Tasks:** 5
- **Files created:** 4

## Accomplishments

- Implemented `linear_kernel` handling both vector dot-product and matrix Gram-matrix cases
- Built `SVM` class that maps the dual problem to cvxopt P,q,G,h,A,b matrices, solves via `cvxopt.solvers.qp`, extracts support vectors (alpha > 1e-5), and computes w and b
- Implemented predict() using the efficient linear decision function X @ w + b with np.sign output
- Wrote manual NumPy metrics: accuracy_score, precision_score, recall_score, confusion_matrix
- End-to-end validation script achieves 99.12% accuracy (42 test errors: 1 false negative, 0 false positives)

## Task Commits

1. **Task 1: linear_kernel** - `5469959` (feat)
2. **Tasks 2+3: SVM fit() and predict()** - `69faa4d` (feat)
3. **Task 4: metrics.py** - `d8d0c8f` (feat)
4. **Task 5: validate_svm.py** - `f00c617` (feat)

## Files Created

- `src/kernels.py` - linear_kernel for dot product and Gram matrix
- `src/model.py` - SVM class with __init__, fit (cvxopt QP), predict (np.sign)
- `src/metrics.py` - accuracy_score, precision_score, recall_score, confusion_matrix
- `tests/validate_svm.py` - end-to-end validation on breast cancer data (99.12% accuracy)

## Decisions Made

- Added a 1e-8 ridge (`np.eye(n) * 1e-8`) to P to guarantee positive semi-definiteness, preventing cvxopt solver failures on borderline cases.
- Bias is computed using only "free" support vectors (0 < alpha < C), which lie exactly on the margin — this gives a more numerically stable bias estimate. Falls back to all support vectors if no free ones exist.
- Used `cvxopt.solvers.options["show_progress"] = False` at module level so all downstream code is silent by default.
- predict() uses the stored w and b vectors rather than re-expanding the kernel sum, giving O(d) instead of O(n_sv * d) prediction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing cvxopt dependency**
- **Found during:** Pre-execution environment check
- **Issue:** `import cvxopt` raised ModuleNotFoundError — not yet installed
- **Fix:** Ran `uv add cvxopt` (installed cvxopt==1.3.3)
- **Files modified:** pyproject.toml, uv.lock
- **Verification:** `uv run python -c "import cvxopt"` succeeded
- **Committed in:** included in task 2 commit (69faa4d) via pyproject.toml

**2. [Rule 2 - Missing Critical] Added 1e-8 ridge to P matrix**
- **Found during:** Task 2 (SVM fit implementation)
- **Issue:** Floating-point accumulation in the outer-product Gram matrix can produce a matrix that cvxopt considers not positive semi-definite, causing solver failure
- **Fix:** Added `P_np += 1e-8 * np.eye(n)` before converting to cvxopt.matrix
- **Files modified:** src/model.py
- **Verification:** Solver returns status='optimal' on both toy and full breast cancer datasets
- **Committed in:** 69faa4d

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing critical for numerical stability)
**Impact on plan:** Both fixes required for correctness. No scope creep.

## Issues Encountered

None beyond the deviations above.

## Known Stubs

None — all data paths are wired end-to-end. validate_svm.py uses real data.

## Next Phase Readiness

- Linear SVM pipeline is fully operational and validated at 99.12% test accuracy
- SVM class API (fit/predict) is scikit-learn compatible — ready for kernel extensions (RBF, Polynomial)
- metrics.py functions are reusable for cross-kernel benchmarking in Phase 2
- The `kernel=` parameter in SVM.__init__ is validated but only 'linear' is implemented — phase 2 can extend without breaking existing code

---
*Phase: 01-foundation-linear-svm*
*Completed: 2026-05-14*
