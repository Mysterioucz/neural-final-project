---
phase: 01-foundation-linear-svm
padded_phase: "01"
fixed_at: 2026-05-14T00:00:00Z
review_path: .planning/phases/01-foundation-linear-svm/01-REVIEW.md
iteration: 1
fix_scope: critical_warning
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-05-14
**Source review:** .planning/phases/01-foundation-linear-svm/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (2 Critical, 5 Warning)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Silent NaN predictions when no support vectors survive the alpha threshold

**Files modified:** `src/model.py`
**Commit:** 49c6aa0
**Applied fix:** Added `if not np.any(sv_mask): raise RuntimeError(...)` immediately after `sv_mask = alphas_all > self._ALPHA_THRESHOLD` in `fit()`, before any support vector extraction. This surfaces the degenerate case as an explicit error rather than silently producing NaN predictions.

---

### CR-02: Test fixture fits scaler on entire dataset before split — leaks test statistics into scaler

**Files modified:** `tests/test_data_pipeline.py`
**Commit:** 3c8c3b1
**Applied fix:** Rewrote `pipeline_data` fixture to split first (using `train_test_split` with `test_size=0.2, random_state=42`), then call `scaler.fit_transform(X_train_raw)` and `scaler.transform(X_test_raw)`. Fixture now returns `(X_train, X_test, y_train, y_test)` instead of the previous `(X_scaled, y_encoded)` tuple. Updated all 6 consuming test methods to unpack the new 4-tuple. Updated `TestManualStandardScaler` to assert mean/std on `X_train` only, and added `test_test_set_stats_differ_from_train` which asserts that `X_test` column-wise mean and std deviate from 0/1 by more than 1e-3 (proving the scaler was not re-fit on the test set). Fixed `TestTrainTestSplit.test_split_shapes_on_actual_data` to follow the correct split-first, then fit-scaler-on-train workflow.

---

### WR-01: `predict()` maps decision-boundary samples to 0.0, not a valid class label

**Files modified:** `src/model.py`
**Commit:** 8f979b0
**Applied fix:** Replaced `return np.sign(decision)` with a two-step return: `raw = np.sign(decision)` followed by `return np.where(raw == 0.0, 1.0, raw)`. Samples exactly on the decision boundary now map to +1.0 (positive class tie-break), with a comment documenting the intent.

---

### WR-02: `train_test_split` does not validate that X and y have the same number of samples

**Files modified:** `src/data_utils.py`
**Commit:** 70eb52d
**Applied fix:** Added an explicit `if X_arr.shape[0] != y_arr.shape[0]: raise ValueError(...)` check after converting inputs to arrays in `train_test_split`, before computing `n_samples`. The error message reports both counts.

---

### WR-03: `SVM.fit()` accepts single-class input but produces a degenerate model

**Files modified:** `src/model.py`
**Commit:** b7260e9
**Applied fix:** Replaced the single-check `if set(np.unique(y)) - {1.0, -1.0}` with a two-step validation: first check for unknown labels using the walrus-operator pattern, then check `if not ({1.0, -1.0} <= unique_labels)` to require both classes present. Each check raises a descriptive `ValueError`.

---

### WR-04: Redundant condition in `free_mask` (first clause is always True)

**Files modified:** `src/model.py`
**Commit:** df3e75b
**Applied fix:** Replaced the two-clause conjunction `(self.alphas_ > self._ALPHA_THRESHOLD) & (self.alphas_ < self.C - self._ALPHA_THRESHOLD)` with the single clause `self.alphas_ < self.C - self._ALPHA_THRESHOLD`. Added a comment explaining that the lower bound is already guaranteed by `sv_mask`.

---

### WR-05: `test_size` boundary rounding silently produces an empty test or train set

**Files modified:** `src/data_utils.py`
**Commit:** bddb62f
**Applied fix:** After computing `n_test` and `n_train` from rounding, added `if n_test == 0 or n_train == 0: raise ValueError(...)` with a message that reports `test_size`, `n_samples`, `n_train`, and `n_test` to make the issue immediately diagnosable.

---

## Post-fix Verification

Both verification checks passed after all fixes were applied:

- `python3 -m pytest tests/test_data_pipeline.py -q` — **23 passed**
- `uv run python tests/validate_svm.py` — **Accuracy 99.12% >= 90% threshold: PASS**

---

_Fixed: 2026-05-14_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
