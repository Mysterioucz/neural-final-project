---
status: testing
phase: 02-kernel-expansion-analysis
source: [02-01-SUMMARY.md]
started: 2026-05-15T00:00:00Z
updated: 2026-05-15T00:00:00Z
---

## Current Test

number: 5
name: CSV report is produced
expected: |
  results/kernel_comparison.csv exists with header row + 3 data rows (one per kernel).
awaiting: user response

## Tests

### 1. RBF kernel trains and predicts
expected: |
  Run the setup block then: svm = SVM(kernel='rbf'); svm.fit(X_train, y_train); preds = svm.predict(X_test)
  Should complete without error and return an array of -1/+1 predictions.
  svm.w_ should be None (weight vector is not defined for non-linear kernels).
result: pass

### 2. Polynomial kernel trains and predicts
expected: |
  Run: svm = SVM(kernel='poly', degree=3, coef0=1.0); svm.fit(X_train, y_train); preds = svm.predict(X_test)
  Should complete without error. svm.w_ should be None.
  Changing degree (e.g., degree=2) should produce a different (but still valid) set of predictions.
result: pass

### 3. Kernel switching changes model behavior
expected: |
  Instantiating SVM with kernel='linear', kernel='rbf', kernel='poly' should each behave differently.
  Linear: svm.w_ is a non-None weight vector after fit.
  RBF / Poly: svm.w_ is None after fit.
  Passing an invalid kernel name like SVM(kernel='sigmoid') should raise a ValueError.
result: pass

### 4. Kernel comparison script runs end-to-end
expected: |
  Run: python scripts/compare_kernels.py
  Should print a formatted ASCII table showing Accuracy, Precision, Recall, training time,
  and support vector count for Linear, RBF, and Polynomial kernels.
  Should print a "Best Kernel" recommendation line at the end.
  Should exit with code 0 (no errors).
result: pass

### 5. CSV report is produced
expected: |
  After running compare_kernels.py, the file results/kernel_comparison.csv should exist.
  Opening it should show one header row and three data rows — one per kernel.
  Columns should include kernel name, accuracy, precision, recall, training time.
result: pass

## Summary

total: 5
passed: 4
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps

[none yet]
