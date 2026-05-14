# Phase 02: Kernel Expansion & Analysis - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase expands the SVM implementation to support non-linear classification via the "Kernel Trick." It involves implementing RBF and Polynomial kernels, refactoring the SVM class to handle non-linear prediction paths (kernel expansion), and performing a comparative analysis across all kernels on the breast cancer dataset.

</domain>

<decisions>
## Implementation Decisions

### Kernel Architecture & Parameters
- **D-01: Specific Init Arguments.** Kernel parameters (`gamma`, `degree`, `coef0`) will be added as explicit arguments to the `SVM.__init__` method, mirroring the `scikit-learn` API.
- **D-02: Centralized Kernel Registry.** Kernel implementations and a factory/registry will be housed in `src/kernels.py` to keep the core SVM logic in `src/model.py` clean.
- **D-03: Parameter Validation.** The class will issue a warning if parameters irrelevant to the selected kernel are provided (e.g., `degree` for `linear`).
- **D-04: Immutable Configuration.** Model parameters are locked after initialization to ensure consistency between `fit()` and `predict()`.

### Non-linear Prediction Path
- **D-05: Internal Branching in Predict.** The `predict()` method will use an internal branch: $O(d)$ dot-product for `linear` and kernel-expansion $\sum \alpha_i y_i K(x_i, x) + b$ for non-linear kernels.
- **D-06: Attribute Safety.** The weight vector `w_` will be set to `None` for non-linear kernels to prevent misuse (as it only exists in the high-dimensional feature space).
- **D-07: Robust Bias Calculation.** Bias $b$ will be calculated as the mean over all "free" support vectors ($0 < \alpha < C$) for numerical stability.
- **D-08: Vectorized Evaluation.** Kernel evaluations during prediction must be vectorized using NumPy broadcasting to ensure performance on the full test set.

### Defaults & Evaluation
- **D-09: Smart Defaults.** Default $\gamma$ will be set to `'scale'` ($1 / (n\_features \times X.var())$). Default `degree` is 3, and `coef0` is 0.
- **D-10: Comparative Reporting.** The final script will output a tabular summary of Accuracy, Precision, Recall, and Training Time to the console and save the raw data to a CSV file.
- **D-11: Recommendation Logic.** The analysis will include an automated "Best Kernel" recommendation based on the highest accuracy achieved.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Specs
- [.planning/PROJECT.md](file:///.planning/PROJECT.md) — Dual Lagrangian mathematical focus.
- [.planning/REQUIREMENTS.md](file:///.planning/REQUIREMENTS.md) — KERN-02, KERN-03, and EVAL-03 requirements.

### Codebase
- [src/model.py](file:///src/model.py) — Existing SVM implementation to be refactored.
- [src/kernels.py](file:///src/kernels.py) — Existing linear kernel implementation.

</canonical_refs>

<deferred>
## Deferred Ideas

- **Grid Search**: Systematic hyperparameter tuning remains in v2 requirements.
- **Visualization**: Decision boundary plotting is deferred to v2.

</deferred>
