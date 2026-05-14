# Phase 02: Kernel Expansion & Analysis - Discussion Log

**Date:** 2026-05-15
**Participants:** Antigravity (the agent), USER

## Gray Area: Kernel Parameter Strategy

**Q: How should these parameters be passed to the SVM class?**
- **Selected:** Specific arguments (mirroring scikit-learn).
- **Rationale:** High discoverability and compatibility with standard ML patterns.

**Q: Where should the kernel logic live?**
- **Selected:** Registry in `src/kernels.py`.
- **Rationale:** Maintains separation of concerns; `src/model.py` focuses on optimization logic.

**Q: How should we handle invalid parameter combinations?**
- **Selected:** Warning.
- **Rationale:** Informs the user of potential misconfiguration without being overly restrictive.

**Q: Should the kernel parameters be mutable after initialization?**
- **Selected:** No.
- **Rationale:** Ensures model state consistency between fitting and prediction.

## Gray Area: Prediction Path Refactor

**Q: How should predict() handle the different paths?**
- **Selected:** Internal branch.
- **Rationale:** Preserves the $O(d)$ optimization for linear kernels while enabling the "Kernel Trick" for others.

**Q: Should we keep the w_ attribute for non-linear kernels?**
- **Selected:** No (set to None).
- **Rationale:** Prevents confusion as the weight vector is not explicitly representable in the input space for non-linear kernels.

**Q: How should the bias b be calculated for non-linear kernels?**
- **Selected:** Mean over free support vectors.
- **Rationale:** Mathematically robust and numerically stable.

**Q: Performance Consideration: Vectorized kernel evaluation?**
- **Selected:** Yes.
- **Rationale:** Critical for performance on large test sets.

## Gray Area: Kernel Hyperparameter Defaults

**Q: What should be the default gamma for the RBF kernel?**
- **Selected:** 'scale' ($1 / (n\_features \times X.var())$).
- **Rationale:** Adapts to the scale of the dataset features.

**Q: What should be the default degree for the Polynomial kernel?**
- **Selected:** 3.
- **Rationale:** Industry-standard default for polynomial non-linearity.

**Q: What should be the default coef0?**
- **Selected:** 0.
- **Rationale:** Standard starting point for polynomial kernels.

**Q: Should we provide a utility to "auto-suggest" parameters?**
- **Selected:** No.
- **Rationale:** Keep the implementation focused on the core math for now.

## Gray Area: Comparative Report Format

**Q: How should the results be presented in the console?**
- **Selected:** Tabular summary.
- **Rationale:** Clear, concise comparison of metrics across models.

**Q: Should we save the comparison to a file?**
- **Selected:** CSV.
- **Rationale:** Allows for easy downstream analysis or plotting.

**Q: Should we include training time in the comparison?**
- **Selected:** Yes.
- **Rationale:** Important practical metric when comparing kernel complexity.

**Q: Should the final report include a "Best Kernel" recommendation?**
- **Selected:** Yes.
- **Rationale:** Provides an automated, data-driven conclusion to the analysis.
