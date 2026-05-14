# Phase 1: Foundation & Linear SVM - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase establishes the project foundation by building a manual data preprocessing pipeline and implementing the core SVM class using the Dual Lagrangian formulation. It focuses on solving the dual problem for a linear kernel using the `cvxopt` QP solver.

</domain>

<decisions>
## Implementation Decisions

### SVM Class Interface
- **D-01:** Mirror the `scikit-learn` API. The SVM class will implement `fit(X, y)` and `predict(X)` methods for intuitive usage and compatibility with standard evaluation patterns.

### Data Pipeline
- **D-02:** Manual NumPy implementation for preprocessing. Feature scaling (Standardization) and label encoding ($\{0,1\} \to \{-1,1\}$) will be implemented using NumPy logic rather than high-level libraries to maintain the "manual implementation" theme.

### Mathematical Logic
- **D-03:** Hybrid Weight/Bias Handling. For the Linear SVM, the weight vector $w$ and bias $b$ will be explicitly calculated and stored after optimization for efficient prediction. However, the class structure will maintain support for support-vector-based prediction to accommodate non-linear kernels in Phase 2.

### Solver Constraints
- **D-04:** Standard Soft Margin ($C=1.0$). The solver will default to a soft margin approach, which is necessary for the likely non-linearly separable nature of the Breast Cancer dataset in its raw feature space.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Specifications
- [.planning/PROJECT.md](file:///.planning/PROJECT.md) — Core vision and mathematical focus (Dual Lagrangian).
- [.planning/REQUIREMENTS.md](file:///.planning/REQUIREMENTS.md) — Acceptance criteria for Data Pipeline and SVM Core Logic.

### Data & Architecture
- [.planning/codebase/ARCHITECTURE.md](file:///.planning/codebase/ARCHITECTURE.md) — Monolithic script structure.
- [.planning/codebase/STACK.md](file:///.planning/codebase/STACK.md) — Python 3.12, NumPy, and `cvxopt` requirements.

</canonical_refs>

<deferred>
## Deferred Ideas

- **Non-linear Kernels**: RBF and Polynomial implementations are explicitly deferred to Phase 2.
- **Cross-Validation**: Grid search and hyperparameter tuning for $C$ are deferred to a later iteration (v2 requirements).
</deferred>
