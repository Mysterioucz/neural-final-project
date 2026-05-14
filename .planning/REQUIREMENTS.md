# Requirements: SVM Lagrangian Classification

**Defined:** 2026-05-14
**Core Value:** Transparent and mathematically rigorous classification using a custom-built SVM dual pipeline on real-world medical data.

## v1 Requirements

Requirements for the initial implementation using Dual Lagrangian formulation and standard QP solvers.

### Data Pipeline

- [ ] **DATA-01**: Load breast cancer data from `data/data.csv`.
- [ ] **DATA-02**: Implement feature scaling (Standardization) using `numpy`.
- [ ] **DATA-03**: Split data into training and testing sets.
- [ ] **DATA-04**: Encode target labels into the $\{-1, 1\}$ space required by SVM dual formulation.

### SVM Core Logic

- [ ] **SVM-01**: Implement the mapping of SVM Dual constraints to Quadratic Programming matrices ($P, q, G, h, A, b$).
- [ ] **SVM-02**: Integrate `cvxopt` solver to find optimal Lagrange multipliers ($\alpha$).
- [ ] **SVM-03**: Extract support vectors and calculate the weight vector $w$ (for linear) and bias $b$.
- [ ] **SVM-04**: Implement the decision function for prediction using support vectors and kernels.

### Kernels

- [ ] **KERN-01**: Implement Linear kernel (dot product).
- [ ] **KERN-02**: Implement RBF (Gaussian) kernel with configurable $\gamma$.
- [ ] **KERN-03**: Implement Polynomial kernel with configurable degree.

### Evaluation

- [ ] **EVAL-01**: Compute Accuracy, Precision, and Recall on the test set.
- [ ] **EVAL-02**: Generate a Confusion Matrix for model evaluation.
- [ ] **EVAL-03**: Create a comparison summary of results across all three kernels.

## v2 Requirements

Deferred improvements for future iterations.

### Visualization & Optimization

- **VIZ-01**: Visualize decision boundaries in a 2D space (using PCA or feature selection).
- **OPT-01**: Implement a simple grid-search or cross-validation loop for $C$ and kernel parameters.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom QP Solver | User preference to focus on SVM logic rather than low-level optimization algorithms. |
| `scikit-learn` SVC | The goal is a manual implementation of the Lagrangian formulation. |
| Multi-class SVM | The Breast Cancer dataset is a binary classification problem. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| SVM-01 | Phase 1 | Pending |
| SVM-02 | Phase 1 | Pending |
| SVM-03 | Phase 1 | Pending |
| SVM-04 | Phase 1 | Pending |
| KERN-01 | Phase 1 | Pending |
| KERN-02 | Phase 2 | Pending |
| KERN-03 | Phase 2 | Pending |
| EVAL-01 | Phase 1 | Pending |
| EVAL-02 | Phase 1 | Pending |
| EVAL-03 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-14*
*Last updated: 2026-05-14 after initial definition*
