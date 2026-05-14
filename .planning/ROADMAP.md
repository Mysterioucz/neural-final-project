# Roadmap: SVM Lagrangian Classification

## Overview

This project implements a custom Support Vector Machine (SVM) using the Dual Lagrangian formulation. We will progress from building a robust data pipeline and foundation with a linear kernel to expanding into non-linear classification using RBF and Polynomial kernels, concluding with a comprehensive performance analysis.

## Phases

- [x] **Phase 1: Foundation & Linear SVM** - Data pipeline setup and baseline linear SVM implementation.
- [ ] **Phase 2: Kernel Expansion & Analysis** - Non-linear kernel implementations and comparative evaluation.

## Phase Details

### Phase 1: Foundation & Linear SVM
**Goal**: Build the end-to-end data pipeline and implement the core SVM Dual formulation with a linear kernel.
**Depends on**: Nothing
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, SVM-01, SVM-02, SVM-03, SVM-04, KERN-01, EVAL-01, EVAL-02
**Success Criteria**:
  1. Breast cancer data is correctly loaded, scaled, and split.
  2. The SVM Dual problem is successfully mapped and solved via `cvxopt`.
  3. Predictions are made using support vectors and Lagrange multipliers.
  4. Accuracy and Confusion Matrix are generated for the linear baseline.
**Plans**: 2 plans

Plans:
- [x] 01-01: Data preprocessing and transformation pipeline. (completed 2026-05-14, 2min)
- [x] 01-02: SVM Dual formulation logic and `cvxopt` integration. (completed 2026-05-14, 2min, 99.12% test accuracy)

### Phase 2: Kernel Expansion & Analysis
**Goal**: Implement advanced kernels and perform a comparative analysis of model performance.
**Depends on**: Phase 1
**Requirements**: KERN-02, KERN-03, EVAL-03
**Success Criteria**:
  1. RBF and Polynomial kernels are implemented and integrated into the SVM class.
  2. Model can switch between kernels via configuration.
  3. A final report compares Accuracy, Precision, and Recall across all three kernels.
**Plans**: 1 plan

Plans:
- [ ] 02-01: Implementation of RBF/Polynomial kernels and final comparative report.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Linear SVM | 2/2 | Complete | 2026-05-14 |
| 2. Kernel Expansion & Analysis | 0/1 | Not started | - |
