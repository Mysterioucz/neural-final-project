# SVM Lagrangian Classification

## What This Is

A Python classification project to predict breast cancer diagnosis using a manually implemented **Dual Lagrangian formulation** of Support Vector Machines (SVM). The project focuses on the mathematical formulation of the SVM dual problem, supporting multiple kernels (Linear, RBF, Polynomial) while utilizing a dedicated mathematical optimization library for solving the Quadratic Programming (QP) problem.

## Core Value

Transparent and mathematically rigorous classification using a custom-built SVM dual pipeline on real-world medical data.

## Requirements

### Validated

- ✓ Codebase mapping complete — initial
- ✓ Project structure and environment initialization — initial

### Active

- [ ] Data preprocessing pipeline (scaling, encoding, train/test split)
- [ ] Custom SVM class implementing the Dual Lagrangian formulation
- [ ] Implementation of Linear, RBF, and Polynomial kernel functions
- [ ] Integration with a QP optimizer library (e.g., `cvxopt` or `scipy.optimize`)
- [ ] Comprehensive evaluation (Accuracy, Confusion Matrix)
- [ ] Performance comparison between different kernels

### Out of Scope

- **Custom QP Solver from scratch**: User chose to use a library for the optimization heavy lifting to focus on the SVM formulation logic.
- **Using `scikit-learn` for SVM training**: The goal is a manual implementation of the math, not using a black-box model.

## Context

- **Dataset**: Wisconsin Breast Cancer dataset located in `data/data.csv`.
- **Environment**: Python 3.12 managed by `uv`.
- **Mathematical Focus**: Solving the dual problem $\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$ subject to $0 \le \alpha_i \le C$ and $\sum \alpha_i y_i = 0$.

## Constraints

- **Tech Stack**: Python 3.12+, NumPy, Pandas.
- **Optimization**: Must use a mathematical QP solver (per user preference B).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| **Dual Formulation** | Allows the "Kernel Trick" to be easily integrated for non-linear classification. | — Pending |
| **B: Library Optimizer** | Balancing the pedagogical value of implementing the SVM logic with the stability of a production-grade QP solver. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-14 after initialization*
