# Phase 01 Research: Foundation & Linear SVM

## Overview
This research covers the mathematical and technical requirements for implementing a Support Vector Machine (SVM) using the Dual Lagrangian formulation and integrating it with the `cvxopt` Quadratic Programming (QP) solver.

## 1. CVXOPT Integration (QP Mapping)
To solve the Soft-Margin SVM dual problem:
$$\min_{\alpha} \frac{1}{2} \alpha^T (y y^T \odot K) \alpha - 1^T \alpha$$
subject to:
$$y^T \alpha = 0$$
$$0 \le \alpha \le C$$

The mapping to the `cvxopt.solvers.qp(P, q, G, h, A, b)` interface is as follows:

- **P**: `cvxopt.matrix` of shape $(n, n)$ where $P_{ij} = y_i y_j (x_i \cdot x_j)$. Note: $y_i y_j (x_i \cdot x_j)$ is the Gram matrix scaled by labels.
- **q**: `cvxopt.matrix` of shape $(n, 1)$ filled with $-1.0$.
- **G**: `cvxopt.matrix` of shape $(2n, n)$ combining $-I$ and $I$ (vertical concatenation of identity matrices) to represent $-\alpha \le 0$ and $\alpha \le C$.
- **h**: `cvxopt.matrix` of shape $(2n, 1)$ combining $n$ zeros and $n$ values of $C$.
- **A**: `cvxopt.matrix` of shape $(1, n)$ containing labels $y$ as floats.
- **b**: `cvxopt.matrix` of shape $(1, 1)$ containing $0.0$.

## 2. Dataset Specifics (Wisconsin Breast Cancer)
- **File**: `data/data.csv`
- **Structure**:
    - **ID**: First column (to be dropped).
    - **Target**: Second column `diagnosis` ('M' for Malignant, 'B' for Benign).
    - **Features**: 30 real-valued columns (radius, texture, perimeter, etc.).
- **Preprocessing Requirements**:
    - **Encoding**: Map 'M' $\to 1.0$ and 'B' $\to -1.0$.
    - **Scaling**: All 30 features have different units and scales; Standardization is mandatory for SVM convergence.

## 3. Implementation Details (NumPy)
- **Manual Standardization**: Implement $x_{scaled} = \frac{x - \mu}{\sigma}$ using `np.mean` and `np.std`. Ensure handling for $\sigma=0$ (though unlikely in this dataset).
- **SVM Logic**:
    - **Support Vectors**: $\alpha_i > 10^{-5}$.
    - **Weights ($w$)**: $w = \sum_{i \in SV} \alpha_i y_i x_i$.
    - **Bias ($b$)**: $b = \frac{1}{|S|} \sum_{s \in S} (y_s - w^T x_s)$ where $S$ is the set of support vectors with $0 < \alpha_s < C$.

## 4. Validation Architecture

### Verification Strategy
- **Solver Status**: Ensure `cvxopt` returns `status: 'optimal'`. Any other status indicates a mapping error or non-convergence.
- **Toy Dataset**: Test first on a small, 2D linearly separable dataset to visualize the decision boundary and margin.
- **Mathematical Integrity**: Verify that $\sum \alpha_i y_i \approx 0$ after optimization.

### Unit Test Targets
- **StandardScaler**: Verify mean/std logic against manual calculations.
- **Matrix Mapping**: Unit test the creation of $P, q, G, h, A, b$ for a small $2 \times 2$ input.
- **Prediction**: Ensure the sign of the decision function correctly maps to class labels.

### Benchmarking
- **Accuracy**: Expected accuracy on the Wisconsin Breast Cancer dataset is $>95\%$.
- **Comparison**: Compare results with `scikit-learn.svm.SVC(kernel='linear', C=1.0)` to ensure implementation parity.

## References
- [CVXOPT QP Documentation](https://cvxopt.org/userguide/coneprog.html#quadratic-programming)
- [SVM Dual Formulation (Stanford CS229)](https://cs229.stanford.edu/notes2021fall/cs229-notes3.pdf)
