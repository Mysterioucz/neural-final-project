# Research: Technology Stack (2026)

## Recommended Stack
- **Numerical Computing**: `numpy` (Standard for matrix operations)
- **Data Handling**: `pandas` (For loading and cleaning `data.csv`)
- **Optimization**: `cvxopt` (Prescriptive choice for Quadratic Programming solvers in Python)
- **Mathematical Alternative**: `scipy.optimize.minimize` (Fallback if `cvxopt` environment is restricted)
- **Visualization**: `matplotlib`, `seaborn` (For decision boundary plotting and metric visualization)

## Rationale
- `cvxopt` is specialized for convex optimization, providing significantly faster and more stable results for the SVM dual problem compared to general-purpose optimizers.
- `numpy` is essential for the kernel matrix (Gram matrix) calculations.

## Confidence Level
- **High**: Standard path for implementing SVM from scratch in academic and research settings.
