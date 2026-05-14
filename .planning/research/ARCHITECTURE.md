# Research: Architecture

## System Structure
1. **Data Module**: Responsbile for loading `data.csv`, handling missing values, and applying feature scaling.
2. **Kernel Module**: Pure function definitions for $K(x, z)$.
3. **Solver Integration**: Mapping the SVM dual problem matrices ($P, q, G, h, A, b$) to the `cvxopt.solvers.qp` format.
4. **SVM Model Class**: 
   - `fit(X, y)`: Constructs matrices and runs the solver.
   - `predict(X)`: Uses support vectors and Lagrange multipliers to compute the decision function.
5. **Evaluation Pipeline**: Computes accuracy, precision, recall, and confusion matrix.

## Data Flow
`CSV` -> `DataFrame` -> `Scaled NumPy Array` -> `Gram Matrix` -> `Lagrange Multipliers (α)` -> `Support Vectors` -> `Predictions`.

## Build Order
1. Data loading and scaling.
2. Linear Kernel + SVM Basic (to verify solver integration).
3. RBF and Polynomial Kernels.
4. Evaluation and Visualization suite.
