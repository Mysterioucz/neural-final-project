# Research: Pitfalls

## Common Mistakes
- **Scaling Gaps**: Applying SVM to raw data (e.g., area vs smoothness) will lead to the solver focusing only on high-magnitude features.
- **Label Encoding**: Using 0/1 for labels instead of -1/1. The Dual Lagrangian math assumes $y_i \in \{-1, 1\}$.
- **Numerical Instability**: The Gram matrix $P$ must be positive semi-definite. Small numerical errors can make it singular; adding a tiny $\epsilon \cdot I$ can help.
- **Overfitting RBF**: Setting $\gamma$ too high creates "islands" around individual points.
- **Support Vector Threshold**: Testing $\alpha_i > 0$ strictly is often prone to floating point error; use $\alpha_i > 1e-5$.

## Prevention
- **Standardization**: Always use `StandardScaler` logic before training.
- **Matrix Verification**: Check the rank and eigenvalues of the Gram matrix if the solver fails to converge.
