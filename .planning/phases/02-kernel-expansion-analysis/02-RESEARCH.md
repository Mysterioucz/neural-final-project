# Phase 02: Kernel Expansion & Analysis - Research

This document outlines the technical research for implementing RBF and Polynomial kernels and the refactor required to support the "Kernel Trick" in the SVM implementation.

## Domain Investigation

### Mathematical Formulations

The "Kernel Trick" allows Support Vector Machines to operate in high-dimensional feature spaces without explicitly computing the coordinates of the data in that space.

1. **Linear Kernel**:
   $$K(x, x') = \langle x, x' \rangle$$
   The simplest kernel, representing a standard dot product.

2. **Polynomial Kernel**:
   $$K(x, x') = (\gamma \langle x, x' \rangle + r)^d$$
   - `gamma` ($\gamma$): Scale factor for the dot product.
   - `coef0` ($r$): Independent term.
   - `degree` ($d$): Degree of the polynomial.

3. **RBF (Gaussian) Kernel**:
   $$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
   - `gamma` ($\gamma$): Controls the "width" of the Gaussian. A small $\gamma$ means a "smooth" boundary; a large $\gamma$ means a boundary that closely follows the training data.

### Default Hyperparameters

To maintain consistency with standard libraries like `scikit-learn`:
- **`gamma='scale'`**: $1 / (n\_features \times X.var())$
- **`gamma='auto'`**: $1 / n\_features$
- **`degree`**: 3
- **`coef0`**: 0.0

## Technical Implementation

### Vectorized Gram Matrix Computation

For high performance, the Gram matrix $K$ must be computed using vectorized NumPy operations.

1. **Linear**:
   ```python
   K = X1 @ X2.T
   ```

2. **Polynomial**:
   ```python
   K = (gamma * (X1 @ X2.T) + coef0) ** degree
   ```

3. **RBF**:
   Using the identity $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2\langle x, y \rangle$:
   ```python
   # sum(X**2) per row
   sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
   sq2 = np.sum(X2**2, axis=1)
   # Gram matrix
   K = np.exp(-gamma * (sq1 + sq2 - 2 * (X1 @ X2.T)))
   ```

### Kernel Trick Prediction

The decision function for non-linear kernels uses the support vectors and their associated Lagrange multipliers:
$$f(X_{test}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(x_i, X_{test}) + b\right)$$

In vectorized form:
```python
# (alphas * labels) shape (n_sv,)
# K(X_SV, X_test) shape (n_sv, n_test)
decision = (self.alphas_ * self.support_vector_labels_) @ K_sv_test + self.b_
return np.sign(decision)
```

## Codebase Integration

### `src/kernels.py` refactor
- Introduce a registry/factory to handle kernel selection.
- Implement `polynomial_kernel` and `rbf_kernel` functions.

### `src/model.py` refactor
- **`__init__`**: Accept `gamma`, `degree`, `coef0`. Store `self.kernel_func_`.
- **`_build_qp_matrices`**: Replace direct call to `linear_kernel` with the selected kernel function.
- **`fit`**: Calculate $b$ as the mean over free support vectors ($0 < \alpha < C$).
- **`predict`**:
  - If `linear`: Use optimized $X w + b$ path.
  - Otherwise: Compute $K(X_{SV}, X_{test})$ and use the kernel expansion formula.

## Validation Architecture

1. **Unit Tests**:
   - Verify kernel function outputs against manual calculations for small matrices.
   - Test 'scale' vs 'auto' gamma calculation.
2. **Integration Test**:
   - Train RBF SVM on a non-linearly separable dataset (e.g., XOR or circles) to verify it can learn non-linear boundaries where linear SVM fails.
3. **Analysis Validation**:
   - Ensure the comparison report script correctly aggregates metrics from multiple runs.
