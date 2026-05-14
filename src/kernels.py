"""
Kernel functions for Support Vector Machine computation.

Provides kernel functions that compute similarity between feature vectors.
Used to build the Gram matrix for the QP dual formulation.
"""

import numpy as np


def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Compute the linear (dot product) kernel between two inputs.

    Handles both vector-vector and matrix-matrix cases:
    - If both inputs are 1-D vectors, returns the scalar dot product.
    - If either input is a 2-D matrix, returns the Gram matrix X1 @ X2.T
      of shape (n_samples_1, n_samples_2).

    Parameters
    ----------
    x1 : np.ndarray
        First input — shape (n_features,) or (n_samples_1, n_features).
    x2 : np.ndarray
        Second input — shape (n_features,) or (n_samples_2, n_features).

    Returns
    -------
    np.ndarray
        Dot product or Gram matrix depending on input dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> v1 = np.array([1.0, 2.0, 3.0])
    >>> v2 = np.array([4.0, 5.0, 6.0])
    >>> linear_kernel(v1, v2)  # scalar dot product
    32.0

    >>> X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> linear_kernel(X1, X2)  # 2x2 Gram matrix
    array([[17., 23.],
           [39., 53.]])
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.ndim == 1 and x2.ndim == 1:
        return np.dot(x1, x2)

    # At least one is 2-D — compute Gram matrix
    return x1 @ x2.T
