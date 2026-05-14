"""
Kernel functions for Support Vector Machine computation.

Provides kernel functions that compute similarity between feature vectors.
Used to build the Gram matrix for the QP dual formulation.

Supported kernels
-----------------
- 'linear'  : K(x, y) = <x, y>
- 'poly'    : K(x, y) = (gamma * <x, y> + coef0) ** degree
- 'rbf'     : K(x, y) = exp(-gamma * ||x - y||^2)
"""

import warnings
from typing import Callable

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


def polynomial_kernel(
    gamma: float,
    degree: int,
    coef0: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Return a polynomial kernel function with fixed hyperparameters.

    The kernel is defined as:
        K(x, y) = (gamma * <x, y> + coef0) ** degree

    Parameters
    ----------
    gamma : float
        Scale factor for the dot product.
    degree : int
        Degree of the polynomial.
    coef0 : float
        Independent (bias) term.

    Returns
    -------
    Callable
        A function ``k(x1, x2)`` that computes the kernel Gram matrix
        of shape (n_samples_1, n_samples_2).

    Examples
    --------
    >>> import numpy as np
    >>> k = polynomial_kernel(gamma=1.0, degree=2, coef0=0.0)
    >>> X1 = np.array([[1.0, 0.0]])
    >>> X2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> k(X1, X2)
    array([[1., 0.]])
    """

    def _kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
        x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
        return (gamma * (x1 @ x2.T) + coef0) ** degree

    return _kernel


def rbf_kernel(gamma: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Return an RBF (Gaussian) kernel function with a fixed gamma.

    The kernel is defined as:
        K(x, y) = exp(-gamma * ||x - y||^2)

    The squared distance is computed efficiently using the identity:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * <x, y>

    Parameters
    ----------
    gamma : float
        Bandwidth parameter.  Must be positive.  Controls the "width"
        of the Gaussian; a small gamma gives a smooth boundary, a large
        gamma gives a boundary that closely follows the training data.

    Returns
    -------
    Callable
        A function ``k(x1, x2)`` that computes the kernel Gram matrix
        of shape (n_samples_1, n_samples_2).

    Examples
    --------
    >>> import numpy as np
    >>> k = rbf_kernel(gamma=1.0)
    >>> X = np.array([[0.0, 0.0]])
    >>> k(X, X)
    array([[1.]])
    """

    def _kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
        x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * <x, y>
        sq1 = np.sum(x1 ** 2, axis=1).reshape(-1, 1)  # (n1, 1)
        sq2 = np.sum(x2 ** 2, axis=1)                  # (n2,)
        sq_dists = sq1 + sq2 - 2.0 * (x1 @ x2.T)       # (n1, n2)
        # Numerical clamp to avoid tiny negatives from floating-point error
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-gamma * sq_dists)

    return _kernel


# Parameters that belong exclusively to each kernel type
_KERNEL_PARAMS: dict[str, set[str]] = {
    "linear": set(),
    "poly": {"degree", "coef0"},
    "rbf": set(),
}


def get_kernel(
    kernel: str,
    gamma: float,
    degree: int = 3,
    coef0: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Kernel factory — returns the appropriate kernel callable.

    Issues ``UserWarning`` when parameters that are irrelevant to the
    requested kernel are explicitly provided (non-default values).

    Parameters
    ----------
    kernel : str
        One of 'linear', 'rbf', or 'poly'.
    gamma : float
        Kernel coefficient (already resolved from 'scale'/'auto' by SVM).
    degree : int, optional (default=3)
        Degree for the polynomial kernel (ignored for others).
    coef0 : float, optional (default=0.0)
        Independent term for the polynomial kernel (ignored for others).

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        A function that computes the Gram matrix for the given kernel.

    Raises
    ------
    ValueError
        If ``kernel`` is not one of the supported types.

    Examples
    --------
    >>> import numpy as np
    >>> k = get_kernel('rbf', gamma=0.5)
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> K = k(X, X)
    >>> K.shape
    (2, 2)
    """
    supported = {"linear", "rbf", "poly"}
    if kernel not in supported:
        raise ValueError(
            f"Unsupported kernel '{kernel}'. Choose from {sorted(supported)}."
        )

    # Warn about parameters that are irrelevant to the selected kernel
    kernel_params = _KERNEL_PARAMS.get(kernel, set())
    provided_non_defaults = {}
    if "degree" not in kernel_params and degree != 3:
        provided_non_defaults["degree"] = degree
    if "coef0" not in kernel_params and coef0 != 0.0:
        provided_non_defaults["coef0"] = coef0

    if provided_non_defaults:
        warnings.warn(
            f"Parameters {list(provided_non_defaults.keys())} are not used by the "
            f"'{kernel}' kernel and will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    if kernel == "linear":
        # Wrap linear_kernel so it always returns a 2-D Gram matrix
        def _linear(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
            x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
            return x1 @ x2.T

        return _linear

    if kernel == "rbf":
        return rbf_kernel(gamma)

    # kernel == 'poly'
    return polynomial_kernel(gamma=gamma, degree=degree, coef0=coef0)
