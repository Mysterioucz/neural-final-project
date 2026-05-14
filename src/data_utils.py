"""
Dataset utility functions for SVM training.

Provides a NumPy-based train/test split without scikit-learn.
"""

import numpy as np


def train_test_split(X, y, test_size: float = 0.2, random_state: int | None = None):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
        Must be in the range (0, 1).
    random_state : int or None, optional (default=None)
        Seed for the random number generator. Pass an integer for
        reproducible output across multiple function calls.

    Returns
    -------
    X_train : np.ndarray of shape (n_train, n_features)
    X_test  : np.ndarray of shape (n_test, n_features)
    y_train : np.ndarray of shape (n_train,)
    y_test  : np.ndarray of shape (n_test,)

    Raises
    ------
    ValueError
        If test_size is not in the range (0, 1).
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, "
            f"got X: {X_arr.shape[0]}, y: {y_arr.shape[0]}."
        )

    n_samples = X_arr.shape[0]
    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]
