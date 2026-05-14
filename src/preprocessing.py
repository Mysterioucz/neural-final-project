"""
Manual feature standardization for SVM preprocessing.

Implements a StandardScaler using only NumPy, computing per-feature
mean and standard deviation, then normalizing to zero mean and unit variance.
"""

import numpy as np


class ManualStandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Uses only NumPy (no scikit-learn). A small epsilon is added to the
    standard deviation to prevent division by zero for constant features.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean computed during fit.
    std_ : np.ndarray of shape (n_features,)
        Per-feature standard deviation (with epsilon) computed during fit.

    Example
    -------
    >>> scaler = ManualStandardScaler()
    >>> X_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    """

    _EPSILON = 1e-8  # Prevents division by zero for constant features

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X) -> "ManualStandardScaler":
        """
        Compute per-feature mean and standard deviation from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : ManualStandardScaler
        """
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0) + self._EPSILON
        return self

    def transform(self, X) -> np.ndarray:
        """
        Standardize features using the pre-computed mean and std.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Standardized data with approximately zero mean and unit variance.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ManualStandardScaler must be fitted before calling transform().")

        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X) -> np.ndarray:
        """
        Fit to data, then transform it.

        Equivalent to calling fit(X).transform(X) but slightly more efficient.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to fit and transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Standardized data.
        """
        return self.fit(X).transform(X)
