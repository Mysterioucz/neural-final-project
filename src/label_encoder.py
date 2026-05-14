"""
Label encoder for the Wisconsin Breast Cancer dataset.

Converts string diagnosis labels to numeric values suitable for SVM training:
  'M' (Malignant) -> 1.0
  'B' (Benign)    -> -1.0
"""

import numpy as np


def encode_labels(y) -> np.ndarray:
    """
    Encode diagnosis labels into numeric values for SVM training.

    Parameters
    ----------
    y : array-like of str
        Label array containing 'M' and 'B' values (e.g. a pd.Series or np.ndarray).

    Returns
    -------
    np.ndarray
        Float64 numpy array with 1.0 for 'M' (Malignant) and -1.0 for 'B' (Benign).

    Raises
    ------
    ValueError
        If any label is not 'M' or 'B'.
    """
    y_arr = np.asarray(y)

    unknown = set(y_arr) - {'M', 'B'}
    if unknown:
        raise ValueError(f"Unknown labels encountered: {unknown}. Expected 'M' or 'B'.")

    encoded = np.where(y_arr == 'M', 1.0, -1.0)
    return encoded.astype(np.float64)
