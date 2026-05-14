"""
Evaluation metrics for binary classification.

All functions are implemented using NumPy only (no scikit-learn).
Positive class is assumed to be +1.0; negative class is -1.0.
"""

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the fraction of correct predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: float = 1.0) -> float:
    """
    Compute precision for the positive class.

    Precision = TP / (TP + FP).
    Returns 0.0 if the denominator is zero (no positive predictions).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : float, optional (default=1.0)
        The label considered as the positive class.

    Returns
    -------
    float
        Precision in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = float(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = float(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: float = 1.0) -> float:
    """
    Compute recall (sensitivity) for the positive class.

    Recall = TP / (TP + FN).
    Returns 0.0 if the denominator is zero (no actual positives).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : float, optional (default=1.0)
        The label considered as the positive class.

    Returns
    -------
    float
        Recall in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = float(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fn = float(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, pos_label: float = 1.0) -> np.ndarray:
    """
    Compute a 2x2 confusion matrix.

    Layout::

        [[TN, FP],
         [FN, TP]]

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : float, optional (default=1.0)
        The label considered as the positive class.

    Returns
    -------
    np.ndarray of shape (2, 2)
        Confusion matrix with integer counts.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    tn = int(np.sum((y_pred != pos_label) & (y_true != pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)
