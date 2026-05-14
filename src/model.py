"""
Support Vector Machine using the Dual Lagrangian formulation.

Solves the soft-margin SVM dual QP problem:
    min_alpha  (1/2) * alpha^T * P * alpha + q^T * alpha
    subject to  G * alpha <= h  and  A * alpha = b

where P = (y * y^T) * K (Gram matrix scaled by labels).
Uses `cvxopt.solvers.qp` for the quadratic programming step.
"""

import numpy as np
import cvxopt

from src.kernels import linear_kernel


# Suppress cvxopt solver output
cvxopt.solvers.options["show_progress"] = False


class SVM:
    """
    Binary Support Vector Machine via Dual Lagrangian formulation.

    Supports a linear kernel by default.  After fitting, the decision
    boundary is stored as weight vector ``w_`` and bias ``b_`` for
    efficient O(d) prediction (no need to iterate over support vectors).

    Parameters
    ----------
    C : float, optional (default=1.0)
        Regularisation parameter.  Smaller values → wider margin, more
        misclassifications allowed.  Larger values → harder margin.
    kernel : str, optional (default='linear')
        Kernel type.  Currently only 'linear' is supported.

    Attributes
    ----------
    support_vectors_ : np.ndarray of shape (n_sv, n_features)
        Training samples that are support vectors (alpha > 1e-5).
    support_vector_labels_ : np.ndarray of shape (n_sv,)
        Class labels of the support vectors.
    alphas_ : np.ndarray of shape (n_sv,)
        Lagrange multipliers for the support vectors.
    w_ : np.ndarray of shape (n_features,)
        Weight vector (valid only for linear kernel).
    b_ : float
        Bias term.
    """

    _ALPHA_THRESHOLD = 1e-5  # Minimum alpha to be considered a support vector

    def __init__(self, C: float = 1.0, kernel: str = "linear") -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}.")
        if kernel != "linear":
            raise NotImplementedError(f"Kernel '{kernel}' is not yet supported.")
        self.C = C
        self.kernel = kernel

        # Attributes set after fit()
        self.support_vectors_: np.ndarray | None = None
        self.support_vector_labels_: np.ndarray | None = None
        self.alphas_: np.ndarray | None = None
        self.w_: np.ndarray | None = None
        self.b_: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_qp_matrices(self, X: np.ndarray, y: np.ndarray):
        """Build the cvxopt QP matrices for the dual SVM problem.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)  — labels in {+1, -1}

        Returns
        -------
        P, q, G, h, A, b : cvxopt.matrix objects (all 'd' type)
        """
        n = X.shape[0]

        # Gram matrix K[i,j] = x_i . x_j  (linear kernel)
        K = linear_kernel(X, X)  # shape (n, n)

        # P[i,j] = y_i * y_j * K[i,j]
        y_col = y.reshape(-1, 1)  # (n,1)
        P_np = (y_col @ y_col.T) * K  # (n, n)
        # Ensure positive semi-definite by adding a tiny ridge
        P_np += 1e-8 * np.eye(n)
        P = cvxopt.matrix(P_np, tc="d")

        # q = [-1, -1, ..., -1]^T  (n x 1)
        q_np = -np.ones((n, 1), dtype=np.float64)
        q = cvxopt.matrix(q_np, tc="d")

        # G = [-I; I]  (2n x n) → encodes 0 <= alpha <= C
        G_np = np.vstack([-np.eye(n), np.eye(n)])  # (2n, n)
        G = cvxopt.matrix(G_np, tc="d")

        # h = [0, ..., 0, C, ..., C]^T  (2n x 1)
        h_np = np.hstack([np.zeros(n), np.full(n, self.C)]).reshape(-1, 1)
        H = cvxopt.matrix(h_np, tc="d")

        # A = y^T  (1 x n)
        A_np = y.reshape(1, -1).astype(np.float64)
        A = cvxopt.matrix(A_np, tc="d")

        # b = 0.0  (1 x 1)
        b_qp = cvxopt.matrix(0.0, tc="d")

        return P, q, G, H, A, b_qp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "SVM":
        """
        Fit the SVM model using the dual QP formulation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training labels; must contain only +1 and -1.

        Returns
        -------
        self : SVM
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if set(np.unique(y)) - {1.0, -1.0}:
            raise ValueError("Labels must be +1 and -1 only.")

        n = X.shape[0]

        # ---- Build and solve QP ----------------------------------------
        P, q, G, h, A, b_qp = self._build_qp_matrices(X, y)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b_qp)

        if solution["status"] != "optimal":
            raise RuntimeError(
                f"cvxopt QP solver did not converge (status='{solution['status']}')."
            )

        # ---- Extract Lagrange multipliers --------------------------------
        alphas_all = np.array(solution["x"]).flatten()  # shape (n,)

        # ---- Identify support vectors ------------------------------------
        sv_mask = alphas_all > self._ALPHA_THRESHOLD
        if not np.any(sv_mask):
            raise RuntimeError(
                "No support vectors found (all alphas below threshold). "
                "Try increasing C or reducing _ALPHA_THRESHOLD."
            )
        self.support_vectors_ = X[sv_mask]          # (n_sv, d)
        self.support_vector_labels_ = y[sv_mask]    # (n_sv,)
        self.alphas_ = alphas_all[sv_mask]           # (n_sv,)

        # ---- Compute weight vector w -------------------------------------
        # w = sum_i alpha_i * y_i * x_i
        self.w_ = np.sum(
            (self.alphas_ * self.support_vector_labels_)[:, np.newaxis]
            * self.support_vectors_,
            axis=0,
        )  # shape (d,)

        # ---- Compute bias b ----------------------------------------------
        # Use only "free" support vectors: 0 < alpha < C (on the margin)
        free_mask = (self.alphas_ > self._ALPHA_THRESHOLD) & (
            self.alphas_ < self.C - self._ALPHA_THRESHOLD
        )
        if np.any(free_mask):
            free_sv = self.support_vectors_[free_mask]
            free_y = self.support_vector_labels_[free_mask]
            self.b_ = float(np.mean(free_y - free_sv @ self.w_))
        else:
            # Fall back to all support vectors if no free ones found
            self.b_ = float(
                np.mean(
                    self.support_vector_labels_ - self.support_vectors_ @ self.w_
                )
            )

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Uses the linear decision function: sign(X @ w + b).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, each element is +1.0 or -1.0.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if self.w_ is None or self.b_ is None:
            raise RuntimeError("SVM must be fitted before calling predict().")

        X = np.asarray(X, dtype=np.float64)
        decision = X @ self.w_ + self.b_
        return np.sign(decision)
