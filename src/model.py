"""
Support Vector Machine using the Dual Lagrangian formulation.

Solves the soft-margin SVM dual QP problem:
    min_alpha  (1/2) * alpha^T * P * alpha + q^T * alpha
    subject to  G * alpha <= h  and  A * alpha = b

where P = (y * y^T) * K (Gram matrix scaled by labels).
Uses `cvxopt.solvers.qp` for the quadratic programming step.

Supported kernels
-----------------
- 'linear' : O(d) prediction via weight vector w
- 'rbf'    : Gaussian kernel, prediction via kernel expansion
- 'poly'   : Polynomial kernel, prediction via kernel expansion
"""

import warnings

import numpy as np
import cvxopt

from src.kernels import get_kernel


# Suppress cvxopt solver output
cvxopt.solvers.options["show_progress"] = False


class SVM:
    """
    Binary Support Vector Machine via Dual Lagrangian formulation.

    Supports linear, RBF, and polynomial kernels.  For the linear kernel,
    prediction uses the efficient O(d) weight-vector path.  For non-linear
    kernels, prediction uses kernel expansion over support vectors:
        f(x) = sign(sum_i alpha_i * y_i * K(x_i, x) + b)

    Parameters
    ----------
    C : float, optional (default=1.0)
        Regularisation parameter.  Smaller values -> wider margin, more
        misclassifications allowed.  Larger values -> harder margin.
    kernel : str, optional (default='linear')
        Kernel type: 'linear', 'rbf', or 'poly'.
    gamma : float or str, optional (default='scale')
        Kernel coefficient for 'rbf' and 'poly'.
        - 'scale' : 1 / (n_features * X.var())
        - 'auto'  : 1 / n_features
        - float   : use directly as gamma value
        Ignored for 'linear' kernel.
    degree : int, optional (default=3)
        Degree of the polynomial kernel.  Ignored for other kernels.
    coef0 : float, optional (default=0.0)
        Independent term in the polynomial kernel.  Ignored for others.

    Attributes
    ----------
    support_vectors_ : np.ndarray of shape (n_sv, n_features)
        Training samples that are support vectors (alpha > _ALPHA_THRESHOLD).
    support_vector_labels_ : np.ndarray of shape (n_sv,)
        Class labels of the support vectors.
    alphas_ : np.ndarray of shape (n_sv,)
        Lagrange multipliers for the support vectors.
    w_ : np.ndarray of shape (n_features,) or None
        Weight vector — valid only for 'linear' kernel.  Set to None for
        non-linear kernels (w only exists in the high-dimensional feature space).
    b_ : float
        Bias term.
    gamma_ : float
        Resolved numeric gamma value used during fit.
    """

    _ALPHA_THRESHOLD = 1e-5  # Minimum alpha to be considered a support vector

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        gamma: float | str = "scale",
        degree: int = 3,
        coef0: float = 0.0,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}.")

        supported_kernels = {"linear", "rbf", "poly"}
        if kernel not in supported_kernels:
            raise ValueError(
                f"Unsupported kernel '{kernel}'. "
                f"Choose from {sorted(supported_kernels)}."
            )

        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            raise ValueError(
                f"gamma must be a float or one of 'scale'/'auto', got '{gamma}'."
            )
        if not isinstance(gamma, str):
            try:
                gamma_f = float(gamma)
            except (TypeError, ValueError):
                raise ValueError(
                    f"gamma must be a float or 'scale'/'auto', got {gamma!r}."
                )
            if gamma_f <= 0:
                raise ValueError(
                    f"gamma must be positive when given as a number, got {gamma}."
                )

        # Warn about parameters irrelevant to the selected kernel
        if kernel == "linear" and degree != 3:
            warnings.warn(
                "Parameter 'degree' is not used by the 'linear' kernel and will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if kernel == "linear" and coef0 != 0.0:
            warnings.warn(
                "Parameter 'coef0' is not used by the 'linear' kernel and will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if kernel == "rbf" and degree != 3:
            warnings.warn(
                "Parameter 'degree' is not used by the 'rbf' kernel and will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if kernel == "rbf" and coef0 != 0.0:
            warnings.warn(
                "Parameter 'coef0' is not used by the 'rbf' kernel and will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        self.C = C
        self.kernel = kernel
        self.gamma = gamma  # raw user-supplied value ('scale', 'auto', or float)
        self.degree = degree
        self.coef0 = coef0

        # Attributes set after fit()
        self.support_vectors_: np.ndarray | None = None
        self.support_vector_labels_: np.ndarray | None = None
        self.alphas_: np.ndarray | None = None
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.gamma_: float | None = None          # resolved numeric gamma
        self._kernel_func_ = None                  # resolved kernel callable

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """Resolve gamma from 'scale'/'auto'/float based on training data X."""
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                var = X.var()
                return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
            else:  # 'auto'
                return 1.0 / X.shape[1]
        return float(self.gamma)

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

        # Gram matrix using the resolved kernel function
        K = self._kernel_func_(X, X)  # shape (n, n)

        # P[i,j] = y_i * y_j * K[i,j]
        y_col = y.reshape(-1, 1)  # (n,1)
        P_np = (y_col @ y_col.T) * K  # (n, n)
        # Ensure positive semi-definiteness by adding a tiny ridge
        P_np += 1e-8 * np.eye(n)
        P = cvxopt.matrix(P_np, tc="d")

        # q = [-1, -1, ..., -1]^T  (n x 1)
        q_np = -np.ones((n, 1), dtype=np.float64)
        q = cvxopt.matrix(q_np, tc="d")

        # G = [-I; I]  (2n x n) -> encodes 0 <= alpha <= C
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
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array of shape (n_samples, n_features), "
                f"got shape {X.shape}."
            )
        y = np.asarray(y, dtype=np.float64)

        unique_labels = set(np.unique(y))
        if unknown := unique_labels - {1.0, -1.0}:
            raise ValueError(f"Unknown labels: {unknown}. Expected +1 and -1.")
        if not ({1.0, -1.0} <= unique_labels):
            raise ValueError("Training data must contain both classes (+1 and -1).")

        n = X.shape[0]

        # ---- Resolve gamma and build kernel function ----------------------
        self.gamma_ = self._resolve_gamma(X)
        self._kernel_func_ = get_kernel(
            self.kernel,
            gamma=self.gamma_,
            degree=self.degree,
            coef0=self.coef0,
        )

        # ---- Build and solve QP ------------------------------------------
        P, q, G, h, A, b_qp = self._build_qp_matrices(X, y)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b_qp)

        if solution["status"] != "optimal":
            raise RuntimeError(
                f"cvxopt QP solver did not converge (status='{solution['status']}')."
            )

        # ---- Extract Lagrange multipliers ---------------------------------
        alphas_all = np.array(solution["x"]).flatten()  # shape (n,)

        # ---- Identify support vectors -------------------------------------
        sv_mask = alphas_all > self._ALPHA_THRESHOLD
        if not np.any(sv_mask):
            raise RuntimeError(
                "No support vectors found (all alphas below threshold). "
                "Try increasing C or reducing _ALPHA_THRESHOLD."
            )
        self.support_vectors_ = X[sv_mask]          # (n_sv, d)
        self.support_vector_labels_ = y[sv_mask]    # (n_sv,)
        self.alphas_ = alphas_all[sv_mask]           # (n_sv,)

        # ---- Identify "free" support vectors (0 < alpha < C) -------------
        free_mask = self.alphas_ < self.C - self._ALPHA_THRESHOLD

        if self.kernel == "linear":
            # ---- Compute weight vector w (linear kernel only) ------------
            # w = sum_i alpha_i * y_i * x_i
            self.w_ = np.sum(
                (self.alphas_ * self.support_vector_labels_)[:, np.newaxis]
                * self.support_vectors_,
                axis=0,
            )  # shape (d,)

            # ---- Compute bias b using free SVs (fall back to all SVs) ----
            if np.any(free_mask):
                free_sv = self.support_vectors_[free_mask]
                free_y = self.support_vector_labels_[free_mask]
                self.b_ = float(np.mean(free_y - free_sv @ self.w_))
            else:
                self.b_ = float(
                    np.mean(
                        self.support_vector_labels_
                        - self.support_vectors_ @ self.w_
                    )
                )

        else:
            # ---- Non-linear kernel: w_ does not exist in input space -----
            self.w_ = None

            # ---- Compute bias b using free SVs via kernel expansion ------
            # For each free SV x_j:  b = y_j - sum_i alpha_i * y_i * K(x_i, x_j)
            if np.any(free_mask):
                free_sv = self.support_vectors_[free_mask]      # (n_free, d)
                free_y = self.support_vector_labels_[free_mask]  # (n_free,)
                # K between all SVs and free SVs: shape (n_sv, n_free)
                K_sv_free = self._kernel_func_(
                    self.support_vectors_, free_sv
                )
                # Decision values at free SVs (without bias): shape (n_free,)
                decision_at_free = (
                    (self.alphas_ * self.support_vector_labels_) @ K_sv_free
                )
                self.b_ = float(np.mean(free_y - decision_at_free))
            else:
                # Fallback: use all support vectors
                K_sv_sv = self._kernel_func_(
                    self.support_vectors_, self.support_vectors_
                )
                decision_at_sv = (
                    (self.alphas_ * self.support_vector_labels_) @ K_sv_sv
                )
                self.b_ = float(
                    np.mean(self.support_vector_labels_ - decision_at_sv)
                )

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for samples in X.

        For 'linear' kernel: uses the efficient O(d) decision function sign(X @ w + b).
        For non-linear kernels: uses kernel expansion
            sign(sum_i alpha_i * y_i * K(x_i, x) + b).

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
        if self.b_ is None or self.support_vectors_ is None:
            raise RuntimeError("SVM must be fitted before calling predict().")

        X = np.asarray(X, dtype=np.float64)

        if self.kernel == "linear":
            # O(d) linear path using weight vector
            decision = X @ self.w_ + self.b_
        else:
            # Kernel expansion over support vectors
            # K_sv_test shape: (n_sv, n_test)
            K_sv_test = self._kernel_func_(self.support_vectors_, X)
            # (n_sv,) @ (n_sv, n_test) -> (n_test,)
            decision = (
                (self.alphas_ * self.support_vector_labels_) @ K_sv_test + self.b_
            )

        raw = np.sign(decision)
        # np.sign(0.0) == 0.0 is not a valid class label.
        # Tie-break: samples exactly on the decision boundary map to +1.
        return np.where(raw == 0.0, 1.0, raw)
