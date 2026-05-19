"""
Binary Logistic Regression with gradient descent (HW4 Part 2).

From-scratch status
-------------------
NumPy vectorised operations only.  No scipy.optimize, no sklearn.  The
sigmoid is a standard mathematical function (analogous to numpy.exp) — it
is not a model provided by any library.
"""

import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid.

    For z >= 0: sigma = 1 / (1 + exp(-z))
    For z <  0: sigma = exp(z) / (1 + exp(z))

    Both forms are algebraically identical but each branch avoids overflow
    in its respective half of the real line:
      - z >> 0: exp(-z) → 0,  so the first branch is safe.
      - z << 0: exp(z)  → 0,  so the second branch is safe.
    The naive form 1/(1+exp(-z)) overflows to inf when z < -709 on float64.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg]  = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent.

    Model:
        P(y=1 | x) = σ(wᵀx + b)  where σ is the sigmoid function.

    Loss (binary cross-entropy, averaged):
        L = -(1/N) Σ [y_i log(p_i + ε) + (1-y_i) log(1-p_i + ε)]

    A tiny epsilon (1e-15) is added inside both log arguments to prevent
    log(0) when predictions saturate to exactly 0 or 1 in finite precision.

    Gradient (vectorised, no Python loop over samples):
        ∇w L = (1/N) Xᵀ(p - y)  +  λ w    (L2 term on weights, not bias)
        ∇b L = (1/N) Σ(p_i - y_i)

    Derivation: the BCE gradient is particularly clean for sigmoid outputs.
    Let r = p - y (residuals). Then:
        ∂L/∂w_j = (1/N) Σ_i r_i x_{ij}  →  (1/N) Xᵀ r

    Early stopping: training halts when |L(t-1) - L(t)| < tol, meaning the
    loss has converged to within numerical tolerance.  This prevents wasted
    epochs after the optimum is effectively reached.

    Parameters
    ----------
    learning_rate : float
        Gradient descent step size (must be > 0).
    n_epochs : int
        Maximum number of gradient descent iterations (must be >= 1).
    tol : float
        Early-stopping threshold on the absolute loss change between
        consecutive epochs (must be >= 0).  Set to 0 to disable.
    regularization : float
        L2 penalty coefficient λ >= 0.  Adds λ‖w‖²/2 to the loss and
        λ w to the weight gradient.  Bias is NOT regularised.
    verbose : bool
        If True, print loss every 100 epochs.
    random_state : int
        Seed for weight initialisation reproducibility.
    """

    _EPS = 1e-15   # numerical guard inside log(·)

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        tol: float = 1e-6,
        regularization: float = 0.0,
        verbose: bool = False,
        random_state: int = 42,
    ):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}")
        if regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization}")

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.tol = tol
        self.regularization = regularization
        self.verbose = verbose
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = None
        self.n_iter_ = None
        self.classes_ = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train by minimising binary cross-entropy with gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — binary labels (any two values).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"LogisticRegression requires exactly 2 classes, "
                f"got {len(unique_classes)}: {unique_classes}"
            )
        self.classes_ = unique_classes

        # Map to 0/1 internally regardless of original label encoding
        y01 = (y == self.classes_[1]).astype(float)

        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        w = rng.normal(0.0, 0.01, d)
        b = 0.0

        self.loss_history_ = []
        prev_loss = np.inf

        for epoch in range(self.n_epochs):
            z = X @ w + b
            p = _sigmoid(z)

            # Loss (BCE + optional L2)
            bce = -(
                y01 * np.log(p + self._EPS) + (1.0 - y01) * np.log(1.0 - p + self._EPS)
            ).mean()
            l2_term = 0.5 * self.regularization * (w @ w) if self.regularization > 0 else 0.0
            loss = bce + l2_term

            if not np.isfinite(loss):
                raise ValueError(
                    f"Loss became non-finite at epoch {epoch} "
                    f"(loss={loss}). Try a smaller learning_rate."
                )

            self.loss_history_.append(float(loss))

            if self.verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch:>5d}  loss={loss:.6f}")

            # Early stopping
            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = epoch + 1
                break
            prev_loss = loss

            # Gradients (vectorised)
            r = p - y01                                    # residuals (N,)
            grad_w = (X.T @ r) / n + self.regularization * w
            grad_b = r.mean()

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        else:
            self.n_iter_ = self.n_epochs

        self.coef_ = w
        self.intercept_ = float(b)
        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        """Return class probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0: P(y = classes_[0] | x)
            Column 1: P(y = classes_[1] | x)
            Rows sum to 1.
        """
        self._check_fitted()
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X, threshold: float = 0.5):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        threshold : float
            Decision threshold on P(y=classes_[1] | x).
            Default 0.5; lower it to increase recall for the positive class.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Values drawn from ``self.classes_``.
        """
        self._check_fitted()
        p1 = self.predict_proba(X)[:, 1]
        return np.where(p1 >= threshold, self.classes_[1], self.classes_[0])

    # ------------------------------------------------------------------
    # decision_function
    # ------------------------------------------------------------------

    def decision_function(self, X):
        """Return raw logits z = X @ coef_ + intercept_.

        Directly comparable to the least-squares classifier's scores:
        both are linear in x, but the two methods optimise different
        objectives, so the scale and orientation may differ.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        z : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict() or predict_proba().")


# ---------------------------------------------------------------------------
# Sanity test (synthetic data only — no real dataset)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # Two well-separated Gaussian blobs, 100 points each
    X0 = rng.normal(loc=[-3, -3], scale=1.0, size=(100, 2))
    X1 = rng.normal(loc=[+3, +3], scale=1.0, size=(100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)

    # ---- basic fit / convergence ----
    clf = LogisticRegression(learning_rate=0.1, n_epochs=2000, tol=1e-8, verbose=False)
    clf.fit(X, y)

    # Loss must be monotonically non-increasing (allow tiny fp blips ≤ 1e-9)
    hist = np.array(clf.loss_history_)
    diffs = np.diff(hist)
    max_increase = diffs.max() if len(diffs) > 0 else 0.0
    assert max_increase < 1e-9, f"Loss increased by {max_increase:.2e} — not monotone"
    print(f"[PASS] Loss monotone. Epochs run: {clf.n_iter_}, "
          f"final loss: {clf.loss_history_[-1]:.6f}")

    # Training accuracy > 95%
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.95, f"Training accuracy too low: {acc:.4f}"
    print(f"[PASS] Training accuracy: {acc:.4f}")

    # predict_proba rows sum to 1
    proba = clf.predict_proba(X)
    assert proba.shape == (200, 2), f"proba shape wrong: {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Rows don't sum to 1: max dev {np.abs(row_sums-1).max()}"
    print("[PASS] predict_proba rows sum to 1")

    # Extreme test points: far positive → proba > 0.99, far negative → proba < 0.01
    x_far_pos = np.array([[20.0, 20.0]])
    x_far_neg = np.array([[-20.0, -20.0]])
    p_pos = clf.predict_proba(x_far_pos)[0, 1]
    p_neg = clf.predict_proba(x_far_neg)[0, 1]
    assert p_pos > 0.99, f"Far positive point: P(y=1)={p_pos:.6f} expected >0.99"
    assert p_neg < 0.01, f"Far negative point: P(y=1)={p_neg:.6f} expected <0.01"
    print(f"[PASS] Extreme points: P(y=1|far+)={p_pos:.6f}, P(y=1|far-)={p_neg:.6f}")

    # Sigmoid stability: z = ±1000 must not produce NaN
    z_extreme = np.array([1000.0, -1000.0])
    s = _sigmoid(z_extreme)
    assert not np.any(np.isnan(s)), f"Sigmoid produced NaN: {s}"
    assert np.isclose(s[0], 1.0) and np.isclose(s[1], 0.0), f"Sigmoid wrong at extremes: {s}"
    print(f"[PASS] Sigmoid stability: σ(1000)={s[0]:.6f}, σ(-1000)={s[1]:.6f}")

    # lr=1e6 + L2 regularization causes weight explosion → L2 term overflows → ValueError
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            LogisticRegression(learning_rate=1e6, n_epochs=500, tol=0.0,
                               regularization=1.0).fit(X, y)
            raise AssertionError("Should have raised ValueError for non-finite loss")
        except ValueError as e:
            print(f"[PASS] Non-finite loss raises ValueError: {e}")

    # 3-class y must raise ValueError
    y3 = np.array([0, 1, 2] * 10)
    X3 = rng.normal(size=(30, 2))
    try:
        LogisticRegression().fit(X3, y3)
        raise AssertionError("Should have raised ValueError for 3-class y")
    except ValueError as e:
        print(f"[PASS] 3-class y raises ValueError: {e}")

    # negative learning_rate
    try:
        LogisticRegression(learning_rate=-0.1)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"[PASS] learning_rate<0 raises ValueError: {e}")

    print("\nAll sanity checks passed.")
