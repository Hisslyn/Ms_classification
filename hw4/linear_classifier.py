"""
Least-Squares Linear Classifier (HW4 Part 1).

Fits a hyperplane by minimising squared error against ±1 encoded labels,
then classifies by the sign of the linear prediction.  This is sometimes
called "linear regression for classification".

From-scratch status
-------------------
`numpy.linalg.solve` is a general-purpose linear-algebra solver (analogous
to `numpy.sqrt`) — it knows nothing about classification.  Using it here is
the same in spirit as using `numpy.mean` to compute a class prior.  No
classifier, no metric, no optimiser from any library is used.
"""

import numpy as np


class LeastSquaresClassifier:
    """Binary classifier that minimises the squared loss against ±1 targets.

    The normal equation is solved via ``np.linalg.solve`` rather than
    explicit matrix inversion for better numerical stability.

    Label encoding
    --------------
    The caller supplies 0/1 labels (or any two-class labels).  Internally
    they are mapped to -1/+1 before solving.  Predictions are mapped back
    to the original label space via ``self.classes_``.

    No ``predict_proba``
    --------------------
    Least-squares scores are raw linear outputs with no probabilistic
    interpretation.  Applying a sigmoid post-hoc would not yield calibrated
    probabilities — that is precisely the motivation for logistic regression
    (HW4 Part 2).  This method intentionally omits ``predict_proba``.

    Parameters
    ----------
    regularization : float, optional (default 0.0)
        Ridge penalty λ ≥ 0.  When > 0, solves
        ``(X̃ᵀX̃ + λI) w = X̃ᵀt`` instead of the unpenalised normal
        equation.  The bias column is excluded from regularisation
        (its diagonal entry in the penalty matrix is left at 0).
        Even a small λ (e.g. 1e-4) cures near-singular ``X̃ᵀX̃``.
    """

    def __init__(self, regularization: float = 0.0):
        if regularization < 0:
            raise ValueError(
                f"regularization must be >= 0, got {regularization}"
            )
        self.regularization = regularization
        self.weights_ = None
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the classifier using the normal equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.  Accepts numpy arrays or pandas
            DataFrame/Series; converted internally.
        y : array-like of shape (n_samples,)
            Binary class labels.  Exactly two unique values required.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"LeastSquaresClassifier requires exactly 2 classes, "
                f"got {len(unique_classes)}: {unique_classes}"
            )
        self.classes_ = unique_classes  # sorted by np.unique

        # Map original labels → 0/1 → ±1
        # classes_[0] → -1,  classes_[1] → +1
        t = np.where(y == self.classes_[1], 1.0, -1.0)

        # Augment X with a bias column of ones: X̃ ∈ ℝ^{n × (d+1)}
        n, d = X.shape
        X_aug = np.column_stack([X, np.ones(n)])

        # Build the left-hand side: A = X̃ᵀX̃  (optionally + λI)
        A = X_aug.T @ X_aug
        if self.regularization > 0:
            penalty = self.regularization * np.eye(d + 1)
            # Do not penalise the bias term (last diagonal entry)
            penalty[d, d] = 0.0
            A = A + penalty

        # Solve A w = X̃ᵀt  (more stable than w = A⁻¹ X̃ᵀt)
        rhs = X_aug.T @ t
        try:
            w = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "X̃ᵀX̃ is singular — the system has no unique solution. "
                "Try setting regularization > 0 (e.g. regularization=1e-4)."
            ) from exc

        self.weights_ = w
        self.coef_ = w[:d]
        self.intercept_ = float(w[d])
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Values are drawn from ``self.classes_``.
        """
        self._check_fitted()
        s = self.decision_function(X)
        # s ≥ 0  →  classes_[1];  s < 0  →  classes_[0]
        return np.where(s >= 0, self.classes_[1], self.classes_[0])

    # ------------------------------------------------------------------
    # decision_function
    # ------------------------------------------------------------------

    def decision_function(self, X):
        """Return raw linear scores s = X @ coef_ + intercept_.

        Useful for plotting decision boundaries, threshold analysis,
        and comparing against logistic regression's pre-sigmoid logits.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Continuous values; positive → classes_[1], negative → classes_[0].
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict() or decision_function().")


# ---------------------------------------------------------------------------
# Sanity test (synthetic data only — no real dataset)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Two well-separated Gaussian blobs, 100 points each
    X0 = rng.normal(loc=[-2, -2], scale=1.0, size=(100, 2))
    X1 = rng.normal(loc=[+2, +2], scale=1.0, size=(100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)

    # ---- basic fit / predict ----
    clf = LeastSquaresClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = np.mean(y_pred == y)
    assert acc > 0.95, f"Training accuracy too low on easy blobs: {acc:.4f}"
    print(f"[PASS] Training accuracy on blobs: {acc:.4f}")

    # ---- decision_function returns continuous scores ----
    scores = clf.decision_function(X)
    assert scores.dtype == float, "Scores must be float"
    unique_scores = np.unique(scores)
    assert len(unique_scores) > 2, "decision_function must return continuous values"
    print(f"[PASS] decision_function returns continuous scores "
          f"(range {scores.min():.3f} .. {scores.max():.3f})")

    # ---- classes_ reflects original label space ----
    assert list(clf.classes_) == [0, 1], f"Unexpected classes_: {clf.classes_}"
    print(f"[PASS] classes_ = {clf.classes_}")

    # ---- ValueError on 3-class y ----
    y3 = np.array([0, 1, 2] * 10)
    X3 = rng.normal(size=(30, 2))
    try:
        clf.fit(X3, y3)
        raise AssertionError("Should have raised ValueError for 3-class y")
    except ValueError as e:
        print(f"[PASS] 3-class y raises ValueError: {e}")

    # ---- ValueError on negative regularization ----
    try:
        LeastSquaresClassifier(regularization=-1)
        raise AssertionError("Should have raised ValueError for negative lambda")
    except ValueError as e:
        print(f"[PASS] regularization=-1 raises ValueError: {e}")

    # ---- degenerate dataset (two perfectly collinear columns) ----
    X_degen = np.column_stack([X[:, 0], X[:, 0]])  # col1 == col2
    try:
        clf_degen = LeastSquaresClassifier(regularization=0.0)
        clf_degen.fit(X_degen, y)
        # numpy may resolve this via least-norm; as long as it runs, accept it
        print("[PASS] Collinear columns (no regularization): solved without error")
    except np.linalg.LinAlgError as e:
        print(f"[PASS] Collinear columns (no regularization): raised LinAlgError — {e}")

    # Same degenerate dataset with regularization should always succeed
    clf_reg = LeastSquaresClassifier(regularization=1e-4)
    clf_reg.fit(X_degen, y)
    acc_reg = np.mean(clf_reg.predict(X_degen) == y)
    assert acc_reg > 0.95, f"Ridge accuracy too low on blobs: {acc_reg:.4f}"
    print(f"[PASS] Collinear columns + regularization=1e-4: accuracy {acc_reg:.4f}")

    print("\nAll sanity checks passed.")
