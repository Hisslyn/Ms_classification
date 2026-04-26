"""
Linear and Quadratic Discriminant Analysis from scratch (HW5 Part 1).

Both LDA and QDA are generative classifiers:
    P(y=c | x)  ∝  P(x | y=c) · P(y=c)
where P(x | y=c) = N(x; μ_c, Σ_c) is a multivariate Gaussian.

Classification uses the log discriminant (log posterior dropping the
x-only normalizing constant).

Design: _GaussianDiscriminant is a private abstract base class that holds
the shared fit / predict / predict_proba / decision_function scaffold.
LDA and QDA each override _fit_covariances() and _discriminant() with
their specific covariance model.  Keeping the differences isolated makes
the code easy to audit against the mathematical derivation.

From-scratch status:
    NumPy linear algebra primitives only (np.linalg.solve, np.linalg.slogdet).
    These are general numerical routines — no statistical model is imported.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Private base
# ---------------------------------------------------------------------------

class _GaussianDiscriminant:
    """Abstract base for LDA and QDA.  Not intended for direct instantiation."""

    def __init__(self, reg_param: float = 1e-6):
        if reg_param < 0:
            raise ValueError(f"reg_param must be >= 0, got {reg_param!r}")
        self.reg_param = reg_param

        self.classes_ = None
        self.priors_ = None
        self.means_ = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Estimate class statistics from training data.

        Computes per-class priors π_c, means μ_c, and covariance(s).
        Pre-computes matrix inverses and log-determinants for fast predict.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.sort(np.unique(y))
        N, n_features = X.shape

        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.priors_ = counts / N
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])

        self._fit_covariances(X, y, counts, n_features)
        return self

    def _fit_covariances(self, X, y, counts, n_features):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # predict / predict_proba / decision_function
    # ------------------------------------------------------------------

    def decision_function(self, X):
        """Raw log-discriminant scores δ_c(x) for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples, K)
            Unnormalized log-posteriors (up to a constant in x).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return self._discriminant(X)

    def predict(self, X):
        """Predict class labels by argmax discriminant score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) — values from self.classes_
        """
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Posterior class probabilities via log-sum-exp normalization.

        Converts log-discriminant scores to probabilities using the
        log-sum-exp trick to prevent overflow:

            log Z_i = max_c δ_c(x_i) + log Σ_c exp(δ_c(x_i) - max_c δ_c(x_i))
            P(y=c | x_i) = exp(δ_c(x_i) - log Z_i)

        Subtracting the row-wise maximum before exponentiating is numerically
        equivalent to the naive formula but avoids overflow for large scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, K) — rows sum to 1
        """
        scores = self.decision_function(X)
        max_scores = scores.max(axis=1, keepdims=True)
        exp_shifted = np.exp(scores - max_scores)
        return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Subclass responsibility
    # ------------------------------------------------------------------

    def _discriminant(self, X):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("Call fit() before predict() or predict_proba().")


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

class LDA(_GaussianDiscriminant):
    """Linear Discriminant Analysis from scratch.

    Models P(x | y=c) = N(x; μ_c, Σ) with a single shared covariance Σ
    (pooled across all classes).  The shared-covariance assumption causes the
    quadratic terms in (x - μ_c)ᵀ Σ⁻¹ (x - μ_c) to cancel, leaving a
    discriminant that is linear in x:

        δ_c(x) = xᵀ Σ⁻¹ μ_c  −  (1/2) μ_cᵀ Σ⁻¹ μ_c  +  log π_c

    Pooled covariance (unbiased estimator, degrees-of-freedom correction):
        Σ = (1 / (N - K)) · Σ_c Σ_{i: y_i=c} (x_i - μ_c)(x_i - μ_c)ᵀ

    Σ⁻¹ is computed via np.linalg.solve(Σ, I) rather than np.linalg.inv(Σ).
    solve uses LU factorisation and is numerically stabler for near-singular
    matrices because it never forms the explicit inverse.

    Parameters
    ----------
    reg_param : float, optional (default=1e-6)
        Ridge regularization: adds ε·I to Σ before inversion.  Prevents
        singular covariance when features are collinear or the dataset is
        small.  Must be >= 0.

    Attributes (after fit)
    ----------------------
    classes_ : ndarray (K,)
    priors_ : ndarray (K,) — class frequencies π_c = N_c / N
    means_ : ndarray (K, n_features) — per-class means μ_c
    covariance_ : ndarray (n_features, n_features) — pooled Σ (with ridge)
    """

    def _fit_covariances(self, X, y, counts, n_features):
        K = len(self.classes_)
        N = len(y)

        scatter = np.zeros((n_features, n_features))
        for c, mu_c in zip(self.classes_, self.means_):
            Xc = X[y == c] - mu_c      # (N_c, d) centered
            scatter += Xc.T @ Xc

        # Unbiased pooled estimator: divide by (N - K)
        cov = scatter / (N - K) + self.reg_param * np.eye(n_features)
        self.covariance_ = cov

        # Solve Σ W = I column-by-column — stabler than explicit inv.
        self._inv_cov = np.linalg.solve(cov, np.eye(n_features))

    def _discriminant(self, X):
        """LDA discriminant: linear in x because Σ is shared."""
        inv_cov = self._inv_cov   # (d, d)
        means = self.means_        # (K, d)

        # xᵀ Σ⁻¹ μ_c  for all (sample, class) pairs → (n_samples, K)
        linear_terms = X @ inv_cov @ means.T

        # (1/2) μ_cᵀ Σ⁻¹ μ_c  for each class → (K,)
        quad_terms = 0.5 * ((means @ inv_cov) * means).sum(axis=1)

        log_priors = np.log(self.priors_)   # (K,)

        return linear_terms - quad_terms + log_priors   # (n_samples, K)


# ---------------------------------------------------------------------------
# QDA
# ---------------------------------------------------------------------------

class QDA(_GaussianDiscriminant):
    """Quadratic Discriminant Analysis from scratch.

    Models P(x | y=c) = N(x; μ_c, Σ_c) with a separate covariance Σ_c per
    class.  Different class covariances produce a quadratic decision boundary:

        δ_c(x) = −(1/2) log|Σ_c|  −  (1/2)(x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c)  +  log π_c

    Per-class covariance (unbiased, Bessel correction):
        Σ_c = (1 / (N_c − 1)) · Σ_{i: y_i=c} (x_i − μ_c)(x_i − μ_c)ᵀ

    log|Σ_c| is computed via np.linalg.slogdet to avoid floating-point
    overflow on the raw determinant of large matrices.

    Σ_c⁻¹ is computed via np.linalg.solve(Σ_c, I) — same rationale as LDA.

    Parameters
    ----------
    reg_param : float, optional (default=1e-6)
        Ridge regularization: adds ε·I to each Σ_c before inversion.
        Must be >= 0.

    Attributes (after fit)
    ----------------------
    classes_ : ndarray (K,)
    priors_ : ndarray (K,) — class frequencies π_c = N_c / N
    means_ : ndarray (K, n_features) — per-class means μ_c
    covariances_ : ndarray (K, n_features, n_features) — per-class Σ_c (with ridge)

    Raises
    ------
    ValueError
        If any class has fewer than n_features + 1 samples (rank-deficient
        covariance estimate even before adding ridge regularization).
        Suggestion: use LDA, collect more data, or reduce dimensionality.
    ValueError
        If any regularized covariance matrix is not positive definite after
        adding the ridge (slogdet sign ≤ 0).  Increase reg_param.
    """

    def _fit_covariances(self, X, y, counts, n_features):
        K = len(self.classes_)

        # Guard: N_c < n_features + 1 → rank-deficient scatter matrix.
        # Detect early and give a clear diagnostic rather than letting
        # slogdet silently return -inf or solve produce garbage.
        for c, n_c in zip(self.classes_, counts):
            if n_c < n_features + 1:
                raise ValueError(
                    f"Class {c!r} has only {int(n_c)} sample(s) but {n_features} "
                    f"feature(s). QDA requires at least n_features + 1 = "
                    f"{n_features + 1} samples per class for a full-rank covariance "
                    "estimate.  Alternatives: use LDA (pooled covariance), collect "
                    "more data, or reduce dimensionality (e.g. PCA)."
                )

        covs = np.empty((K, n_features, n_features))
        inv_covs = np.empty((K, n_features, n_features))
        log_dets = np.empty(K)

        for ci, (c, mu_c) in enumerate(zip(self.classes_, self.means_)):
            n_c = int(counts[ci])
            Xc = X[y == c] - mu_c               # (N_c, d) centered
            cov = Xc.T @ Xc / (n_c - 1) + self.reg_param * np.eye(n_features)
            covs[ci] = cov

            # slogdet returns (sign, log|det|); avoids overflow from raw det.
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError(
                    f"Regularized covariance for class {c!r} is not positive definite "
                    f"(slogdet sign={sign}).  Increase reg_param or use LDA."
                )
            log_dets[ci] = log_det

            inv_covs[ci] = np.linalg.solve(cov, np.eye(n_features))

        self.covariances_ = covs
        self._inv_covs = inv_covs
        self._log_dets = log_dets

    def _discriminant(self, X):
        """QDA discriminant: quadratic in x due to per-class Σ_c."""
        n_samples = X.shape[0]
        K = len(self.classes_)
        scores = np.empty((n_samples, K))
        log_priors = np.log(self.priors_)

        for ci, (mu_c, inv_cov, log_det) in enumerate(
            zip(self.means_, self._inv_covs, self._log_dets)
        ):
            diff = X - mu_c                              # (n_samples, d)
            # Mahalanobis² = (x-μ)ᵀ Σ⁻¹ (x-μ), one row at a time (vectorised).
            # diff @ inv_cov → (n, d); element-wise * diff then sum over d → (n,)
            mahal = (diff @ inv_cov * diff).sum(axis=1)  # (n_samples,)
            scores[:, ci] = -0.5 * log_det - 0.5 * mahal + log_priors[ci]

        return scores


# ---------------------------------------------------------------------------
# Sanity tests (synthetic data only — no real dataset)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print("=" * 60)
    print("LDA / QDA Sanity Tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def make_data_shared_cov(rng, n_per_class=100):
        """Three Gaussian blobs with a single shared covariance (LDA's world)."""
        cov = np.array([[2.0, 0.8], [0.8, 1.0]])
        means = [np.array([-4.0, 0.0]),
                 np.array([0.0, 3.0]),
                 np.array([4.0, -1.0])]
        Xs = [rng.multivariate_normal(mu, cov, size=n_per_class) for mu in means]
        X = np.vstack(Xs)
        y = np.repeat([0, 1, 2], n_per_class)
        return X, y

    def make_data_distinct_cov(rng, n_per_class=100):
        """Three Gaussians with clearly different covariances (QDA's world)."""
        covs = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),        # isotropic
            np.array([[4.0, 0.0], [0.0, 0.3]]),        # elongated along x
            np.array([[0.3, 0.0], [0.0, 4.0]]),        # elongated along y
        ]
        means = [np.array([-4.0, 0.0]),
                 np.array([2.0, 2.0]),
                 np.array([2.0, -2.0])]
        Xs = [rng.multivariate_normal(mu, cov, size=n_per_class)
              for mu, cov in zip(means, covs)]
        X = np.vstack(Xs)
        y = np.repeat([0, 1, 2], n_per_class)
        return X, y

    def check_proba_sums(model, X, label):
        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), (
            f"{label}: predict_proba rows don't sum to 1; "
            f"max dev = {np.abs(row_sums - 1.0).max():.2e}"
        )
        print(f"[PASS] {label}: predict_proba rows sum to 1")

    # ------------------------------------------------------------------
    # 1. LDA on shared-covariance data (assumption holds exactly)
    # ------------------------------------------------------------------
    print("\n--- Test 1: LDA on shared-covariance data ---")
    X_lda, y_lda = make_data_shared_cov(rng)

    lda = LDA(reg_param=1e-6)
    lda.fit(X_lda, y_lda)

    train_acc = np.mean(lda.predict(X_lda) == y_lda)
    assert train_acc > 0.90, f"LDA training acc too low: {train_acc:.4f}"
    print(f"[PASS] LDA training accuracy = {train_acc:.4f}  (>0.90 required)")

    check_proba_sums(lda, X_lda, "LDA")

    scores = lda.decision_function(X_lda)
    assert scores.shape == (300, 3), f"decision_function shape: {scores.shape}"
    print(f"[PASS] LDA decision_function shape = {scores.shape}")

    # ------------------------------------------------------------------
    # 2. QDA on distinct-covariance data (assumption holds)
    # ------------------------------------------------------------------
    print("\n--- Test 2: QDA on distinct-covariance data ---")
    X_qda, y_qda = make_data_distinct_cov(rng)

    qda = QDA(reg_param=1e-6)
    qda.fit(X_qda, y_qda)

    train_acc_q = np.mean(qda.predict(X_qda) == y_qda)
    assert train_acc_q > 0.90, f"QDA training acc too low: {train_acc_q:.4f}"
    print(f"[PASS] QDA training accuracy = {train_acc_q:.4f}  (>0.90 required)")

    check_proba_sums(qda, X_qda, "QDA")

    scores_q = qda.decision_function(X_qda)
    assert scores_q.shape == (300, 3), f"QDA decision_function shape: {scores_q.shape}"
    print(f"[PASS] QDA decision_function shape = {scores_q.shape}")

    # ------------------------------------------------------------------
    # 3. QDA ValueError: too few samples for a class (5 points, 10 dims)
    # ------------------------------------------------------------------
    print("\n--- Test 3: QDA ValueError for under-sampled class ---")
    X_small = rng.normal(size=(5, 10))    # 5 samples, 10 features — rank-deficient
    y_small = np.zeros(5, dtype=int)
    X_ok = rng.normal(size=(50, 10))
    y_ok = np.ones(50, dtype=int)
    X_bad = np.vstack([X_small, X_ok])
    y_bad = np.concatenate([y_small, y_ok])

    try:
        QDA().fit(X_bad, y_bad)
        raise AssertionError("QDA should have raised ValueError for under-sampled class")
    except ValueError as exc:
        print(f"[PASS] QDA raised ValueError: {exc}")

    # ------------------------------------------------------------------
    # 4. LDA on distinct-covariance data — lower accuracy (wrong assumption)
    # ------------------------------------------------------------------
    print("\n--- Test 4: LDA on distinct-covariance data (assumption violated) ---")
    lda_on_qda_data = LDA(reg_param=1e-6)
    lda_on_qda_data.fit(X_qda, y_qda)
    acc_lda_on_qda = np.mean(lda_on_qda_data.predict(X_qda) == y_qda)
    print(f"[INFO] LDA accuracy on QDA-style data = {acc_lda_on_qda:.4f}  "
          f"(expect lower than QDA's {train_acc_q:.4f})")
    # Not a hard threshold — just confirm LDA still runs without error
    print("[PASS] LDA ran successfully on distinct-covariance data")

    # ------------------------------------------------------------------
    # 5. Binary classification (K=2)
    # ------------------------------------------------------------------
    print("\n--- Test 5: Binary classification (K=2) ---")
    X_bin = np.vstack([
        rng.multivariate_normal([-2.0, -2.0], np.eye(2), size=80),
        rng.multivariate_normal([+2.0, +2.0], np.eye(2), size=80),
    ])
    y_bin = np.array([0] * 80 + [1] * 80)

    lda_bin = LDA().fit(X_bin, y_bin)
    qda_bin = QDA().fit(X_bin, y_bin)

    for name, clf in [("LDA binary", lda_bin), ("QDA binary", qda_bin)]:
        acc = np.mean(clf.predict(X_bin) == y_bin)
        assert acc > 0.90, f"{name} acc too low: {acc:.4f}"
        print(f"[PASS] {name} training accuracy = {acc:.4f}")
        check_proba_sums(clf, X_bin, name)

    # ------------------------------------------------------------------
    # 6. Constructor validation
    # ------------------------------------------------------------------
    print("\n--- Test 6: Constructor validation ---")
    try:
        LDA(reg_param=-1.0)
        raise AssertionError("Should have raised ValueError")
    except ValueError as exc:
        print(f"[PASS] LDA(reg_param=-1) raises ValueError: {exc}")

    try:
        QDA(reg_param=-0.01)
        raise AssertionError("Should have raised ValueError")
    except ValueError as exc:
        print(f"[PASS] QDA(reg_param=-0.01) raises ValueError: {exc}")

    print("\n" + "=" * 60)
    print("All sanity checks passed.")
    print("=" * 60)
