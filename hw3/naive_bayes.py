"""Discrete Naive Bayes classifier implemented from scratch using NumPy.

Assumes features are already integer-coded (categorical or binned continuous).
Binning of continuous features is the caller's responsibility — use
common.preprocessing.Binner before passing data to this class.
"""

import warnings

import numpy as np


class DiscreteNaiveBayes:
    """Discrete Naive Bayes classifier with Laplace (additive) smoothing.

    Applies Bayes' theorem under the naive (conditional-independence)
    assumption. Suitable for discrete (integer-valued) feature matrices.
    Continuous features must be discretized by the caller before fitting.

    All probabilities are stored and computed in log space to prevent
    floating-point underflow on datasets with many features.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Laplace smoothing parameter (additive smoothing).
        - alpha=1.0  — standard Laplace smoothing (recommended).
        - alpha=0.0  — no smoothing; a RuntimeWarning is issued at predict
          time if a feature value was never seen for a given class during
          training, because log(0) = -inf makes that class impossible.
        - 0 < alpha < 1 — "Lidstone" smoothing (fractional).
        Must be >= 0.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted unique class labels.
    class_priors_ : np.ndarray of shape (n_classes,)
        P(y = c) estimated from training frequencies.
    feature_values_ : list of length n_features
        feature_values_[j] is the sorted array of unique values seen for
        feature j during training. Used to determine vocabulary size |V_j|
        for smoothing.

    Raises
    ------
    ValueError
        If alpha < 0.
    """

    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Estimate class priors and per-feature conditional probabilities.

        Contract: X must contain only non-negative integer values. Floating-
        point inputs are cast to int; fractional parts are truncated silently.
        The caller is responsible for encoding / binning features before
        calling fit().

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Integer feature matrix.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=int)
        y = np.asarray(y)

        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Class counts and priors
        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.class_priors_ = class_counts / n_samples
        self._log_class_priors = np.log(self.class_priors_)

        # Per-feature vocabulary and conditional log-probabilities
        self.feature_values_ = []
        # val_to_logprob_[j]: dict {value: np.array of shape (n_classes,)}
        self._val_to_logprob = []
        # log_unseen_probs_[j]: np.array of shape (n_classes,) for unseen values
        self._log_unseen_probs = []

        for j in range(n_features):
            col = X[:, j]
            values = np.sort(np.unique(col))
            self.feature_values_.append(values)
            vocab_size = len(values)

            val_dict = {}
            for v in values:
                log_p_vec = np.empty(n_classes, dtype=float)
                for ci, cls in enumerate(self.classes_):
                    mask = y == cls
                    cnt = (col[mask] == v).sum()
                    # Laplace-smoothed conditional prob
                    # alpha=0 + cnt=0 → log(0) = -inf: suppress numpy warning;
                    # the -inf is intentional (impossible event under MLE).
                    with np.errstate(divide="ignore"):
                        log_p_vec[ci] = np.log(
                            (cnt + self.alpha) / (class_counts[ci] + self.alpha * vocab_size)
                        )
                val_dict[v] = log_p_vec

            # Unseen-value log prob: alpha / (N_c + alpha * |V_j|)
            # If alpha == 0, this is 0.0 → log(0) = -inf
            unseen_vec = np.empty(n_classes, dtype=float)
            for ci, cls in enumerate(self.classes_):
                p_unseen = self.alpha / (class_counts[ci] + self.alpha * vocab_size)
                # log(0) produces -inf; we store it deliberately — see predict docstring
                with np.errstate(divide="ignore"):
                    unseen_vec[ci] = np.log(p_unseen) if p_unseen > 0 else -np.inf

            self._val_to_logprob.append(val_dict)
            self._log_unseen_probs.append(unseen_vec)

        return self

    def predict(self, X):
        """Predict class labels using the MAP rule.

        For each test point, computes:
            log P(y=c) + sum_j log P(x_j | y=c)

        and returns the class with the highest log-posterior.

        Unseen feature values (not observed for a class during training)
        are assigned the smoothed probability alpha / (N_c + alpha * |V_j|).
        When alpha=0 this probability is 0.0 (log = -inf), making that class
        impossible. A RuntimeWarning is emitted in that case to alert the
        caller that zero smoothing is interacting with out-of-vocabulary values.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)

        Returns
        -------
        np.ndarray of shape (n_test,)
        """
        log_posteriors = self._compute_log_posteriors(X)
        return self.classes_[np.argmax(log_posteriors, axis=1)]

    def predict_proba(self, X):
        """Return normalized per-class probability estimates.

        Log-posteriors are converted to probabilities using the log-sum-exp
        trick for numerical stability:

            log Z = log( sum_c exp(log_score_c) )
                  = max_score + log( sum_c exp(log_score_c - max_score) )

        Subtracting max_score before exponentiating prevents overflow
        without affecting the normalized result.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)

        Returns
        -------
        np.ndarray of shape (n_test, n_classes)
            Rows sum to 1.
        """
        log_posteriors = self._compute_log_posteriors(X)
        # log-sum-exp trick: subtract row max for numerical stability
        max_scores = log_posteriors.max(axis=1, keepdims=True)
        shifted = log_posteriors - max_scores
        exp_scores = np.exp(shifted)
        row_sums = exp_scores.sum(axis=1, keepdims=True)
        return exp_scores / row_sums

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        if not hasattr(self, "_log_class_priors"):
            raise RuntimeError("Call fit() before predict().")

    def _compute_log_posteriors(self, X):
        """Compute unnormalized log-posterior for each (test point, class) pair.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)

        Returns
        -------
        np.ndarray of shape (n_test, n_classes)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=int)
        n_test = X.shape[0]
        n_classes = len(self.classes_)

        # Start with log priors, broadcast to (n_test, n_classes)
        scores = np.tile(self._log_class_priors, (n_test, 1))

        for j, (val_dict, unseen_lp) in enumerate(
            zip(self._val_to_logprob, self._log_unseen_probs)
        ):
            col = X[:, j]
            # Build (n_test, n_classes) contribution for this feature
            contrib = np.tile(unseen_lp, (n_test, 1))  # default: unseen
            for v, log_p_vec in val_dict.items():
                mask = col == v
                if mask.any():
                    contrib[mask] = log_p_vec

            if self.alpha == 0 and np.any(np.isinf(contrib)):
                warnings.warn(
                    f"Feature {j} has unseen value(s) at predict time with alpha=0. "
                    "This gives log-probability -inf, making affected classes impossible. "
                    "Use alpha > 0 (Laplace smoothing) to avoid this.",
                    RuntimeWarning,
                    stacklevel=3,
                )

            scores += contrib

        return scores


# ---------------------------------------------------------------------------
# Sanity test (synthetic discrete data only — no real dataset)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # 3 features, 3 values each (0/1/2), 2 classes, 100 rows
    # Class 0: strongly prefers value=0 on feature 0
    # Class 1: strongly prefers value=2 on feature 0
    n = 100
    y_synth = rng.integers(0, 2, size=n)

    def make_feature(y, pref_val, n_vals=3, strength=0.85):
        col = np.empty(n, dtype=int)
        for i, cls in enumerate(y):
            if rng.random() < strength:
                col[i] = pref_val[cls]
            else:
                col[i] = rng.integers(0, n_vals)
        return col

    X_synth = np.column_stack([
        make_feature(y_synth, pref_val={0: 0, 1: 2}),
        make_feature(y_synth, pref_val={0: 1, 1: 1}, strength=0.5),  # weak feature
        make_feature(y_synth, pref_val={0: 0, 1: 2}),
    ])

    # Simple 80/20 split
    split = int(0.8 * n)
    X_tr, X_te = X_synth[:split], X_synth[split:]
    y_tr, y_te = y_synth[:split], y_synth[split:]

    print("=== DiscreteNaiveBayes Sanity Test ===")

    nb = DiscreteNaiveBayes(alpha=1.0)
    nb.fit(X_tr, y_tr)

    train_acc = (nb.predict(X_tr) == y_tr).mean()
    print(f"Training accuracy (expect >90%): {train_acc:.4f}")
    assert train_acc > 0.90, f"Training accuracy too low: {train_acc:.4f}"

    test_preds = nb.predict(X_te)
    test_acc = (test_preds == y_te).mean()
    print(f"Test accuracy: {test_acc:.4f}")

    # predict_proba rows must sum to 1
    proba = nb.predict_proba(X_te)
    row_sums = proba.sum(axis=1)
    print(f"predict_proba row sums — min={row_sums.min():.8f}  max={row_sums.max():.8f}")
    assert np.allclose(row_sums, 1.0), "predict_proba rows must sum to 1"

    # Unseen feature value must not crash (alpha=1.0)
    X_unseen = np.array([[99, 99, 99]])  # values never seen in training
    pred_unseen = nb.predict(X_unseen)
    proba_unseen = nb.predict_proba(X_unseen)
    assert np.isclose(proba_unseen.sum(), 1.0), "unseen value proba must sum to 1"
    print(f"Unseen value prediction: class={pred_unseen[0]}, proba={proba_unseen}")

    # alpha=0 + unseen value should warn
    nb_zero = DiscreteNaiveBayes(alpha=0.0)
    nb_zero.fit(X_tr, y_tr)
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        nb_zero.predict(X_unseen)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
        "Expected RuntimeWarning for alpha=0 + unseen value"
    )
    print(f"RuntimeWarning raised for alpha=0 + unseen value: True")

    print("\nAll assertions passed.")
