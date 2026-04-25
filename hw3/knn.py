"""k-Nearest Neighbors classifier implemented from scratch using NumPy."""

import numpy as np


class KNN:
    """k-Nearest Neighbors classifier.

    A lazy learner: fit() stores the training set; all computation happens
    at predict time. Distance is computed via vectorized broadcasting so no
    Python-level loop over test points is needed.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to use (must be >= 1).
    distance : {"euclidean", "manhattan"}
        Distance metric used to find neighbors.
    weights : {"uniform", "distance"}
        Voting scheme.
        - "uniform": each of the k neighbors gets one vote.
        - "distance": each neighbor is weighted by 1/distance. A training
          point at distance 0 gets infinite weight and its label wins
          outright (see predict docstring).
        Note: "distance" weighting is an extension beyond the brief's
        minimum requirements.

    Raises
    ------
    ValueError
        If any constructor argument is invalid.
    """

    _VALID_DISTANCES = {"euclidean", "manhattan"}
    _VALID_WEIGHTS = {"uniform", "distance"}

    def __init__(self, k: int = 5, distance: str = "euclidean", weights: str = "uniform"):
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if distance not in self._VALID_DISTANCES:
            raise ValueError(
                f"distance must be one of {sorted(self._VALID_DISTANCES)}, got {distance!r}"
            )
        if weights not in self._VALID_WEIGHTS:
            raise ValueError(
                f"weights must be one of {sorted(self._VALID_WEIGHTS)}, got {weights!r}"
            )
        self.k = k
        self.distance = distance
        self.weights = weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Store training data. kNN has no learning phase.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.sort(np.unique(self.y_train))
        return self

    def predict(self, X):
        """Predict class labels for each row in X.

        For each test point the k nearest training points are found and a
        vote is taken:
        - uniform: majority class wins; ties broken by lowest label value.
        - distance: each neighbor contributes weight 1/d; if any neighbor
          has distance 0, only those zero-distance neighbors vote (their
          weight is treated as infinite, dominating any finite weights).
          Ties broken by lowest label value.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)

        Returns
        -------
        np.ndarray of shape (n_test,)

        Raises
        ------
        ValueError
            If k > number of training points.
        """
        X = np.asarray(X, dtype=float)
        self._check_is_fitted()
        n_train = self.X_train.shape[0]
        if self.k > n_train:
            raise ValueError(
                f"k={self.k} is greater than the number of training points ({n_train})"
            )

        dist_matrix = self._compute_distances(X)          # (n_test, n_train)
        neighbor_idx = np.argpartition(dist_matrix, self.k, axis=1)[:, : self.k]
        neighbor_dist = dist_matrix[np.arange(len(X))[:, None], neighbor_idx]
        neighbor_labels = self.y_train[neighbor_idx]

        scores = self._vote(neighbor_labels, neighbor_dist)  # (n_test, n_classes)
        # Tie-breaking: argmax returns the first maximum, and classes_ is sorted,
        # so the lowest label wins on a tie.
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Return per-class probability estimates.

        Each row sums to 1. Under uniform weights this is simply the
        fraction of neighbors belonging to each class. Under distance
        weights it is the normalized weighted vote count per class.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)

        Returns
        -------
        np.ndarray of shape (n_test, n_classes)

        Raises
        ------
        ValueError
            If k > number of training points.
        """
        X = np.asarray(X, dtype=float)
        self._check_is_fitted()
        n_train = self.X_train.shape[0]
        if self.k > n_train:
            raise ValueError(
                f"k={self.k} is greater than the number of training points ({n_train})"
            )

        dist_matrix = self._compute_distances(X)
        neighbor_idx = np.argpartition(dist_matrix, self.k, axis=1)[:, : self.k]
        neighbor_dist = dist_matrix[np.arange(len(X))[:, None], neighbor_idx]
        neighbor_labels = self.y_train[neighbor_idx]

        scores = self._vote(neighbor_labels, neighbor_dist)  # (n_test, n_classes)
        row_sums = scores.sum(axis=1, keepdims=True)
        return scores / row_sums

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Call fit() before predict().")

    def _compute_distances(self, X):
        """Dispatch to the correct distance method."""
        if self.distance == "euclidean":
            return self._euclidean(X)
        return self._manhattan(X)

    def _euclidean(self, X):
        """Compute Euclidean distance matrix via broadcasting.

        Parameters
        ----------
        X : np.ndarray of shape (n_test, n_features)

        Returns
        -------
        np.ndarray of shape (n_test, n_train)
            D[i, j] = ||X[i] - X_train[j]||_2
        """
        # X[:, None, :] shape: (n_test, 1, n_features)
        # self.X_train[None, :, :] shape: (1, n_train, n_features)
        diff = X[:, None, :] - self.X_train[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def _manhattan(self, X):
        """Compute Manhattan distance matrix via broadcasting.

        Parameters
        ----------
        X : np.ndarray of shape (n_test, n_features)

        Returns
        -------
        np.ndarray of shape (n_test, n_train)
            D[i, j] = sum_f |X[i, f] - X_train[j, f]|
        """
        diff = X[:, None, :] - self.X_train[None, :, :]
        return np.abs(diff).sum(axis=2)

    def _vote(self, neighbor_labels, neighbor_dist):
        """Aggregate neighbor votes into a score matrix.

        Parameters
        ----------
        neighbor_labels : np.ndarray of shape (n_test, k)
        neighbor_dist   : np.ndarray of shape (n_test, k)

        Returns
        -------
        np.ndarray of shape (n_test, n_classes)
            Unnormalized scores; larger = more votes for that class.
        """
        n_test = neighbor_labels.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_test, n_classes), dtype=float)

        # Map each label to its index in classes_
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}

        if self.weights == "uniform":
            for ci, cls in enumerate(self.classes_):
                scores[:, ci] = (neighbor_labels == cls).sum(axis=1)
        else:
            # distance weighting: weight = 1/d; zero-distance neighbors dominate
            for i in range(n_test):
                dists = neighbor_dist[i]          # shape (k,)
                labels = neighbor_labels[i]       # shape (k,)
                zero_mask = dists == 0.0
                if zero_mask.any():
                    # Only zero-distance neighbors vote, each with weight 1
                    for lbl in labels[zero_mask]:
                        scores[i, label_to_idx[lbl]] += 1.0
                else:
                    inv = 1.0 / dists
                    for j, lbl in enumerate(labels):
                        scores[i, label_to_idx[lbl]] += inv[j]

        return scores


# ---------------------------------------------------------------------------
# Sanity test (synthetic data only — no real dataset)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Two well-separated Gaussian blobs, 50 points each
    X0 = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(50, 2))
    X1 = rng.normal(loc=[4.0, 4.0], scale=0.8, size=(50, 2))
    X_all = np.vstack([X0, X1])
    y_all = np.array([0] * 50 + [1] * 50)

    # Shuffle
    perm = rng.permutation(100)
    X_all, y_all = X_all[perm], y_all[perm]

    # Simple 80/20 split (no sklearn — manual)
    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    results = {}

    for dist in ("euclidean", "manhattan"):
        for w in ("uniform", "distance"):
            knn = KNN(k=5, distance=dist, weights=w)
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            acc = (preds == y_test).mean()
            results[(dist, w)] = acc

    # k=1 must memorize training set → 100% train accuracy
    knn1 = KNN(k=1, distance="euclidean", weights="uniform")
    knn1.fit(X_train, y_train)
    train_acc_k1 = (knn1.predict(X_train) == y_train).mean()

    # predict_proba rows must sum to 1
    proba = knn1.predict_proba(X_test)
    row_sums = proba.sum(axis=1)

    # ValueError on k > n_train
    try:
        bad = KNN(k=999)
        bad.fit(X_train, y_train)
        bad.predict(X_test)
        val_err_raised = False
    except ValueError:
        val_err_raised = True

    print("=== KNN Sanity Test ===")
    print(f"k=1 train accuracy (expect 1.0): {train_acc_k1:.4f}")
    assert train_acc_k1 == 1.0, "k=1 should memorize training data"

    for (dist, w), acc in results.items():
        print(f"  distance={dist:<12} weights={w:<10} test_acc={acc:.4f}")
        assert acc > 0.85, f"Expected >85% on easy blobs, got {acc:.4f} for ({dist}, {w})"

    print(f"predict_proba row sums — min={row_sums.min():.6f}  max={row_sums.max():.6f}")
    assert np.allclose(row_sums, 1.0), "predict_proba rows must sum to 1"

    print(f"ValueError raised for k > n_train: {val_err_raised}")
    assert val_err_raised, "Should raise ValueError when k > n_train"

    print("\nAll assertions passed.")
