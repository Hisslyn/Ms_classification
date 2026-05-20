"""Feature preprocessing: scaling and binning, implemented from scratch."""

import numpy as np


class StandardScaler:
    """Standardize features to zero mean and unit variance.

    Computes mean and population std (ddof=0) from training data, then applies
    the transform X_scaled = (X - mean) / std to any array. When std is 0 for
    a feature (constant column), it is treated as 1 to avoid division by zero,
    leaving that feature unchanged after mean subtraction (it becomes all zeros).

    Attributes:
        mean_: np.ndarray of shape (n_features,) — per-feature training mean.
        std_:  np.ndarray of shape (n_features,) — per-feature training
               population std (ddof=0); zeros replaced by 1.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute mean and population std (ddof=0) from training data X.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        # Replace zero std with 1 so constant features become all-zeros after scaling
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        """Apply standardization using stored mean and std.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            np.ndarray of same shape, standardized.

        Raises:
            RuntimeError: If fit has not been called yet.
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit to X, then transform X.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            np.ndarray of same shape, standardized.
        """
        return self.fit(X).transform(X)


class Binner:
    """Discretize continuous features into integer bin indices.

    Supports two strategies:
    - "uniform": equal-width bins over [min, max] of each feature.
    - "quantile": equal-frequency bins based on training data quantiles.

    Bin edges are learned from training data and reused for test data,
    preventing data leakage. Values outside training range are clipped to
    the boundary bins.

    Args:
        n_bins: Number of bins per feature (default 5).
        strategy: "uniform" or "quantile".

    Attributes:
        edges_: List of np.ndarray, one per feature, containing bin edges.
                Uniform strategy: length n_bins + 1. Quantile strategy:
                length <= n_bins + 1 (duplicates removed via np.unique).
    """

    def __init__(self, n_bins=5, strategy="uniform"):
        if strategy not in ("uniform", "quantile"):
            raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")
        self.n_bins = n_bins
        self.strategy = strategy
        self.edges_ = None

    def fit(self, X):
        """Compute bin edges from training data X.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.edges_ = []
        for j in range(n_features):
            col = X[:, j]
            if self.strategy == "uniform":
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)
            else:  # quantile
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(col, quantiles)
                # Deduplicate edges that collapse (can happen with many ties)
                edges = np.unique(edges)
            self.edges_.append(edges)
        return self

    def transform(self, X):
        """Bin X using stored edges, returning integer indices in [0, len(edges)-2].

        Values below the lowest edge go to bin 0; values above the highest
        edge go to the last bin (clipping behavior).

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            np.ndarray of int, same shape as X.

        Raises:
            RuntimeError: If fit has not been called yet.
        """
        if self.edges_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        scalar_input = X.ndim == 1
        if scalar_input:
            X = X.reshape(-1, 1)

        result = np.empty_like(X, dtype=int)
        for j, edges in enumerate(self.edges_):
            # np.searchsorted returns index in [0, len(edges)]; shift to [0, n_bins-1]
            idx = np.searchsorted(edges, X[:, j], side="right") - 1
            idx = np.clip(idx, 0, len(edges) - 2)
            result[:, j] = idx

        return result.ravel() if scalar_input else result

    def fit_transform(self, X):
        """Fit to X, then transform X.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            np.ndarray of int, same shape as X.
        """
        return self.fit(X).transform(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, 3))
    X_test = rng.normal(size=(20, 3))

    # StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    print(f"StandardScaler — train mean (should be ~0): {X_tr_scaled.mean(axis=0).round(6)}")
    print(f"StandardScaler — train std  (should be ~1): {X_tr_scaled.std(axis=0).round(6)}")
    print(f"StandardScaler — test mean (not guaranteed 0): {X_te_scaled.mean(axis=0).round(4)}")

    # Test zero-std column
    X_const = np.column_stack([X_train, np.ones(100)])
    scaler2 = StandardScaler()
    X_scaled_const = scaler2.fit_transform(X_const)
    print(f"\nConstant column after scaling (should be all 0): {np.unique(X_scaled_const[:, -1])}")

    # Binner (uniform)
    binner_u = Binner(n_bins=5, strategy="uniform")
    bins_u = binner_u.fit_transform(X_train)
    print(f"\nUniform binner — unique bin values: {np.unique(bins_u)}")  # should be 0-4

    # Binner (quantile)
    binner_q = Binner(n_bins=5, strategy="quantile")
    bins_q = binner_q.fit_transform(X_train)
    print(f"Quantile binner — bin counts (should be ~equal): {np.bincount(bins_q[:, 0])}")
