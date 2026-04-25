"""Train/test splitting and cross-validation utilities implemented from scratch."""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """Split arrays into train and test subsets.

    When stratify=True, each class is split independently so that class
    proportions are (approximately) preserved in both subsets. This mirrors
    sklearn's StratifiedShuffleSplit logic but uses only NumPy.

    Args:
        X: Array-like of shape (n_samples, n_features) or (n_samples,).
        y: 1-D array-like of labels, length n_samples.
        test_size: Fraction of the dataset to include in the test split (0, 1).
        random_state: Seed for np.random.default_rng for reproducibility.
        stratify: If True, preserve class proportions in each split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    if stratify:
        train_idx, test_idx = _stratified_split(y, test_size, rng)
    else:
        n = len(y)
        perm = rng.permutation(n)
        n_test = max(1, int(np.floor(test_size * n)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _stratified_split(y, test_size, rng):
    """Return (train_indices, test_indices) preserving class proportions.

    For each class, shuffles its indices and takes the first test_size
    fraction as test; the rest as train.

    Args:
        y: 1-D numpy array of labels.
        test_size: Fraction for the test set.
        rng: np.random.Generator instance.

    Returns:
        Tuple (train_idx, test_idx) of numpy index arrays.
    """
    classes = np.sort(np.unique(y))
    train_idx, test_idx = [], []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(np.floor(test_size * len(cls_idx))))
        test_idx.append(cls_idx[:n_test])
        train_idx.append(cls_idx[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


class KFold:
    """K-Fold cross-validator, optionally stratified.

    Splits data into k consecutive (but shuffled when shuffle=True) folds.
    When stratify=True, folds are built per-class and interleaved so each
    fold has approximately the same class distribution as the full dataset.

    Args:
        n_splits: Number of folds (k). Must be >= 2.
        shuffle: Whether to shuffle each class's samples before splitting.
        random_state: Seed for reproducibility (used only when shuffle=True).
        stratify: If True, preserve class proportions across folds.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42, stratify=True):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y):
        """Generate indices to split data into training and validation sets.

        Args:
            X: Array-like of shape (n_samples, ...). Only length is used.
            y: 1-D array-like of labels.

        Yields:
            Tuple (train_idx, val_idx) of numpy index arrays for each fold.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.random_state)

        if self.stratify:
            fold_indices = self._stratified_folds(y, rng)
        else:
            indices = np.arange(n)
            if self.shuffle:
                rng.shuffle(indices)
            fold_indices = np.array_split(indices, self.n_splits)

        for k in range(self.n_splits):
            val_idx = fold_indices[k]
            train_idx = np.concatenate([fold_indices[i] for i in range(self.n_splits) if i != k])
            yield train_idx, val_idx

    def _stratified_folds(self, y, rng):
        """Build stratified folds by distributing each class round-robin.

        For each class, optionally shuffles, splits into n_splits chunks,
        and assigns chunk i to fold i. All class chunks for fold i are
        concatenated to form that fold's indices.

        Args:
            y: 1-D numpy array of labels.
            rng: np.random.Generator instance.

        Returns:
            List of n_splits numpy index arrays (one per fold).
        """
        classes = np.sort(np.unique(y))
        fold_bins = [[] for _ in range(self.n_splits)]
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            if self.shuffle:
                rng.shuffle(cls_idx)
            chunks = np.array_split(cls_idx, self.n_splits)
            for k, chunk in enumerate(chunks):
                fold_bins[k].append(chunk)
        return [np.concatenate(bins) for bins in fold_bins]


if __name__ == "__main__":
    # Sanity check: verify stratified proportions are preserved
    rng = np.random.default_rng(0)
    y = np.array([0] * 70 + [1] * 30)  # 70/30 split
    X = np.arange(len(y)).reshape(-1, 1)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(y_tr)}, Test size: {len(y_te)}")
    print(f"Train class-1 rate: {y_tr.mean():.3f} (expected ~0.30)")
    print(f"Test  class-1 rate: {y_te.mean():.3f} (expected ~0.30)")

    print("\nKFold (stratified, k=5):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42, stratify=True)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        rate = y[va_idx].mean()
        print(f"  Fold {fold}: val size={len(va_idx)}, class-1 rate={rate:.3f}")
