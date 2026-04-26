"""
Decision Tree Classifier from scratch (HW5 Part 2).

CART-style classification tree for continuous features.  Supports Gini
impurity and information entropy as splitting criteria.

Threshold-split strategy: for each feature, candidate thresholds are the
midpoints between consecutive unique values.  This avoids redundant splits
(identical threshold pairs produce identical splits) and is asymptotically
equivalent to trying all distinct cut-points.

Vectorized impurity evaluation: samples are sorted once per feature; a
cumulative class-count matrix (K × n_samples) then lets all candidate
thresholds be evaluated in one NumPy pass — O(n·K + T) per feature rather
than O(n·T·K) for a Python loop over thresholds.

From-scratch status:
    NumPy operations only.  No sklearn, scipy, or any library providing
    tree construction, impurity, or splitting.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _Node:
    """Internal representation of one node in the decision tree.

    Leaf nodes: is_leaf=True; prediction and class_distribution are set.
    Internal nodes: feature_index, threshold, left, right are set.
    Both: n_samples and impurity are set (useful for inspection).
    """

    __slots__ = (
        "is_leaf", "prediction", "class_distribution",
        "feature_index", "threshold", "left", "right",
        "n_samples", "impurity",
    )

    def __init__(self):
        self.is_leaf = False
        self.prediction = None
        self.class_distribution = None
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.n_samples = 0
        self.impurity = 0.0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """CART-style decision tree for multi-class classification.

    Handles continuous features via threshold splits.  All four standard
    stopping criteria are enforced simultaneously during tree construction.

    Leaf prediction: majority-vote class.  Ties are broken by the lowest
    class label — deterministic because classes_ is sorted and argmax
    returns the first occurrence of the maximum.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default "gini"
        Impurity measure used to evaluate splits.
        Gini:    G = 1 - Σ_c p_c²
        Entropy: H = -Σ_c p_c log₂(p_c)  (with 0·log(0) = 0 convention)
    max_depth : int or None, default None
        Maximum tree depth.  None = no limit.
    min_samples_split : int, default 2
        A node with fewer than this many samples is made a leaf immediately.
    min_samples_leaf : int, default 1
        Any split creating a child with fewer than this many samples is
        rejected, even if it would reduce impurity.
    min_impurity_decrease : float, default 0.0
        A split is accepted only if
            gain × (n_node / n_total) ≥ min_impurity_decrease
        where n_total is the full training-set size and n_node is the
        current node size.  This down-weights gains from small (deep) nodes,
        requiring them to show proportionally more improvement to justify a
        split.  Matches the sklearn convention.
    random_state : int, default 42
        Kept for API consistency with other classifiers in this project.
        Currently unused — all tie-breaking is deterministic.

    Attributes (after fit)
    ----------------------
    classes_ : ndarray (K,) — sorted unique class labels
    n_classes_ : int
    root_ : _Node — root of the built tree
    n_nodes_ : int — total number of nodes (leaves + internal)
    max_depth_reached_ : int — actual depth of the deepest leaf
    feature_names_ : list of str or None — set if X was a DataFrame
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got {criterion!r}"
            )
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError(
                f"max_depth must be a positive int or None, got {max_depth!r}"
            )
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be an int >= 2, got {min_samples_split!r}"
            )
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be an int >= 1, got {min_samples_leaf!r}"
            )
        if min_impurity_decrease < 0:
            raise ValueError(
                f"min_impurity_decrease must be >= 0, got {min_impurity_decrease!r}"
            )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.root_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_nodes_ = 0
        self.max_depth_reached_ = 0
        self.feature_names_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Build the decision tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            If a pandas DataFrame, column names are stored in feature_names_.
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                self.feature_names_ = list(X.columns)
        except ImportError:
            pass

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        n_total = len(y)

        self.n_nodes_ = 0
        self.max_depth_reached_ = 0
        self.root_ = self._build(X, y, depth=0, n_total=n_total)
        return self

    def predict(self, X):
        """Predict class labels for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) — values from self.classes_
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.array([self._route(x, self.root_).prediction for x in X])

    def predict_proba(self, X):
        """Class probability estimates from the leaf each sample reaches.

        Probabilities are empirical class proportions stored at the leaf —
        no Laplace smoothing is applied.  Classes absent from a leaf get
        probability 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) — rows sum to 1
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.array(
            [self._route(x, self.root_).class_distribution for x in X]
        )

    def print_tree(self, feature_names=None, max_lines: int = 50):
        """Print an ASCII representation of the tree.

        Each internal node shows: feature name, split threshold, sample count,
        and impurity.  Each leaf shows: predicted class, sample count, impurity.
        Output is truncated at max_lines with a summary note.

        Parameters
        ----------
        feature_names : list of str or None
            Display names for feature columns.  Falls back to self.feature_names_
            then to "X[j]" index notation.
        max_lines : int, default 50
            Maximum lines before truncation.
        """
        self._check_fitted()
        names = feature_names or self.feature_names_
        lines = []
        self._collect_lines(self.root_, prefix="", is_last=True,
                            feature_names=names, lines=lines)
        truncated = len(lines) > max_lines
        for line in lines[:max_lines]:
            print(line)
        if truncated:
            print(
                f"  ... truncated ({self.n_nodes_} nodes total, "
                f"depth reached {self.max_depth_reached_})"
            )

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build(self, X, y, depth, n_total):
        node = _Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        self.n_nodes_ += 1
        self.max_depth_reached_ = max(self.max_depth_reached_, depth)

        # Stop if pure, too small, or depth limit reached
        stop = (
            node.impurity == 0.0
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return self._make_leaf(node, y)

        split = self._best_split(X, y, n_total)
        if split is None:
            return self._make_leaf(node, y)

        feat_idx, threshold, _ = split
        left_mask = X[:, feat_idx] <= threshold
        node.feature_index = feat_idx
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1, n_total)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1, n_total)
        return node

    def _make_leaf(self, node, y):
        node.is_leaf = True
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        node.class_distribution = counts / counts.sum()
        # Majority class; first max used for tie-breaking (lowest class label)
        node.prediction = self.classes_[int(np.argmax(counts))]
        return node

    # ------------------------------------------------------------------
    # Split finding — vectorized over thresholds per feature
    # ------------------------------------------------------------------

    def _best_split(self, X, y, n_total):
        """Return (feature_idx, threshold, gain) for the globally best split.

        For each feature:
        1. Sort samples by feature value.
        2. Build a (K × n_samples) cumulative class-count matrix.
        3. Use searchsorted to map each midpoint threshold to a split
           position, then read off left/right class counts in one indexing
           step — no Python loop over thresholds.
        4. Compute impurity and gain vectorized over all thresholds.

        Returns None if no valid split exists (all stopping criteria fail
        or all features are constant).
        """
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for j in range(n_features):
            col = X[:, j]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            # Midpoints between consecutive unique values as candidates
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            # Sort once; build cumulative class-count matrix (K, n_samples)
            sorted_idx = np.argsort(col, kind="stable")
            sorted_col = col[sorted_idx]
            sorted_y = y[sorted_idx]

            # class_indicators[k, i] = 1 iff sorted sample i is class k
            class_ind = (
                sorted_y[np.newaxis, :] == self.classes_[:, np.newaxis]
            ).astype(np.int32)                                  # (K, n)
            cum = class_ind.cumsum(axis=1)                      # (K, n)

            # Split position t: left contains sorted samples [0 .. t], right [t+1 .. n-1]
            # searchsorted(side='right') returns insertion point after equal values,
            # so subtracting 1 gives the last index with value <= threshold.
            split_pos = np.searchsorted(sorted_col, thresholds, side="right") - 1

            # Valid: left non-empty (>= 0) and right non-empty (< n_samples - 1)
            valid = (split_pos >= 0) & (split_pos < n_samples - 1)
            split_pos_v = split_pos[valid]
            thresholds_v = thresholds[valid]
            if len(split_pos_v) == 0:
                continue

            # Left / right class counts: (K, T) where T = number of valid thresholds
            left_counts = cum[:, split_pos_v].astype(float)           # (K, T)
            total_counts = cum[:, -1].astype(float)                    # (K,)
            right_counts = total_counts[:, np.newaxis] - left_counts   # (K, T)

            n_lefts = (split_pos_v + 1).astype(float)                  # (T,)
            n_rights = float(n_samples) - n_lefts                      # (T,)

            # Reject splits violating min_samples_leaf
            leaf_ok = (
                (n_lefts >= self.min_samples_leaf)
                & (n_rights >= self.min_samples_leaf)
            )
            if not leaf_ok.any():
                continue

            # Impurity of children, vectorized over T thresholds
            left_probs = left_counts / n_lefts[np.newaxis, :]    # (K, T)
            right_probs = right_counts / n_rights[np.newaxis, :] # (K, T)

            if self.criterion == "gini":
                left_imp = 1.0 - (left_probs ** 2).sum(axis=0)   # (T,)
                right_imp = 1.0 - (right_probs ** 2).sum(axis=0) # (T,)
            else:  # entropy
                with np.errstate(divide="ignore", invalid="ignore"):
                    ll = np.where(left_probs > 0, np.log2(left_probs), 0.0)
                    rl = np.where(right_probs > 0, np.log2(right_probs), 0.0)
                left_imp = -(left_probs * ll).sum(axis=0)
                right_imp = -(right_probs * rl).sum(axis=0)

            weighted_child = (n_lefts * left_imp + n_rights * right_imp) / n_samples
            gain = parent_impurity - weighted_child               # (T,)

            # min_impurity_decrease: gain × (n_node / n_total) must clear threshold
            impurity_ok = (
                gain * (n_samples / n_total) >= self.min_impurity_decrease
            )
            valid_split = leaf_ok & impurity_ok
            if not valid_split.any():
                continue

            # Best threshold for this feature (first argmax → lowest label tie-break)
            masked = np.where(valid_split, gain, -np.inf)
            best_local = int(np.argmax(masked))
            if gain[best_local] > best_gain:
                best_gain = float(gain[best_local])
                best_feature = j
                best_threshold = float(thresholds_v[best_local])

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------
    # Impurity
    # ------------------------------------------------------------------

    def _impurity(self, y):
        """Gini or entropy impurity for a label array at one node."""
        n = len(y)
        if n == 0:
            return 0.0
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        probs = counts / n
        if self.criterion == "gini":
            return float(1.0 - (probs ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(probs > 0, np.log2(probs), 0.0)
        return float(-(probs * lp).sum())

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _route(self, x, node):
        """Return the leaf node reached by sample x."""
        if node.is_leaf:
            return node
        if x[node.feature_index] <= node.threshold:
            return self._route(x, node.left)
        return self._route(x, node.right)

    # ------------------------------------------------------------------
    # print_tree helpers
    # ------------------------------------------------------------------

    def _feat_name(self, j, feature_names):
        if feature_names and j < len(feature_names):
            return feature_names[j]
        return f"X[{j}]"

    def _collect_lines(self, node, prefix, is_last, feature_names, lines):
        connector = "└── " if is_last else "├── "
        if node.is_leaf:
            lines.append(
                f"{prefix}{connector}Leaf: predict={node.prediction}  "
                f"n={node.n_samples}  impurity={node.impurity:.4f}"
            )
        else:
            fname = self._feat_name(node.feature_index, feature_names)
            lines.append(
                f"{prefix}{connector}{fname} <= {node.threshold:.4f}  "
                f"n={node.n_samples}  impurity={node.impurity:.4f}"
            )
            child_prefix = prefix + ("    " if is_last else "│   ")
            self._collect_lines(node.left, child_prefix, False, feature_names, lines)
            self._collect_lines(node.right, child_prefix, True, feature_names, lines)

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict() or predict_proba().")


# ---------------------------------------------------------------------------
# Sanity tests (synthetic data only — no real dataset)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(99)
    print("=" * 60)
    print("DecisionTreeClassifier Sanity Tests")
    print("=" * 60)

    # --- Iris-like synthetic: 3 classes, 4 features, 150 samples ---
    def make_iris_like(rng, n_per_class=50):
        means = [
            np.array([5.0, 3.5, 1.5, 0.3]),
            np.array([5.9, 2.8, 4.3, 1.3]),
            np.array([6.6, 3.0, 5.6, 2.0]),
        ]
        cov = np.diag([0.3, 0.1, 0.4, 0.05])
        Xs = [rng.multivariate_normal(mu, cov, size=n_per_class) for mu in means]
        X = np.vstack(Xs)
        y = np.repeat([0, 1, 2], n_per_class)
        return X, y

    X_iris, y_iris = make_iris_like(rng)

    # Test 1: unconstrained tree should memorize
    print("\n--- Test 1: Unconstrained tree (should memorize training data) ---")
    dt = DecisionTreeClassifier(criterion="gini")
    dt.fit(X_iris, y_iris)
    train_acc = np.mean(dt.predict(X_iris) == y_iris)
    assert train_acc == 1.0, f"Expected 100% train acc, got {train_acc:.4f}"
    print(f"[PASS] Training accuracy = {train_acc:.4f} (100% expected)")
    print(f"       Nodes: {dt.n_nodes_}  Max depth reached: {dt.max_depth_reached_}")

    # Test 2: max_depth=2 reduces size and accuracy
    print("\n--- Test 2: max_depth=2 reduces tree size ---")
    dt2 = DecisionTreeClassifier(criterion="gini", max_depth=2)
    dt2.fit(X_iris, y_iris)
    acc2 = np.mean(dt2.predict(X_iris) == y_iris)
    assert dt2.n_nodes_ < dt.n_nodes_, (
        f"max_depth=2 tree should have fewer nodes: {dt2.n_nodes_} vs {dt.n_nodes_}"
    )
    assert dt2.max_depth_reached_ <= 2
    print(f"[PASS] max_depth=2: nodes={dt2.n_nodes_}, depth={dt2.max_depth_reached_}, "
          f"train acc={acc2:.4f}")

    # Test 3: predict_proba rows sum to 1
    print("\n--- Test 3: predict_proba rows sum to 1 ---")
    proba = dt.predict_proba(X_iris)
    assert proba.shape == (150, 3), f"proba shape: {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Max row-sum dev: {np.abs(row_sums-1).max():.2e}"
    print(f"[PASS] predict_proba shape={proba.shape}, rows sum to 1")

    # Test 4: entropy criterion produces reasonable accuracy
    print("\n--- Test 4: criterion='entropy' ---")
    dt_ent = DecisionTreeClassifier(criterion="entropy")
    dt_ent.fit(X_iris, y_iris)
    acc_ent = np.mean(dt_ent.predict(X_iris) == y_iris)
    assert acc_ent > 0.90, f"Entropy tree acc too low: {acc_ent:.4f}"
    print(f"[PASS] entropy tree: train acc={acc_ent:.4f}, nodes={dt_ent.n_nodes_}")

    # --- Edge cases ---
    print("\n--- Test 5: Single-class y → single leaf ---")
    X_one = rng.normal(size=(20, 3))
    y_one = np.zeros(20, dtype=int)
    dt_one = DecisionTreeClassifier()
    dt_one.fit(X_one, y_one)
    assert dt_one.n_nodes_ == 1 and dt_one.root_.is_leaf
    assert np.all(dt_one.predict(X_one) == 0)
    print(f"[PASS] Single-class: n_nodes={dt_one.n_nodes_}, "
          f"prediction={dt_one.root_.prediction}")

    print("\n--- Test 6: n_samples=1 → single leaf ---")
    dt_tiny = DecisionTreeClassifier()
    dt_tiny.fit(np.array([[1.0, 2.0]]), np.array([7]))
    assert dt_tiny.n_nodes_ == 1 and dt_tiny.root_.is_leaf
    assert dt_tiny.predict(np.array([[1.0, 2.0]]))[0] == 7
    print(f"[PASS] n=1 sample: prediction={dt_tiny.root_.prediction}")

    print("\n--- Test 7: All-identical features → single leaf ---")
    X_const = np.ones((30, 4)) * 3.14
    y_const = np.array([0, 1] * 15)
    dt_const = DecisionTreeClassifier()
    dt_const.fit(X_const, y_const)
    assert dt_const.n_nodes_ == 1 and dt_const.root_.is_leaf
    print(f"[PASS] Constant features: single leaf, "
          f"prediction={dt_const.root_.prediction}")

    print("\n--- Test 8: min_samples_leaf=10 prevents small leaves ---")
    dt_leaf = DecisionTreeClassifier(min_samples_leaf=10)
    dt_leaf.fit(X_iris, y_iris)
    # Traverse all leaves and check their size
    def all_leaves(node):
        if node.is_leaf:
            yield node
        else:
            yield from all_leaves(node.left)
            yield from all_leaves(node.right)
    min_leaf_n = min(n.n_samples for n in all_leaves(dt_leaf.root_))
    assert min_leaf_n >= 10, f"Found leaf with {min_leaf_n} samples < 10"
    print(f"[PASS] min_samples_leaf=10: smallest leaf has {min_leaf_n} samples")

    print("\n--- Test 9: min_impurity_decrease=0.5 produces small tree ---")
    dt_imdec = DecisionTreeClassifier(min_impurity_decrease=0.5)
    dt_imdec.fit(X_iris, y_iris)
    # With a high threshold, the tree should be much smaller than unconstrained
    assert dt_imdec.n_nodes_ < dt.n_nodes_, (
        f"min_impurity_decrease=0.5 didn't constrain the tree: "
        f"{dt_imdec.n_nodes_} vs {dt.n_nodes_}"
    )
    print(f"[PASS] min_impurity_decrease=0.5: nodes={dt_imdec.n_nodes_} "
          f"(unconstrained={dt.n_nodes_})")

    print("\n--- Test 10: print_tree runs on max_depth=3 tree ---")
    dt_print = DecisionTreeClassifier(max_depth=3)
    dt_print.fit(X_iris, y_iris)
    print("  Tree output:")
    dt_print.print_tree(feature_names=["sepal_l", "sepal_w", "petal_l", "petal_w"])
    print("[PASS] print_tree completed without error")

    print("\n--- Test 11: Constructor validation ---")
    for bad_kwargs, label in [
        ({"criterion": "mse"}, "bad criterion"),
        ({"max_depth": 0}, "max_depth=0"),
        ({"max_depth": 1.5}, "max_depth=float"),
        ({"min_samples_split": 1}, "min_samples_split=1"),
        ({"min_samples_leaf": 0}, "min_samples_leaf=0"),
        ({"min_impurity_decrease": -0.1}, "min_impurity_decrease<0"),
    ]:
        try:
            DecisionTreeClassifier(**bad_kwargs)
            raise AssertionError(f"Should have raised ValueError for {label}")
        except ValueError as exc:
            print(f"[PASS] {label}: {exc}")

    print("\n" + "=" * 60)
    print("All sanity checks passed.")
    print("=" * 60)
