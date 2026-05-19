"""
Generate all report figures from trained classifiers.

Run from the project root:
    python report/generate_figures.py

Outputs go to report/figures/.  All algorithm configs match the verified
notebook runs (best configs are hardcoded per the spec).
"""

import sys
import os
import io

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data_loader import load_heart_disease, get_feature_types, impute_missing, encode_categoricals
from common.preprocessing import StandardScaler, Binner
from common.split import train_test_split, KFold
from common.metrics import accuracy, confusion_matrix

from hw3.knn import KNN
from hw3.naive_bayes import DiscreteNaiveBayes
from hw4.linear_classifier import LeastSquaresClassifier
from hw4.logistic_regression import LogisticRegression
from hw5.lda_qda import LDA, QDA
from hw5.decision_tree import DecisionTreeClassifier

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SAVE_KW = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"  saved {name}")


def plot_confusion(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    savefig(filename)


def load_and_preprocess():
    """Standard preprocessing pipeline used by all HWs (no scaling/binning)."""
    df = load_heart_disease("data/heart_disease_uci.csv")
    feature_types = get_feature_types()
    cont_cols = feature_types["continuous"]
    cat_cols = feature_types["categorical"]

    df = impute_missing(df, cont_cols, cat_cols, zero_coded_cols=feature_types["zero_coded_missing"])
    df, _ = encode_categoricals(df, cat_cols)
    df = df.drop(columns=["num"])

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# HW3 Figures
# ---------------------------------------------------------------------------

def generate_hw3_figures(X_train, X_test, y_train, y_test):
    print("Generating HW3 figures...")

    # Scale for kNN
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # --- kNN k sweep ---
    ks = [1, 3, 5, 7, 11, 15, 21, 31]
    acc_euc = []
    acc_man = []
    for k in ks:
        knn_e = KNN(k=k, distance="euclidean").fit(X_tr_sc, y_train)
        knn_m = KNN(k=k, distance="manhattan").fit(X_tr_sc, y_train)
        acc_euc.append(accuracy(y_test, knn_e.predict(X_te_sc)))
        acc_man.append(accuracy(y_test, knn_m.predict(X_te_sc)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, acc_euc, marker="o", label="Euclidean")
    ax.plot(ks, acc_man, marker="s", label="Manhattan")
    ax.set_xlabel("k")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("kNN: Accuracy vs k")
    ax.legend()
    ax.set_xticks(ks)
    savefig("hw3_knn_k_sweep.png")

    # --- kNN best confusion ---
    best_knn = KNN(k=31, distance="manhattan", weights="uniform").fit(X_tr_sc, y_train)
    cm = confusion_matrix(y_test, best_knn.predict(X_te_sc))
    plot_confusion(cm, "kNN (k=31, Manhattan) — Test Set", "hw3_knn_confusion.png")

    # Bin for NB
    binner = Binner(n_bins=5, strategy="quantile")
    X_tr_bin = binner.fit_transform(X_train)
    X_te_bin = binner.transform(X_test)

    # --- NB alpha sweep ---
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    acc_alpha = []
    for a in alphas:
        nb = DiscreteNaiveBayes(alpha=a).fit(X_tr_bin, y_train)
        acc_alpha.append(accuracy(y_test, nb.predict(X_te_bin)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(alphas, acc_alpha, marker="o")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Naive Bayes: Accuracy vs alpha (n_bins=5)")
    ax.set_xticks(alphas)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    savefig("hw3_nb_alpha_sweep.png")

    # --- NB bins sweep ---
    bins_list = [3, 5, 7, 10]
    acc_bins = []
    for nb_val in bins_list:
        b = Binner(n_bins=nb_val, strategy="quantile")
        Xtr_b = b.fit_transform(X_train)
        Xte_b = b.transform(X_test)
        nb = DiscreteNaiveBayes(alpha=1.0).fit(Xtr_b, y_train)
        acc_bins.append(accuracy(y_test, nb.predict(Xte_b)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bins_list, acc_bins, marker="o")
    ax.set_xlabel("n_bins")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Naive Bayes: Accuracy vs n_bins (alpha=1.0)")
    ax.set_xticks(bins_list)
    savefig("hw3_nb_bins_sweep.png")

    # --- NB best confusion (best config: alpha=0.01, n_bins=7) ---
    best_binner = Binner(n_bins=7, strategy="quantile")
    X_tr_best = best_binner.fit_transform(X_train)
    X_te_best = best_binner.transform(X_test)
    best_nb = DiscreteNaiveBayes(alpha=0.01).fit(X_tr_best, y_train)
    cm_nb = confusion_matrix(y_test, best_nb.predict(X_te_best))
    plot_confusion(cm_nb, "Naive Bayes (alpha=0.01, n_bins=7) — Test Set",
                   "hw3_nb_confusion.png")


# ---------------------------------------------------------------------------
# HW4 Figures
# ---------------------------------------------------------------------------

def generate_hw4_figures(X_train, X_test, y_train, y_test, feature_names):
    print("Generating HW4 figures...")

    # Scale for LS and LR
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # --- LS lambda sweep ---
    lambdas_raw = [0, 0.01, 0.1, 1.0, 10.0, 100.0]
    acc_ls = []
    for lam in lambdas_raw:
        ls = LeastSquaresClassifier(regularization=lam).fit(X_tr_sc, y_train)
        acc_ls.append(accuracy(y_test, ls.predict(X_te_sc)))

    # Plot λ=0 at x=1e-3 on log axis
    lambdas_plot = [1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
    labels_plot = ["0 (→1e-3)", "0.01", "0.1", "1.0", "10.0", "100.0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lambdas_plot, acc_ls, marker="o")
    ax.set_xlabel("Ridge λ (log scale; λ=0 plotted at 1e-3)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Least-Squares: Accuracy vs Ridge λ")
    ax.set_xticks(lambdas_plot)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(labels_plot))
    savefig("hw4_ls_lambda_sweep.png")

    # --- LS score histogram ---
    best_ls = LeastSquaresClassifier(regularization=0).fit(X_tr_sc, y_train)
    scores_ls = best_ls.decision_function(X_te_sc)
    mask0 = y_test == 0
    mask1 = y_test == 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores_ls[mask0], bins=20, alpha=0.6, label="Class 0", color="steelblue")
    ax.hist(scores_ls[mask1], bins=20, alpha=0.6, label="Class 1", color="tomato")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="threshold=0")
    ax.set_xlabel("Decision Function Score")
    ax.set_ylabel("Count")
    ax.set_title("Least-Squares: Score Distribution on Test Set")
    ax.legend()
    savefig("hw4_ls_score_histogram.png")

    # --- LS confusion ---
    cm_ls = confusion_matrix(y_test, best_ls.predict(X_te_sc))
    plot_confusion(cm_ls, "Least-Squares (λ=0) — Test Set", "hw4_ls_confusion.png")

    # --- LR learning-rate sweep (loss curves) ---
    lrs = [0.001, 0.01, 0.1, 1.0]
    fig, ax = plt.subplots(figsize=(10, 6))
    for lr_val in lrs:
        lr_clf = LogisticRegression(
            learning_rate=lr_val, n_epochs=2000, regularization=0.0, tol=0
        ).fit(X_tr_sc, y_train)
        ax.semilogy(lr_clf.loss_history_, label=f"lr={lr_val}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Logistic Regression: Loss Curves for Different Learning Rates")
    ax.legend()
    savefig("hw4_lr_lr_sweep.png")

    # --- LR final-config loss curve ---
    best_lr = LogisticRegression(
        learning_rate=0.1, n_epochs=2000, regularization=0.1, tol=0
    ).fit(X_tr_sc, y_train)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(best_lr.loss_history_)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Logistic Regression: Convergence (lr=0.1, λ=0.1, 2000 epochs)")
    savefig("hw4_lr_loss_curve.png")

    # --- LR lambda sweep ---
    lr_lambdas = [0, 0.01, 0.1, 1.0, 10.0]
    acc_lr_lam = []
    for lam in lr_lambdas:
        clf = LogisticRegression(
            learning_rate=0.1, n_epochs=2000, regularization=lam, tol=0
        ).fit(X_tr_sc, y_train)
        acc_lr_lam.append(accuracy(y_test, clf.predict(X_te_sc)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lr_lambdas, acc_lr_lam, marker="o")
    ax.set_xlabel("L2 λ")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Logistic Regression: Accuracy vs L2 λ")
    ax.set_xticks(lr_lambdas)
    savefig("hw4_lr_lambda_sweep.png")

    # --- LR coefficients bar chart ---
    coefs = best_lr.coef_
    intercept = best_lr.intercept_
    all_coefs = np.append(coefs, intercept)
    all_names = list(feature_names) + ["intercept"]

    order = np.argsort(np.abs(all_coefs))[::-1]
    sorted_coefs = all_coefs[order]
    sorted_names = [all_names[i] for i in order]
    colors = ["steelblue"] * len(coefs) + ["darkorange"]
    sorted_colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_coefs)), sorted_coefs, color=sorted_colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Logistic Regression: Feature Coefficients (sorted by |coef|)")
    ax.invert_yaxis()

    # Legend patches
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color="steelblue", label="Feature weight")
    orange_patch = mpatches.Patch(color="darkorange", label="Intercept")
    ax.legend(handles=[blue_patch, orange_patch])
    savefig("hw4_lr_coefficients.png")

    # --- LR confusion ---
    cm_lr = confusion_matrix(y_test, best_lr.predict(X_te_sc))
    plot_confusion(cm_lr, "Logistic Regression (lr=0.1, λ=0.1) — Test Set",
                   "hw4_lr_confusion.png")

    # --- Score comparison scatter ---
    scores_lr = best_lr.decision_function(X_te_sc)
    scores_ls2 = best_ls.decision_function(X_te_sc)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color, label in [(0, "steelblue", "Class 0"), (1, "tomato", "Class 1")]:
        mask = y_test == cls
        ax.scatter(scores_ls2[mask], scores_lr[mask], alpha=0.6,
                   color=color, label=label, s=40)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Least-Squares Decision Function")
    ax.set_ylabel("Logistic Regression Logit")
    ax.set_title("LS vs LR Decision Scores on Test Set")
    ax.legend()
    savefig("hw4_score_comparison.png")


# ---------------------------------------------------------------------------
# HW5 Figures
# ---------------------------------------------------------------------------

def generate_hw5_figures(X_train, X_test, y_train, y_test, feature_names):
    print("Generating HW5 figures...")

    # Scale for LDA/QDA and Decision Tree (tree doesn't need scaling but consistent)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # --- LDA ---
    lda = LDA(reg_param=1e-6).fit(X_tr_sc, y_train)
    cm_lda = confusion_matrix(y_test, lda.predict(X_te_sc))
    plot_confusion(cm_lda, "LDA — Test Set", "hw5_lda_confusion.png")

    # --- QDA ---
    qda = QDA(reg_param=1e-6).fit(X_tr_sc, y_train)
    cm_qda = confusion_matrix(y_test, qda.predict(X_te_sc))
    plot_confusion(cm_qda, "QDA — Test Set", "hw5_qda_confusion.png")

    # --- LDA vs QDA per-feature variance ratio ---
    # Per-class covariances for QDA; compare diagonal (variance) of class1 vs class0
    cov0 = qda.covariances_[0]
    cov1 = qda.covariances_[1]
    var_ratio = np.diag(cov1) / np.diag(cov0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(feature_names))
    bars = ax.bar(x_pos, var_ratio)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="ratio = 1 (LDA assumption)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Var(class 1) / Var(class 0)  [QDA diagonal]")
    ax.set_title("Per-Feature Variance Ratio: Class 1 vs Class 0\n(ratio ≠ 1 justifies QDA over LDA)")
    ax.legend()
    savefig("hw5_lda_qda_variance.png")

    # --- Decision Tree: depth sweep ---
    # Decision tree uses original (unscaled) features
    depths = [2, 3, 5, 7, 10, 15, None]
    depth_labels = [str(d) if d is not None else "None" for d in depths]
    train_accs = []
    test_accs = []
    n_nodes_list = []

    for d in depths:
        dt = DecisionTreeClassifier(criterion="gini", min_samples_leaf=1, max_depth=d).fit(X_train, y_train)
        train_accs.append(accuracy(y_train, dt.predict(X_train)))
        test_accs.append(accuracy(y_test, dt.predict(X_test)))
        n_nodes_list.append(dt.n_nodes_)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(depths))
    ax1.plot(x_pos, train_accs, marker="o", label="Train Accuracy", color="steelblue")
    ax1.plot(x_pos, test_accs, marker="s", label="Test Accuracy", color="tomato")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(depth_labels)
    ax1.set_title("Decision Tree: Accuracy and Size vs max_depth")
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.bar(x_pos, n_nodes_list, alpha=0.25, color="green", label="n_nodes")
    ax2.set_ylabel("Number of Nodes", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right")
    savefig("hw5_tree_depth_sweep.png")

    # --- Decision Tree: min_samples_leaf sweep ---
    leaf_vals = [1, 2, 5, 10, 20, 50]
    test_accs_leaf = []
    n_nodes_leaf = []

    for mlf in leaf_vals:
        dt = DecisionTreeClassifier(criterion="gini", min_samples_leaf=mlf, max_depth=None).fit(X_train, y_train)
        test_accs_leaf.append(accuracy(y_test, dt.predict(X_test)))
        n_nodes_leaf.append(dt.n_nodes_)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(leaf_vals, test_accs_leaf, marker="o", color="tomato", label="Test Accuracy")
    ax1.set_xlabel("min_samples_leaf")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Decision Tree: Accuracy and Size vs min_samples_leaf")
    ax1.set_xticks(leaf_vals)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.bar(leaf_vals, n_nodes_leaf, alpha=0.25, color="green", label="n_nodes", width=2)
    ax2.set_ylabel("Number of Nodes", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right")
    savefig("hw5_tree_min_leaf_sweep.png")

    # --- Decision Tree: gini vs entropy ---
    dt_gini = DecisionTreeClassifier(criterion="gini", min_samples_leaf=20).fit(X_train, y_train)
    dt_ent = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20).fit(X_train, y_train)

    acc_gini = accuracy(y_test, dt_gini.predict(X_test))
    acc_ent = accuracy(y_test, dt_ent.predict(X_test))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    criteria = ["gini", "entropy"]
    accs = [acc_gini, acc_ent]
    nodes = [dt_gini.n_nodes_, dt_ent.n_nodes_]

    axes[0].bar(criteria, accs, color=["steelblue", "tomato"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Test Accuracy by Criterion")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

    axes[1].bar(criteria, nodes, color=["steelblue", "tomato"])
    axes[1].set_ylabel("Number of Nodes")
    axes[1].set_title("Tree Size by Criterion")
    for i, v in enumerate(nodes):
        axes[1].text(i, v + 0.2, str(v), ha="center")

    fig.suptitle("Decision Tree: Gini vs Entropy (min_samples_leaf=20)")
    plt.tight_layout()
    savefig("hw5_tree_gini_vs_entropy.png")

    # --- Best tree confusion ---
    best_tree = DecisionTreeClassifier(criterion="gini", min_samples_leaf=20, max_depth=None).fit(X_train, y_train)
    cm_tree = confusion_matrix(y_test, best_tree.predict(X_test))
    plot_confusion(cm_tree, "Decision Tree (gini, min_leaf=20) — Test Set",
                   "hw5_tree_confusion.png")

    # --- Tree structure text ---
    buf = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = buf
    best_tree.print_tree(feature_names=feature_names, max_lines=200)
    sys.stdout = _old_stdout
    tree_text = buf.getvalue()

    txt_path = os.path.join(OUT, "hw5_tree_structure.txt")
    with open(txt_path, "w") as f:
        f.write(f"Decision Tree (criterion=gini, min_samples_leaf=20, max_depth=None)\n")
        f.write(f"n_nodes={best_tree.n_nodes_}  max_depth_reached={best_tree.max_depth_reached_}\n\n")
        f.write(tree_text)
    print("  saved hw5_tree_structure.txt")


# ---------------------------------------------------------------------------
# Cross-HW Figure
# ---------------------------------------------------------------------------

def generate_cross_hw_figure(X_train, X_test, y_train, y_test, feature_names):
    print("Generating cross-HW comparison figure...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42, stratify=True)

    # Scaled data for kNN, LS, LR, LDA, QDA
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Binned data for NB
    binner = Binner(n_bins=5, strategy="quantile")
    X_tr_bin = binner.fit_transform(X_train)
    X_te_bin = binner.transform(X_test)

    # Best-config classifiers
    classifiers = {
        "kNN": (KNN(k=31, distance="manhattan", weights="uniform"), X_tr_sc, X_te_sc),
        "NB": (DiscreteNaiveBayes(alpha=0.01), X_tr_bin, X_te_bin),
        "LS": (LeastSquaresClassifier(regularization=0), X_tr_sc, X_te_sc),
        "LR": (LogisticRegression(learning_rate=0.1, n_epochs=2000, regularization=0.1, tol=0),
               X_tr_sc, X_te_sc),
        "LDA": (LDA(reg_param=1e-6), X_tr_sc, X_te_sc),
        "QDA": (QDA(reg_param=1e-6), X_tr_sc, X_te_sc),
        "Tree": (DecisionTreeClassifier(criterion="gini", min_samples_leaf=20, max_depth=None),
                 X_train, X_test),
    }

    test_accs = {}
    cv_means = {}
    cv_stds = {}

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    for name, (clf_template, X_tr, X_te) in classifiers.items():
        # Test accuracy (fit on train, evaluate on test)
        import copy
        clf = copy.deepcopy(clf_template)
        clf.fit(X_tr, y_train)
        test_accs[name] = accuracy(y_test, clf.predict(X_te))

        # 5-fold CV on full dataset
        # Rebuild scaled/binned versions on each fold
        fold_accs = []
        for tr_idx, val_idx in kf.split(X_full, y_full):
            X_f_tr, X_f_val = X_full[tr_idx], X_full[val_idx]
            y_f_tr, y_f_val = y_full[tr_idx], y_full[val_idx]

            if name == "NB":
                b = Binner(n_bins=5, strategy="quantile")
                Xf_tr_t = b.fit_transform(X_f_tr)
                Xf_val_t = b.transform(X_f_val)
            elif name == "Tree":
                Xf_tr_t = X_f_tr
                Xf_val_t = X_f_val
            else:
                sc = StandardScaler()
                Xf_tr_t = sc.fit_transform(X_f_tr)
                Xf_val_t = sc.transform(X_f_val)

            fold_clf = copy.deepcopy(clf_template)
            fold_clf.fit(Xf_tr_t, y_f_tr)
            fold_accs.append(accuracy(y_f_val, fold_clf.predict(Xf_val_t)))

        cv_means[name] = np.mean(fold_accs)
        cv_stds[name] = np.std(fold_accs)

    algo_names = list(classifiers.keys())
    x = np.arange(len(algo_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, [test_accs[n] for n in algo_names], width,
                   label="Test Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, [cv_means[n] for n in algo_names], width,
                   yerr=[cv_stds[n] for n in algo_names], capsize=4,
                   label="5-Fold CV Mean ± Std", color="tomato")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-HW Comparison: Test Accuracy vs 5-Fold CV")
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Annotate bar values
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    savefig("cross_hw_comparison.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Output directory: {OUT}")

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
    print(f"Data loaded: train={X_train.shape}, test={X_test.shape}, features={feature_names}")

    generate_hw3_figures(X_train, X_test, y_train, y_test)
    generate_hw4_figures(X_train, X_test, y_train, y_test, feature_names)
    generate_hw5_figures(X_train, X_test, y_train, y_test, feature_names)
    generate_cross_hw_figure(X_train, X_test, y_train, y_test, feature_names)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
