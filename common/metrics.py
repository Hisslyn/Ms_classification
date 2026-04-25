"""Classification metrics implemented from scratch using NumPy only."""

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute the confusion matrix.

    Rows are true classes, columns are predicted classes.
    Entry [i, j] is the count of samples with true label labels[i]
    and predicted label labels[j].

    Args:
        y_true: 1-D array-like of true labels.
        y_pred: 1-D array-like of predicted labels.
        labels: Optional list of label values to include (defines row/col order).
                If None, sorted unique values from both arrays are used.

    Returns:
        np.ndarray of shape (n_classes, n_classes).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    else:
        labels = np.asarray(labels)
    n = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t], label_to_idx[p]] += 1
    return matrix


def accuracy(y_true, y_pred):
    """Fraction of correctly classified samples.

    Args:
        y_true: 1-D array-like of true labels.
        y_pred: 1-D array-like of predicted labels.

    Returns:
        float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def _per_class_prf(y_true, y_pred, labels):
    """Compute per-class TP, FP, FN counts.

    Returns:
        Tuple (tp, fp, fn) each of shape (n_classes,).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp, fp, fn


def precision(y_true, y_pred, average="binary", pos_label=1):
    """Precision score with multiple averaging strategies.

    Precision for class k = TP_k / (TP_k + FP_k). Returns 0.0 when the
    denominator is zero (no samples predicted as that class).

    Args:
        y_true: 1-D array-like of true labels.
        y_pred: 1-D array-like of predicted labels.
        average: One of "binary", "macro", "micro", "none".
                 "binary"  — precision for pos_label only.
                 "macro"   — unweighted mean over all classes.
                 "micro"   — globally pool TP/FP across classes.
                 "none"    — return per-class array.
        pos_label: The positive class for average="binary".

    Returns:
        float (for "binary", "macro", "micro") or np.ndarray (for "none").
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    tp, fp, fn = _per_class_prf(y_true, y_pred, labels)

    denom = tp + fp
    # Avoid division by zero: classes with no predicted positives get 0.0
    per_class = np.where(denom > 0, tp / denom, 0.0)

    if average == "none":
        return per_class
    if average == "macro":
        return float(np.mean(per_class))
    if average == "micro":
        micro_denom = tp.sum() + fp.sum()
        return float(tp.sum() / micro_denom) if micro_denom > 0 else 0.0
    if average == "binary":
        idx = np.where(labels == pos_label)[0]
        if len(idx) == 0:
            return 0.0
        return float(per_class[idx[0]])
    raise ValueError(f"Unknown average: {average!r}")


def recall(y_true, y_pred, average="binary", pos_label=1):
    """Recall score with multiple averaging strategies.

    Recall for class k = TP_k / (TP_k + FN_k). Returns 0.0 when the
    denominator is zero (no true samples in that class).

    Args:
        y_true: 1-D array-like of true labels.
        y_pred: 1-D array-like of predicted labels.
        average: One of "binary", "macro", "micro", "none".
        pos_label: The positive class for average="binary".

    Returns:
        float (for "binary", "macro", "micro") or np.ndarray (for "none").
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    tp, fp, fn = _per_class_prf(y_true, y_pred, labels)

    denom = tp + fn
    per_class = np.where(denom > 0, tp / denom, 0.0)

    if average == "none":
        return per_class
    if average == "macro":
        return float(np.mean(per_class))
    if average == "micro":
        micro_denom = tp.sum() + fn.sum()
        return float(tp.sum() / micro_denom) if micro_denom > 0 else 0.0
    if average == "binary":
        idx = np.where(labels == pos_label)[0]
        if len(idx) == 0:
            return 0.0
        return float(per_class[idx[0]])
    raise ValueError(f"Unknown average: {average!r}")


def f1(y_true, y_pred, average="binary", pos_label=1):
    """F1 score: harmonic mean of precision and recall.

    F1 = 2 * P * R / (P + R). Returns 0.0 when P + R = 0.

    Args:
        y_true: 1-D array-like of true labels.
        y_pred: 1-D array-like of predicted labels.
        average: One of "binary", "macro", "micro", "none".
        pos_label: The positive class for average="binary".

    Returns:
        float (for "binary", "macro", "micro") or np.ndarray (for "none").
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    tp, fp, fn = _per_class_prf(y_true, y_pred, labels)

    p_denom = tp + fp
    r_denom = tp + fn
    per_p = np.where(p_denom > 0, tp / p_denom, 0.0)
    per_r = np.where(r_denom > 0, tp / r_denom, 0.0)
    pr_sum = per_p + per_r
    per_f1 = np.where(pr_sum > 0, 2 * per_p * per_r / pr_sum, 0.0)

    if average == "none":
        return per_f1
    if average == "macro":
        return float(np.mean(per_f1))
    if average == "micro":
        micro_p_denom = tp.sum() + fp.sum()
        micro_r_denom = tp.sum() + fn.sum()
        micro_p = tp.sum() / micro_p_denom if micro_p_denom > 0 else 0.0
        micro_r = tp.sum() / micro_r_denom if micro_r_denom > 0 else 0.0
        denom = micro_p + micro_r
        return float(2 * micro_p * micro_r / denom) if denom > 0 else 0.0
    if average == "binary":
        idx = np.where(labels == pos_label)[0]
        if len(idx) == 0:
            return 0.0
        return float(per_f1[idx[0]])
    raise ValueError(f"Unknown average: {average!r}")


def classification_report(y_true, y_pred):
    """Return a formatted string report of per-class and aggregate metrics.

    Displays per-class precision, recall, F1, and support; plus accuracy
    and macro/micro averages. Mirrors the layout of scikit-learn's report
    for familiarity, but computed entirely from scratch.

    Args:
        y_true: 1-D array-like of true labels.
        y_pred: 1-D array-like of predicted labels.

    Returns:
        str: Multi-line formatted table.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))

    per_p = precision(y_true, y_pred, average="none")
    per_r = recall(y_true, y_pred, average="none")
    per_f = f1(y_true, y_pred, average="none")
    support = np.array([(y_true == lbl).sum() for lbl in labels])

    col_w = 10
    header = f"{'':>12}  {'precision':>{col_w}}  {'recall':>{col_w}}  {'f1-score':>{col_w}}  {'support':>{col_w}}"
    lines = [header, ""]

    for i, lbl in enumerate(labels):
        lines.append(
            f"{str(lbl):>12}  {per_p[i]:>{col_w}.4f}  {per_r[i]:>{col_w}.4f}  {per_f[i]:>{col_w}.4f}  {support[i]:>{col_w}}"
        )

    lines.append("")
    acc = accuracy(y_true, y_pred)
    total = support.sum()
    lines.append(f"{'accuracy':>12}  {'':>{col_w}}  {'':>{col_w}}  {acc:>{col_w}.4f}  {total:>{col_w}}")

    mac_p = precision(y_true, y_pred, average="macro")
    mac_r = recall(y_true, y_pred, average="macro")
    mac_f = f1(y_true, y_pred, average="macro")
    lines.append(f"{'macro avg':>12}  {mac_p:>{col_w}.4f}  {mac_r:>{col_w}.4f}  {mac_f:>{col_w}.4f}  {total:>{col_w}}")

    mic_p = precision(y_true, y_pred, average="micro")
    mic_r = recall(y_true, y_pred, average="micro")
    mic_f = f1(y_true, y_pred, average="micro")
    lines.append(f"{'micro avg':>12}  {mic_p:>{col_w}.4f}  {mic_r:>{col_w}.4f}  {mic_f:>{col_w}.4f}  {total:>{col_w}}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Sanity check on tiny synthetic data
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    # Expected: [[2,1],[1,2]] — 2 TN, 1 FP, 1 FN, 2 TP

    print(f"Accuracy:           {accuracy(y_true, y_pred):.4f}")   # 4/6 ≈ 0.6667
    print(f"Precision (binary): {precision(y_true, y_pred):.4f}")  # 2/3 ≈ 0.6667
    print(f"Recall (binary):    {recall(y_true, y_pred):.4f}")     # 2/3 ≈ 0.6667
    print(f"F1 (binary):        {f1(y_true, y_pred):.4f}")         # 2/3 ≈ 0.6667
    print(f"Precision (macro):  {precision(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 (none):          {f1(y_true, y_pred, average='none')}")
    print()
    print(classification_report(y_true, y_pred))
