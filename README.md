# ML Classification Project

**Course:** AUA Machine Learning, Spring 2026  
**Homeworks covered:** HW3 (kNN + Naive Bayes), HW4 (Linear Classifier + Logistic Regression), HW5 (LDA/QDA + Decision Tree)

---

## 1. Project Overview

This repository implements a series of classification algorithms **from scratch** (NumPy + Pandas only — no scikit-learn or scipy.stats for algorithms) applied to the UCI Heart Disease dataset. A shared `common/` module provides data loading, preprocessing, splitting, and metrics that are reused identically across all three homeworks so results are directly comparable.

---

## 2. Dataset

**Source:** UCI Heart Disease (combined four-location version)  
Kaggle: [redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

Place the CSV at: `data/heart_disease_uci.csv` (not tracked by git).

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row identifier — **dropped** (meaningless) |
| `age` | continuous | Age in years |
| `sex` | categorical | Patient sex |
| `dataset` | categorical | Source location — **dropped** (see note below) |
| `cp` | categorical | Chest pain type |
| `trestbps` | continuous | Resting blood pressure (mm Hg) |
| `chol` | continuous | Serum cholesterol (mg/dl) |
| `fbs` | categorical | Fasting blood sugar > 120 mg/dl |
| `restecg` | categorical | Resting ECG results |
| `thalch` | continuous | Maximum heart rate achieved |
| `exang` | categorical | Exercise-induced angina |
| `oldpeak` | continuous | ST depression induced by exercise |
| `slope` | categorical | Slope of peak exercise ST segment |
| `ca` | categorical | Number of major vessels (0–3) coloured by fluoroscopy |
| `thal` | categorical | Thalassemia type |
| `num` | int (0–4) | Original severity label — **retained** for reference |
| `target` | int (0/1) | **Binarized target** (see below) |

### Target Binarization

```
target = 1 if num > 0 else 0
```

`num` encodes severity on a 0–4 scale. We collapse to a binary classification problem: **0 = no disease, 1 = disease present (any severity)**. This is the standard binarization used in the UCI Heart Disease benchmark literature.

### Dropped Columns

- `id`: Sequential row identifier with no predictive value.
- `dataset`: Indicates which of the four collection sites (Cleveland, Hungary, Switzerland, VA Long Beach) a row came from. Including it would let models exploit geographic/institutional confounders rather than clinical features — a form of data leakage for the intended task.

---

## 3. Project Structure

```
ml_classification/
├── README.md                  ← This file
├── data/
│   └── heart_disease_uci.csv  ← Place dataset here manually (not in git)
├── common/
│   ├── __init__.py            ← Re-exports all public symbols
│   ├── data_loader.py         ← CSV loading, binarization, imputation, encoding
│   ├── preprocessing.py       ← StandardScaler and Binner
│   ├── split.py               ← train_test_split and KFold (stratified)
│   └── metrics.py             ← confusion_matrix, accuracy, precision, recall, f1, classification_report
├── hw3/
│   ├── __init__.py            ← Re-exports KNN and DiscreteNaiveBayes
│   ├── knn.py                 ← KNN class (lazy learner, vectorized broadcasting)
│   ├── naive_bayes.py         ← DiscreteNaiveBayes (log-space, Laplace smoothing)
│   └── hw3_evaluation.ipynb   ← Evaluation notebook (sweeps, CV, confusion matrices, analysis)
├── hw4/
│   ├── __init__.py              ← Re-exports LeastSquaresClassifier, LogisticRegression
│   ├── linear_classifier.py     ← LeastSquaresClassifier (normal equation, optional ridge)
│   ├── logistic_regression.py   ← LogisticRegression (BCE + gradient descent, L2 reg)
│   └── hw4_evaluation.ipynb     ← Evaluation notebook
├── hw5/                       ← LDA/QDA + Decision Tree (not yet implemented)
└── report/                    ← Final report deliverable (not yet written)
```

---

## 4. Shared Module (`common/`)

### `data_loader.py`

**`get_feature_types() → dict`**  
Returns `{"continuous": [...], "categorical": [...]}` — the only place dataset-specific column names are hardcoded. All transformation functions are general.

**`load_heart_disease(path) → pd.DataFrame`**  
Reads CSV, binarizes `num` into `target`, drops `id` and `dataset`. No imputation or encoding — those are separate steps.

**`impute_missing(df, continuous_cols, categorical_cols) → pd.DataFrame`**  
- Continuous: **median imputation** — robust to outliers, no distributional assumptions.
- Categorical: **mode imputation** — most frequent category. Deterministic tie-breaking (smallest value).
- Returns a new DataFrame; input is never mutated.

**`encode_categoricals(df, categorical_cols) → (df, mapping_dict)`**  
Maps each categorical column to integer codes using sorted unique values → 0, 1, 2, … Sorting guarantees reproducibility regardless of row order. Returns the mapping dict so the same encoding can be applied to held-out data or inverted.

---

### `preprocessing.py`

**`StandardScaler`**  
Formula: `X_scaled = (X - μ) / σ` where μ and σ are computed column-wise on training data only. Test data uses training statistics (no leakage).

- `fit(X)` stores `mean_` and `std_`.
- `transform(X)` applies the formula.
- `fit_transform(X)` = fit + transform in one call.
- **Zero-std handling:** when a feature is constant (σ = 0), σ is replaced by 1, so the feature becomes all-zeros after mean subtraction. This avoids division by zero without silently dropping the column. Documented here as a design decision — in practice, constant features carry no information and can be dropped as a preprocessing step.

**`Binner`**  
Discretizes continuous features into integer bin indices for use in HW3 Naive Bayes.

- `strategy="uniform"`: equal-width bins via `np.linspace(min, max, n_bins+1)`.
- `strategy="quantile"`: equal-frequency bins via `np.percentile`. Duplicate edges (from heavy ties) are deduplicated with `np.unique`.
- Bin edges are fit on training data; test values are clipped to the boundary bins — no out-of-range errors.
- Output: integer array with values in `[0, n_bins - 1]`.

---

### `split.py`

**`train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)`**  
Stratified by default: for each class, shuffles its indices independently and takes the first `test_size` fraction as test. This ensures class proportions are preserved exactly (up to rounding) in both subsets. Uses `np.random.default_rng(random_state)` for reproducibility.

**`KFold(n_splits=5, shuffle=True, random_state=42, stratify=True)`**  
Stratified K-Fold: for each class, shuffles and splits into `n_splits` chunks, then distributes chunk `k` to fold `k`. All class chunks for fold `k` are concatenated. Yields `(train_idx, val_idx)` tuples via `.split(X, y)`.

---

### `metrics.py`

All metrics take `y_true, y_pred` as numpy arrays.

**`confusion_matrix`**: Rows = true labels, columns = predicted labels. Labels inferred from sorted unique values if not provided.

**`accuracy`**: `correct / total`

**`precision`, `recall`, `f1`** — per-class formulas:
```
precision_k = TP_k / (TP_k + FP_k)
recall_k    = TP_k / (TP_k + FN_k)
f1_k        = 2 * precision_k * recall_k / (precision_k + recall_k)
```

Averaging modes:
| `average` | Meaning |
|-----------|---------|
| `"binary"` | Score for `pos_label` only (default `pos_label=1`) |
| `"macro"` | Unweighted mean of per-class scores |
| `"micro"` | Pool TP/FP/FN globally across all classes |
| `"none"` | Return per-class array |

**Edge case:** any denominator that equals zero returns `0.0` (no exception raised). This happens when a class has no predicted positives (precision), no true positives (recall), or both (F1). This convention is documented here so graders understand the choice.

**`classification_report`**: Pretty-printed table — per-class P/R/F1/support, accuracy, macro avg, micro avg.

---

## 5. Reproducibility

- `random_state=42` is the fixed seed everywhere.
- `np.random.default_rng(42)` is used (the modern NumPy Generator API), not the legacy `np.random.seed`.
- **The same train/test split (80/20, stratified) is reused across HW3, HW4, and HW5** so algorithm comparisons are fair.

---

## 6. Hard Rules (From-Scratch Policy)

1. **Allowed libraries:** NumPy, Pandas, Matplotlib. `numpy.linalg` for matrix ops is fine.
2. **Forbidden:** scikit-learn, scipy.stats for distributions, xgboost, or any library that implements an algorithm/metric/split/scaler being built here.
3. Every algorithm class exposes `fit(X, y)` and `predict(X)`.
4. No hardcoded column names, class counts, or dataset-specific shapes inside algorithm code.
5. Docstrings on every public class and method.

---

## 7. HW3 — kNN

### Algorithm overview

k-Nearest Neighbors is a **lazy learner**: `fit()` only stores the training set. All computation happens at `predict()` time. There is no explicit model or parameter estimation — the training data *is* the model.

For a new point **x**, the algorithm:
1. Computes the distance from **x** to every training point.
2. Selects the k training points with the smallest distances.
3. Takes a vote among those k neighbors to determine the predicted class.

### Distance metrics

| Metric | Formula |
|--------|---------|
| Euclidean | `d(x, xᵢ) = sqrt(Σ (xⱼ - xᵢⱼ)²)` |
| Manhattan | `d(x, xᵢ) = Σ |xⱼ - xᵢⱼ|` |

Both are computed as full `(n_test × n_train)` matrices via NumPy broadcasting — no Python-level loop over test points.

### Vote weighting modes

| Mode | Behavior |
|------|----------|
| `"uniform"` | Each of the k neighbors gets one vote; majority wins. |
| `"distance"` | Each neighbor contributes weight `1/d`; sum of weights per class wins. |

**`weights="distance"` is an extension beyond the brief's minimum requirements.** The brief requires at least two distance metrics; distance-weighted voting is added to give richer behaviour for analysis.

### Tie-breaking convention

When two or more classes have equal vote totals, the class with the **lowest label value** wins. This is deterministic and matches `np.argmax` on a sorted `classes_` array.

### Edge case handling

| Situation | Behaviour |
|-----------|-----------|
| `k > n_train` | `ValueError` raised at `predict()` time with a descriptive message. |
| Zero distance (distance weighting) | The zero-distance neighbor(s) receive infinite implicit weight. Only those neighbors vote; any finite-weight neighbors are ignored. This avoids `NaN` from `1/0` while preserving the correct semantic (exact match dominates). |

### Implementation: `hw3/knn.py`

```python
from hw3.knn import KNN

knn = KNN(k=5, distance="euclidean", weights="uniform")
knn.fit(X_train, y_train)
y_pred  = knn.predict(X_test)       # shape (n_test,)
y_proba = knn.predict_proba(X_test) # shape (n_test, n_classes), rows sum to 1
```

### Sanity test

```bash
python hw3/knn.py
```

Verifies on a synthetic two-Gaussian-blob dataset:
- k=1 achieves 100% training accuracy (memorisation).
- Both distance metrics and both weight modes achieve >85% test accuracy.
- `predict_proba` rows sum to 1.
- `ValueError` is raised when `k > n_train`.

---

## 8. HW3 — Naive Bayes

### Algorithm overview

Discrete Naive Bayes applies Bayes' theorem under the naive (conditional-independence) assumption:

```
P(y | x) ∝ P(y) · ∏_j P(x_j | y)
```

The MAP prediction is the class c that maximises this expression. Computation is done in **log space** to avoid floating-point underflow on long feature vectors:

```
log P(y | x) ∝ log P(y) + Σ_j log P(x_j | y)
```

This class is discrete-only: **the caller must bin continuous features before passing them to `fit()`**. The `Binner` class in `common/preprocessing.py` handles this.

### Laplace (additive) smoothing

Without smoothing, a feature value unseen for a class during training gives `P(x_j=v | y=c) = 0`, making the entire product zero regardless of other features.

Laplace smoothing adds a pseudo-count `alpha` to each (feature, value, class) cell:

```
P(x_j = v | y = c) = (count(x_j=v, y=c) + α) / (N_c + α · |V_j|)
```

where `|V_j|` is the vocabulary size (number of unique values for feature j seen in training) and `N_c` is the count of class-c training samples. `alpha=1.0` is standard Laplace; `alpha → 0` approaches MLE.

### Unseen feature values at predict time

If a value for feature j was never seen during training, the smoothed probability is:

```
P(x_j = v_unseen | y = c) = α / (N_c + α · |V_j|)
```

When `alpha=0`, this is `0.0`, giving `log(0) = -inf` and making that class impossible. A `RuntimeWarning` is emitted in that case so the caller is aware.

### `predict_proba` and the log-sum-exp trick

Log-posteriors are converted to probabilities using:

```
log Z = max_c(score_c) + log Σ_c exp(score_c - max_c(score_c))
P(y=c | x) = exp(score_c - log Z)
```

Subtracting the row maximum before exponentiating prevents overflow without changing the normalised result.

### Why quantile binning for continuous features

Quantile binning (`Binner(strategy="quantile")`) divides the **rank order** of training points into equal-frequency bins, so each bin contains approximately `N_train / n_bins` points. Uniform binning divides the **value range** into equal-width intervals, which leaves upper bins nearly empty for skewed features like `chol` (right-tailed) and `oldpeak` (heavily concentrated near 0). Empty or near-empty bins produce noisy conditional probability estimates — exactly what Laplace smoothing is trying to fix. Quantile binning avoids this problem at the source.

### Implementation: `hw3/naive_bayes.py`

```python
from common.preprocessing import Binner
from hw3.naive_bayes import DiscreteNaiveBayes

binner = Binner(n_bins=5, strategy="quantile")
X_train_binned = binner.fit_transform(X_train_continuous)
X_test_binned  = binner.transform(X_test_continuous)

nb = DiscreteNaiveBayes(alpha=1.0)
nb.fit(X_train_binned, y_train)
y_pred  = nb.predict(X_test_binned)        # shape (n_test,)
y_proba = nb.predict_proba(X_test_binned)  # shape (n_test, n_classes), rows sum to 1
```

---

## 9. HW3 — Evaluation

### Notebook

`hw3/hw3_evaluation.ipynb` — run with `jupyter notebook hw3/hw3_evaluation.ipynb`.

### Preprocessing pipelines

Two separate pipelines are used because each algorithm has different requirements:

| Step | kNN pipeline | Naive Bayes pipeline |
|------|-------------|---------------------|
| Continuous features | `StandardScaler` (z-score normalization) | `Binner(n_bins=5, strategy="quantile")` |
| Categorical features | Integer codes as-is | Integer codes as-is |
| Why | kNN distance is dominated by large-range features without scaling | NB requires discrete values; binning converts continuous to integer bins |

Both pipelines are fit on training data only; test data is transformed using training statistics.

### Hyperparameter sweeps

**kNN:** `k` ∈ {1, 3, 5, 7, 11, 15, 21, 31} × `distance` ∈ {euclidean, manhattan}

**Naive Bayes alpha:** `alpha` ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}

**Naive Bayes n_bins:** `n_bins` ∈ {3, 5, 7, 10} (alpha=1.0 fixed)

### Best configurations (actual results)

| Algorithm | Best config | CV acc (5-fold) | Test acc | Macro F1 |
|-----------|------------|----------------|----------|----------|
| kNN | k=21, distance=manhattan | 0.8291 ± 0.0193 | **0.8197** | 0.8175 |
| Naive Bayes | alpha=0.01, n_bins=5 | 0.8223 ± 0.0227 | **0.8033** | 0.7995 |

### Key findings

- Manhattan distance outperforms Euclidean consistently across all k values — likely because it is less sensitive to the integer-coded categoricals and outliers in `chol`/`oldpeak`.
- NB accuracy is invariant to alpha (0.01–5.0) on this split: no out-of-vocabulary values exist in the test set, so smoothing affects only probability magnitudes, not the argmax decision.
- n_bins=5 is the NB sweet spot — coarser bins lose resolution; finer bins introduce data sparsity.
- Both models achieve similar accuracy (~80–82%), confirming that the naive independence assumption is surprisingly robust for classification on this dataset despite clearly correlated features (`age`, `chol`, `trestbps`, `thalch`).

---

## 10. HW4 — Least-Squares Linear Classifier

### Algorithm overview

Least-squares linear classification fits a hyperplane by **minimising the squared error between the linear output and ±1 encoded labels**, then classifies a new point by the **sign** of that linear output.  The resulting decision boundary is linear in the feature space — identical in geometry to logistic regression's boundary — but obtained by solving a regression problem rather than maximising a likelihood.

Binary labels y ∈ {0, 1} are encoded internally as t ∈ {-1, +1}:

```
t = 2y - 1   (i.e.  0 → -1,  1 → +1)
```

Callers always pass 0/1 labels; the class converts internally and maps predictions back to the original label space.

### Normal equation

The feature matrix is **augmented** with a column of ones to absorb the bias term:

```
X̃ = [X | 1]   ∈ ℝ^{n × (d+1)}
```

The least-squares solution minimises `‖X̃w - t‖²` and satisfies the normal equation:

```
X̃ᵀX̃ w = X̃ᵀ t
```

**With ridge regularisation (λ > 0):**

```
(X̃ᵀX̃ + λI̊) w = X̃ᵀ t
```

where I̊ is the identity matrix with its **last diagonal entry set to 0**, so the bias term is excluded from regularisation.  This is the standard convention: penalising the bias would shift the decision boundary based on the scale of features, which is undesirable.

### `np.linalg.solve` vs explicit inversion

The normal equation is solved via:

```python
w = np.linalg.solve(X̃ᵀX̃ + λI̊,  X̃ᵀ t)
```

rather than `w = inv(X̃ᵀX̃) @ X̃ᵀ t`.  `solve` uses LU factorisation and avoids forming the inverse, which is both faster and more numerically stable for ill-conditioned systems.

**From-scratch justification:** `np.linalg.solve` is a general linear-algebra primitive (analogous to `np.sqrt`) — it has no knowledge of classification.  Using it is equivalent to using `np.mean` to compute a class prior.  No classifier, metric, optimiser, or regression library is used.

### Decision rule

```
ŷ = classes_[1]  if  xᵀ coef_ + intercept_ ≥ 0
ŷ = classes_[0]  otherwise
```

### Why no `predict_proba`

Least-squares scores are raw real numbers with no probabilistic interpretation.  Applying a sigmoid post-hoc (i.e. `σ(s)`) would produce values between 0 and 1, but they would **not** be calibrated probabilities: the squared-loss objective gives no guarantee that the output is a posterior.  Logistic regression (HW4 Part 2) addresses this exactly — it maximises the **log-likelihood** of the Bernoulli model, which directly produces calibrated posteriors via the sigmoid link.

### Sensitivity to outliers

Least-squares penalises every prediction error by its square.  A point that lies far from the decision boundary but on the wrong side contributes a large quadratic penalty and can pull the hyperplane significantly.  This contrasts with logistic regression's cross-entropy loss, where the gradient saturates for confident correct predictions — making it much less sensitive to well-separated outliers.  This difference is a key motivation for preferring logistic regression in practice, and will be revisited in the HW4 analysis notebook.

### Implementation: `hw4/linear_classifier.py`

```python
from hw4.linear_classifier import LeastSquaresClassifier

clf = LeastSquaresClassifier(regularization=0.0)   # or e.g. 1e-4 for ridge
clf.fit(X_train, y_train)

y_pred   = clf.predict(X_test)           # shape (n_test,) — original label space
scores   = clf.decision_function(X_test) # shape (n_test,) — continuous raw scores
```

**Attributes after `fit`:**
- `classes_` — sorted unique labels from training y
- `coef_` — weight vector of shape (n_features,)
- `intercept_` — scalar bias term
- `weights_` — full augmented vector `[coef_ | intercept_]`

### Sanity test

```bash
python hw4/linear_classifier.py
```

Verifies on a synthetic two-Gaussian-blob dataset (no real data):
- Training accuracy > 95% on well-separated blobs.
- `decision_function` returns continuous values (not just 0/1).
- `ValueError` raised for 3-class y.
- `ValueError` raised for `regularization < 0`.
- Collinear columns without regularization raise `LinAlgError` with a helpful message.
- Collinear columns with `regularization=1e-4` solve cleanly.

---

## 11. HW4 — Logistic Regression

### Algorithm overview

Binary logistic regression models the posterior directly:

```
P(y=1 | x) = σ(wᵀx + b)
```

where σ is the sigmoid function. Parameters are estimated by maximising the log-likelihood, equivalent to minimising **binary cross-entropy (BCE)**:

```
L = -(1/N) Σ_i [y_i log(p_i + ε) + (1-y_i) log(1-p_i + ε)]
```

A tiny `ε = 1e-15` is added inside both log arguments to prevent `log(0)` when predictions saturate in finite precision. This is a numerical guard only — it has no effect unless the sigmoid output is exactly 0 or 1.

### Gradient derivation

For the sigmoid output, the BCE gradient has the remarkably clean form:

```
∂L/∂w_j = (1/N) Σ_i (p_i - y_i) x_{ij}   →   ∇w L = (1/N) Xᵀ(p - y)
∂L/∂b   = (1/N) Σ_i (p_i - y_i)           →   ∇b L = mean(p - y)
```

The derivation: `∂L/∂z_i = p_i - y_i` (the sigmoid and cross-entropy cancel to give this simple residual). Chain rule to weights: `∂z_i/∂w_j = x_{ij}`. Vectorised over all samples: `Xᵀ r` where `r = p - y`.

### L2 regularization

Adding `λ‖w‖²/2` to the loss contributes `λw` to `∇w L`. The bias is NOT regularised — consistent with the convention in the least-squares classifier (penalising the bias shifts the intercept based on feature scale, which is undesirable).

### Numerically stable sigmoid

```python
# z >= 0: 1 / (1 + exp(-z))      — safe because exp(-z) → 0 for large z
# z <  0: exp(z) / (1 + exp(z))  — safe because exp(z)  → 0 for z → -∞
```

The naive `1/(1+exp(-z))` overflows to `inf` for `z < -709` on float64. The two-branch form avoids this.

### Early stopping

Training halts when `|L(t-1) - L(t)| < tol`. This detects effective convergence without requiring a fixed epoch budget. Set `tol=0` to run exactly `n_epochs` iterations.

### Implementation: `hw4/logistic_regression.py`

```python
from hw4.logistic_regression import LogisticRegression

lr = LogisticRegression(learning_rate=0.1, n_epochs=2000, tol=1e-7, regularization=0.1)
lr.fit(X_train, y_train)

y_pred    = lr.predict(X_test)              # shape (n_test,)
y_proba   = lr.predict_proba(X_test)        # shape (n_test, 2), rows sum to 1
logits    = lr.decision_function(X_test)    # shape (n_test,) — pre-sigmoid scores
loss_hist = lr.loss_history_                # list of per-epoch BCE losses
```

**Attributes after `fit`:**
- `classes_` — sorted unique labels from training y
- `coef_` — weight vector of shape (n_features,)
- `intercept_` — scalar bias
- `loss_history_` — list of loss values per epoch
- `n_iter_` — actual epochs run (may be < n_epochs if early stopping triggered)

### Sanity test

```bash
python hw4/logistic_regression.py
```

Verifies:
- Loss is monotonically non-increasing.
- Training accuracy > 95% on well-separated blobs.
- `predict_proba` rows sum to 1.
- Extreme points give probabilities > 0.99 and < 0.01.
- Sigmoid stability: `σ(±1000)` does not produce NaN.
- Non-finite loss raises `ValueError` with a clear message.

---

## 12. HW4 — Evaluation

### Notebook

`hw4/hw4_evaluation.ipynb` — run with:

```bash
jupyter notebook hw4/hw4_evaluation.ipynb
```

Or non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace hw4/hw4_evaluation.ipynb
```

### Preprocessing pipeline

Both HW4 algorithms use the same pipeline (unlike HW3 where kNN and NB needed different transformations):

| Step | HW4 pipeline |
|------|-------------|
| Load + binarize | `load_heart_disease` |
| Impute missing | median (continuous), mode (categorical) |
| Encode categoricals | sorted integer codes |
| Split | stratified 80/20, `random_state=42` — identical to HW3 |
| Scale | `StandardScaler` on ALL features (fit on train, applied to test) |

Scaling is essential for both algorithms: the conditioning of X̃ᵀX̃ (least-squares) and the convergence rate of gradient descent (logistic regression) both depend on feature scale.

### Best configurations

| Algorithm | Config | CV acc (5-fold) | Test acc | Macro F1 |
|-----------|--------|-----------------|----------|----------|
| Least-Squares | λ=0 (no reg needed) | 0.8047 ± 0.0282 | **0.7814** | 0.7795 |
| Logistic Regression | lr=0.1, λ=0.1, 2000 epochs | 0.8128 ± 0.0295 | **0.7814** | 0.7779 |

### Key findings

- **Regularization for least-squares:** Test accuracy is flat across all λ ∈ {0, 0.01, 0.1, 1.0, 10.0, 100.0} after standardization. X̃ᵀX̃ is already well-conditioned — ridge regularization neither helps nor hurts. Best λ defaults to 0.

- **Convergence speed:** `lr=0.1` reaches the BCE optimum in ~742 epochs. `lr=0.001` is still 60 epochs away from convergence at epoch 10,000. For this dataset, any `lr ≥ 0.1` converges cleanly because the scaled feature matrix is well-conditioned.

- **Both HW4 methods tie at 78.14% test accuracy**, roughly 3.8 pp below the best HW3 result (kNN 81.97%). The gap suggests the optimal boundary has slight non-linearity that kNN's locally adaptive decision captures.

- **Score correlation:** LS and LR decision scores correlate at r ≈ 0.95 — they carve out nearly the same boundary despite different loss functions. The loss-function difference matters more when class distributions have heavy tails or severe imbalance.

- **Top LR features by |coef|:** `cp` (chest pain type, −0.40), `exang` (exercise angina, +0.36), `sex` (+0.36), `oldpeak` (ST depression, +0.34). All signs are clinically consistent with known cardiovascular disease risk factors.

---

## 13. How to Run

### Verify implementations

```bash
python hw3/knn.py                # kNN sanity test
python hw3/naive_bayes.py        # NB sanity test
python hw4/linear_classifier.py  # Least-squares sanity test
python hw4/logistic_regression.py  # Logistic regression sanity test
```

### Run evaluation notebooks

```bash
jupyter notebook hw3/hw3_evaluation.ipynb
jupyter notebook hw4/hw4_evaluation.ipynb
```

Or execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace hw3/hw3_evaluation.ipynb
```

### General setup

```bash
# No installation beyond standard scientific Python stack
pip install numpy pandas matplotlib
```

### Verify common/ module

```bash
python common/metrics.py
python common/split.py
python common/preprocessing.py
python common/data_loader.py
```

Each script runs a self-contained sanity check on synthetic data.
