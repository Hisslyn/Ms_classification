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
├── hw5/
│   ├── __init__.py            ← Re-exports LDA, QDA, and DecisionTreeClassifier
│   ├── lda_qda.py             ← LDA and QDA (generative, Gaussian class-conditionals)
│   ├── decision_tree.py       ← DecisionTreeClassifier (CART, Gini/entropy, depth/leaf pruning)
│   └── hw5_evaluation.ipynb   ← Evaluation notebook (sweeps, CV, confusion matrices, analysis)
└── report/                    ← Final report (report.md, report.pdf, figures/)
```

---

## 4. Shared Module (`common/`)

### `data_loader.py`

**`get_feature_types() → dict`**  
Returns a dict with three keys: `{"continuous": [...], "categorical": [...], "zero_coded_missing": [...]}` — the only place dataset-specific column names are hardcoded. `zero_coded_missing` lists columns where `0` is physiologically impossible and encodes a missing value in the UCI CSV (`chol`, `trestbps`); callers pass this list to `impute_missing` via its `zero_coded_cols` argument. All transformation functions are otherwise general.

**`load_heart_disease(path) → pd.DataFrame`**  
Reads CSV, binarizes `num` into `target`, drops `id` and `dataset`. No imputation or encoding — those are separate steps.

**`impute_missing(df, continuous_cols, categorical_cols, zero_coded_cols=None) → pd.DataFrame`**  
- `zero_coded_cols` (optional): columns where `0` encodes a missing value rather than a real measurement — those zeros are replaced with `NaN` before imputation.
- Continuous: **median imputation** — robust to outliers, no distributional assumptions.
- Categorical: **mode imputation** — most frequent category. Deterministic tie-breaking (smallest value).
- Returns a new DataFrame; input is never mutated.

**`encode_categoricals(df, categorical_cols) → (df, mapping_dict)`**  
Maps each categorical column to integer codes using sorted unique values → 0, 1, 2, … Sorting guarantees reproducibility regardless of row order. Returns the mapping dict so the same encoding can be applied to held-out data or inverted.

---

### `preprocessing.py`

**`StandardScaler`**  
Formula: `X_scaled = (X - μ) / σ` where μ and σ are computed column-wise on training data only (σ is population std, `ddof=0`). Test data uses training statistics (no leakage).

- `fit(X)` stores `mean_` and `std_`.
- `transform(X)` applies the formula.
- `fit_transform(X)` = fit + transform in one call.
- **Zero-std handling:** when a feature is constant (σ = 0), σ is replaced by 1, so the feature becomes all-zeros after mean subtraction. This avoids division by zero without silently dropping the column. Documented here as a design decision — in practice, constant features carry no information and can be dropped as a preprocessing step.

**`Binner`**  
Discretizes continuous features into integer bin indices for use in HW3 Naive Bayes.

- `strategy="uniform"`: equal-width bins via `np.linspace(min, max, n_bins+1)`.
- `strategy="quantile"`: equal-frequency bins via `np.percentile`. Duplicate edges (from heavy ties) are deduplicated with `np.unique`.
- Bin edges are fit on training data; test values are clipped to the boundary bins — no out-of-range errors.
- Output: integer array with values in `[0, n_bins - 1]` for `strategy="uniform"`. With `strategy="quantile"`, duplicate edges are removed via `np.unique`, so the actual upper bound is `len(unique_edges) - 2`, which may be less than `n_bins - 1` when a feature has heavy ties.

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
| kNN | k=31, distance=manhattan, weights=uniform | 0.8087 ± 0.0212 | **0.7978** | 0.7954 |
| Naive Bayes | alpha=0.01, n_bins=7 | 0.8114 ± 0.0178 | **0.8033** | 0.8001 |

### Key findings

- Manhattan distance outperforms Euclidean consistently across all k values — likely because it is less sensitive to the integer-coded categoricals and outliers in `chol`/`oldpeak`.
- NB accuracy is invariant to alpha (0.01–5.0) on this split: no out-of-vocabulary values exist in the test set, so smoothing affects only probability magnitudes, not the argmax decision.
- n_bins=7 is the NB sweet spot — n_bins=5 (78.69%) loses discriminative resolution; n_bins=10 introduces data sparsity; seven bins gives the best bias-variance tradeoff at this training set size.
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
- Collinear columns without regularization: raises `LinAlgError` with a helpful message if NumPy's solver fails, or solves via least-norm if NumPy resolves the singularity internally — both outcomes are accepted.
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

lr = LogisticRegression(learning_rate=0.1, n_epochs=2000, tol=1e-8, regularization=0.1)
lr.fit(X_train, y_train)

y_pred    = lr.predict(X_test)              # shape (n_test,); default threshold=0.5
y_pred_hr = lr.predict(X_test, threshold=0.3)  # lower threshold → higher recall
y_proba   = lr.predict_proba(X_test)        # shape (n_test, 2), rows sum to 1
logits    = lr.decision_function(X_test)    # shape (n_test,) — pre-sigmoid scores
loss_hist = lr.loss_history_                # list of per-epoch losses (BCE + L2 penalty)
```

**Attributes after `fit`:**
- `classes_` — sorted unique labels from training y
- `coef_` — weight vector of shape (n_features,)
- `intercept_` — scalar bias
- `loss_history_` — list of loss values per epoch
- `n_iter_` — actual epochs run (may be < n_epochs if early stopping triggered)

**`predict(X, threshold=0.5)`** — classifies as `classes_[1]` when `P(y=classes_[1] | x) ≥ threshold`. Lower the threshold to increase recall for the positive class (at the cost of precision); raise it to do the opposite.

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
| Least-Squares | λ=0 (no reg needed) | 0.7870 ± 0.0219 | **0.7978** | 0.7930 |
| Logistic Regression | lr=0.1, λ=0.1, 2000 epochs | 0.7897 ± 0.0258 | **0.8142** | 0.8101 |

### Key findings

- **Regularization for least-squares:** Test accuracy is flat across all λ ∈ {0, 0.01, 0.1, 1.0, 10.0, 100.0} after standardization. X̃ᵀX̃ is already well-conditioned — ridge regularization neither helps nor hurts. Best λ defaults to 0.

- **Convergence speed:** `lr=0.1` reaches the BCE optimum in ~742 epochs. `lr=0.001` had not converged after 2,000 epochs in the learning-rate sweep (final loss 0.503 vs. optimum 0.423). For this dataset, any `lr ≥ 0.1` converges cleanly because the scaled feature matrix is well-conditioned.

- **Logistic regression (81.42%) outperforms least-squares (79.78%) by 1.64 pp** on the test set. Both HW4 linear methods are competitive with kNN (79.78%), and all three are within one CV standard deviation of each other.

- **Score correlation:** LS and LR decision scores correlate at r ≈ 0.995 — they carve out nearly the same boundary despite different loss functions. The loss-function difference matters more when class distributions have heavy tails or severe imbalance.

- **Top LR features by |coef|:** `sex` (+0.40), `cp` (chest pain type, −0.40), `exang` (exercise angina, +0.35), `oldpeak` (ST depression, +0.34). All signs are clinically consistent with known cardiovascular disease risk factors.

---

## 13. HW5 — LDA and QDA

### Generative model framing

Both LDA and QDA are **generative classifiers**: they model the class-conditional density
`P(x | y=c)` as a multivariate Gaussian, then combine it with the class prior `P(y=c)` via
Bayes' rule to obtain the posterior:

```
P(y=c | x)  ∝  P(x | y=c) · P(y=c)
           =  N(x; μ_c, Σ_c) · π_c
```

Classification is by argmax posterior, equivalently argmax **log** posterior (dropping the
x-only normalizing constant — the log determinant of `P(x)` is the same for all classes
and cancels).

### Estimated parameters

For each class c, from the training set:

| Quantity | Formula |
|---------|---------|
| Prior `π_c` | `N_c / N` |
| Mean `μ_c` | `mean of X where y=c` |
| **LDA:** pooled `Σ` | `(1/(N−K)) · Σ_c Σ_{i:y_i=c} (x_i−μ_c)(x_i−μ_c)ᵀ` |
| **QDA:** per-class `Σ_c` | `(1/(N_c−1)) · Σ_{i:y_i=c} (x_i−μ_c)(x_i−μ_c)ᵀ` |

### Discriminant functions

**LDA** (shared Σ → the quadratic terms in x cancel → linear boundary):

```
δ_c(x) = xᵀ Σ⁻¹ μ_c  −  (1/2) μ_cᵀ Σ⁻¹ μ_c  +  log π_c
```

**QDA** (separate Σ_c → quadratic boundary):

```
δ_c(x) = −(1/2) log|Σ_c|  −  (1/2)(x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c)  +  log π_c
```

### Numerical stability

| Concern | Solution |
|---------|---------|
| Singular covariance (correlated features, small N_c) | Ridge: add `ε·I` (`reg_param`, default `1e-6`) to every covariance before inversion |
| Matrix inversion | `np.linalg.solve(Σ, I)` instead of `np.linalg.inv(Σ)` — computes the inverse via LU factorisation, which is more numerically stable than calling `inv` directly |
| Log-determinant overflow | `np.linalg.slogdet` returns `(sign, log|det|)` directly — avoids computing `det` then `log` of a potentially huge number |
| Under-sampled QDA class | `N_c < n_features + 1` → rank-deficient scatter matrix; detected at fit time with a clear `ValueError` suggesting LDA, more data, or PCA |
| Posterior normalization | Log-sum-exp trick: `P(y=c|x) = exp(δ_c − log Σ_c exp(δ_c))`; subtract row-max before exp to prevent overflow |

### When to prefer LDA vs QDA

| Scenario | Prefer |
|----------|--------|
| Small dataset (`N_c` is small relative to `n_features`) | **LDA** — fewer parameters to estimate; pooling stabilizes Σ |
| Class covariances are visually or statistically similar | **LDA** — the shared-Σ assumption holds; LDA is more interpretable |
| Classes clearly have different shapes / orientations | **QDA** — separate Σ_c captures this; LDA will underfit |
| Enough data per class (`N_c ≫ n_features`) | **QDA** — can afford the extra parameters without overfitting |

### Implementation: `hw5/lda_qda.py`

```python
from hw5.lda_qda import LDA, QDA

# LDA
lda = LDA(reg_param=1e-6)
lda.fit(X_train, y_train)
y_pred  = lda.predict(X_test)          # shape (n_test,) — original labels
proba   = lda.predict_proba(X_test)    # shape (n_test, K), rows sum to 1
scores  = lda.decision_function(X_test)# shape (n_test, K), raw log-discriminants

# QDA
qda = QDA(reg_param=1e-6)
qda.fit(X_train, y_train)
y_pred  = qda.predict(X_test)
proba   = qda.predict_proba(X_test)
scores  = qda.decision_function(X_test)
```

**Attributes after `fit`:**

| Attribute | LDA | QDA |
|-----------|-----|-----|
| `classes_` | sorted unique labels | sorted unique labels |
| `priors_` | `(K,)` | `(K,)` |
| `means_` | `(K, n_features)` | `(K, n_features)` |
| `covariance_` | `(n_features, n_features)` pooled Σ (with ridge: `scatter/(N-K) + ε·I`) | — |
| `covariances_` | — | `(K, n_features, n_features)` per-class Σ_c (with ridge: `Xc.T@Xc/(n_c-1) + ε·I`) |

### Design note

`_GaussianDiscriminant` is a private abstract base class shared by LDA and QDA.
The shared scaffold (`fit`, `predict`, `predict_proba`, `decision_function`) lives in the base;
each subclass overrides only `_fit_covariances()` and `_discriminant()` — the two methods that
differ between the two models.

### Sanity test

```bash
python hw5/lda_qda.py
```

Verifies (on synthetic 2D data, 3 classes):
1. LDA on shared-covariance data: training accuracy > 90 %.
2. QDA on distinct-covariance data: training accuracy > 90 %.
3. QDA on 5-sample class with 10 features: `ValueError` raised as expected.
4. LDA on distinct-covariance data: still runs without error (accuracy is printed for reference; no threshold is asserted).
5. Both handle binary (K=2) inputs with accuracy > 90 % and `predict_proba` rows summing to 1.
6. `LDA(reg_param=-1)` and `QDA(reg_param=-0.01)` raise `ValueError`.

---

## 14. HW5 — Decision Tree

### Algorithm overview

Standard CART-style classification tree for continuous features.

**Splitting:** for each internal node, the best (feature, threshold) pair is chosen by maximizing **information gain** relative to a chosen impurity measure.  Candidate thresholds are midpoints between consecutive unique values — semantically equivalent to trying every distinct split but without redundant evaluations.

**Gini impurity:**
```
G(node) = 1 - Σ_c p_c²
```
**Entropy:**
```
H(node) = -Σ_c p_c log₂(p_c)      (with 0·log 0 = 0)
```
**Information gain:**
```
Gain = parent_impurity  -  (n_left/n) · G(left)  -  (n_right/n) · G(right)
```

### Vectorized threshold evaluation

Impurity is evaluated for all candidate thresholds of a single feature in one NumPy pass:

1. Sort samples by feature value once → `O(n log n)`.
2. Build cumulative class-count matrix `(K × n)` via `cumsum` along the sample axis.
3. Use `np.searchsorted` to map each midpoint threshold to a split position (last index ≤ threshold).
4. Read left/right counts by integer indexing into the cumulative matrix → no Python loop over thresholds.

Total cost per feature: `O(n·K + T)` where T = number of unique-value midpoints.  This keeps the depth sweep from the notebook fast.

### Stopping criteria

| Criterion | Meaning |
|-----------|---------|
| `max_depth` | Don't split if current depth ≥ max_depth. |
| `min_samples_split` | Don't split if node has < min_samples_split samples. |
| `min_samples_leaf` | Reject any split creating a child with < min_samples_leaf samples. |
| `min_impurity_decrease` | Reject a split if `gain × (n_node / n_total) < min_impurity_decrease`. The `n_node / n_total` weighting down-weights gains from small (deep) nodes, requiring proportionally more gain to justify a split — a regularization effect matching the sklearn convention. |

**Leaf prediction:** majority class.  Ties broken by the lowest class label (deterministic — `np.argmax` on the sorted `classes_` array returns the first occurrence of the maximum count).

### Implementation: `hw5/decision_tree.py`

```python
from hw5.decision_tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=20)
dt.fit(X_train, y_train)

y_pred  = dt.predict(X_test)          # shape (n_test,)
y_proba = dt.predict_proba(X_test)    # shape (n_test, n_classes), rows sum to 1
dt.print_tree(feature_names=feature_cols, max_lines=50)
```

**Attributes after `fit`:**

| Attribute | Description |
|-----------|-------------|
| `classes_` | Sorted unique class labels |
| `n_classes_` | Number of classes |
| `root_` | Root `_Node` object |
| `n_nodes_` | Total node count (leaves + internal) |
| `max_depth_reached_` | Actual depth of deepest leaf |
| `feature_names_` | Set if input was a DataFrame |

### Sanity test

```bash
python hw5/decision_tree.py
```

Verifies (synthetic iris-like data, 3 classes, 4 features, 150 samples):
1. Unconstrained tree achieves 100% training accuracy.
2. `max_depth=2` reduces node count and training accuracy.
3. `predict_proba` rows sum to 1.
4. `criterion="entropy"` produces > 90% training accuracy.
5. Single-class y → single leaf (no error).
6. n_samples=1 → single leaf.
7. All-identical features → single leaf.
8. `min_samples_leaf=10` → smallest leaf has ≥ 10 samples.
9. `min_impurity_decrease=0.5` → dramatically fewer nodes than unconstrained.
10. `print_tree()` runs and prints readable output.
11. All invalid constructor arguments raise `ValueError`.

---

## 15. HW5 — Evaluation

### Notebook

`hw5/hw5_evaluation.ipynb` — run with `jupyter notebook hw5/hw5_evaluation.ipynb`.

Or execute non-interactively:
```bash
jupyter nbconvert --to notebook --execute --inplace hw5/hw5_evaluation.ipynb
```

### Preprocessing summary

| Pipeline | Scaling | Used by |
|----------|---------|---------|
| `X_train_sc` / `X_test_sc` | `StandardScaler` | LDA, QDA |
| `X_train` / `X_test` | None | Decision Tree |

**Why scale for LDA/QDA:** covariance estimation and its inverse are scale-sensitive — features with large absolute ranges dominate off-diagonal terms, distorting the Gaussian fit.

**Why not scale for Decision Tree:** trees split on raw thresholds; only rank order matters. Unscaled features preserve clinically interpretable thresholds (`cp ≤ 0.5`, `thalch ≤ 141`).

### Best configurations found

| Algorithm | Best Config | 5-fold CV | Test Acc | Macro F1 |
|-----------|------------|-----------|----------|----------|
| LDA | reg_param=1e-6 | 0.7870 ± 0.0219 | **0.7978** | 0.7930 |
| QDA | reg_param=1e-6 | 0.7857 ± 0.0278 | **0.7923** | 0.7877 |
| Decision Tree | gini, min_leaf=20 | 0.7532 ± 0.0251 | **0.7432** | 0.7361 |

### Cross-HW comparison (all seven classifiers)

| Algorithm | Best Config | 5-fold CV | Test Acc | Macro F1 |
|-----------|------------|-----------|----------|----------|
| kNN | k=31, manhattan, uniform | 0.8087 ± 0.0212 | **0.7978** | 0.7954 |
| Naive Bayes | alpha=0.01, n_bins=7 | 0.8114 ± 0.0178 | **0.8033** | 0.8001 |
| Least-Squares | λ=0 | 0.7870 ± 0.0219 | 0.7978 | 0.7930 |
| Logistic Regression | lr=0.1, λ=0.1 | 0.7897 ± 0.0258 | **0.8142** | 0.8101 |
| LDA | reg_param=1e-6 | 0.7870 ± 0.0219 | 0.7978 | 0.7930 |
| QDA | reg_param=1e-6 | 0.7857 ± 0.0278 | 0.7923 | 0.7877 |
| Decision Tree | gini, min_leaf=20 | 0.7532 ± 0.0251 | 0.7432 | 0.7361 |

### Key findings

**Algorithm family ranking:** Logistic Regression (81.42%) > Naive Bayes (80.33%) > kNN = Least-Squares = LDA (all 79.78%) > QDA (79.23%) > Decision Tree (74.32%).  The top five cluster within 2.2 pp; the decision tree sits distinctly lower.

**LDA vs QDA:** LDA marginally outperforms QDA by 0.55 pp (79.78% vs 79.23%).  After correcting the `chol` zero-coded missings, the class-variance asymmetry that previously favoured QDA collapses (chol variance ratio drops from ~3.0 to 0.98), and LDA's shared-covariance assumption is now the better fit.  Both CV bands nearly coincide (78.70% ± 2.19% vs 78.57% ± 2.78%).

**Decision Tree:** the best DT (min_samples_leaf=20, unconstrained depth) achieves 74.32% test accuracy — roughly 5–7 pp behind the leading methods.  The depth sweep confirms classic overfitting: training accuracy reaches 100% (unconstrained tree memorizes the data) while test accuracy peaks at max_depth=3 (73.22%) and declines thereafter.  `min_samples_leaf=20` is more effective regularization than limiting depth on this dataset.

**Cross-algorithm feature agreement:** `cp` (chest pain type) and `exang` (exercise-induced angina) are identified as top features by LDA mean separation (0.80 and 0.87 z-units), logistic regression coefficients (|coef| 0.40 and 0.35), and decision tree root splits.  Three independent algorithms pointing to the same features is strong evidence of genuine clinical importance.

**Practical recommendation:** Logistic Regression for maximum accuracy; LDA when estimation stability matters (fewer parameters than QDA); Decision Tree (min_leaf=20) for interpretability.  All models share the same missing-data limitation: `ca` and `thal` have high missingness rates, and median/mode imputation is a simplification that a production system should address more carefully.

---

## 16. How to Run

### Verify implementations

```bash
python hw3/knn.py                  # kNN sanity test
python hw3/naive_bayes.py          # NB sanity test
python hw4/linear_classifier.py    # Least-squares sanity test
python hw4/logistic_regression.py  # Logistic regression sanity test
python hw5/lda_qda.py              # LDA / QDA sanity test
python hw5/decision_tree.py        # Decision Tree sanity test
```

### Run evaluation notebooks

```bash
jupyter notebook hw3/hw3_evaluation.ipynb
jupyter notebook hw4/hw4_evaluation.ipynb
jupyter notebook hw5/hw5_evaluation.ipynb
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
