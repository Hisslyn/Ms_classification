---
title: "Classification Algorithms from Scratch — A Comparative Study on UCI Heart Disease Data"
author: "[Author name placeholder — fill in manually before submission]"
date: "April 2026"
---

# 1. Introduction

## 1.1 Project overview

This report covers a three-homework series completed for the Machine Learning course at the American University of Armenia (AUA), Spring 2026. Over the course of HW3, HW4, and HW5, seven classification algorithms were implemented entirely from scratch: k-Nearest Neighbors (kNN), Discrete Naive Bayes, Least-Squares linear classifier, Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and a CART-style Decision Tree. The implementation constraint was strict — only NumPy, Pandas, and Matplotlib were permitted. No scikit-learn, scipy.stats, xgboost, or any library that implements an algorithm, metric, splitter, or scaler was used.

To make cross-homework comparisons meaningful, a shared `common/` module was built before any algorithm work began. It provides data loading, missing-value imputation, categorical encoding, stratified splitting, and evaluation metrics — utilities used identically across all three homeworks. The same dataset, the same target formulation, and the same train/test split are reused throughout, so any difference in test accuracy reflects algorithmic differences rather than preprocessing or sampling differences.

The headline finding is that all seven algorithms cluster within a narrow accuracy band on this dataset (roughly 76–82% test accuracy on the held-out test set), with kNN and QDA slightly leading and the linear models trailing by a margin within one cross-validation standard deviation. The decision tree, while not the most accurate, provides the most interpretable model.

## 1.2 Dataset

The dataset used is the UCI Heart Disease dataset, specifically the combined four-location version published by redwankarimsony on Kaggle. It aggregates records from four collection sites: Cleveland, Hungary, Switzerland, and Long Beach VA, yielding 920 patients in total.

The dataset contains 14 clinical features plus a `dataset` source indicator. Features span a range of clinical measurements:

| Column | Type | Description |
|--------|------|-------------|
| `age` | continuous | Age in years |
| `sex` | categorical | Patient sex |
| `cp` | categorical | Chest pain type |
| `trestbps` | continuous | Resting blood pressure (mm Hg) |
| `chol` | continuous | Serum cholesterol (mg/dl) |
| `fbs` | categorical | Fasting blood sugar > 120 mg/dl |
| `restecg` | categorical | Resting ECG results |
| `thalch` | continuous | Maximum heart rate achieved |
| `exang` | categorical | Exercise-induced angina |
| `oldpeak` | continuous | ST depression induced by exercise |
| `slope` | categorical | Slope of peak exercise ST segment |
| `ca` | categorical | Number of major vessels coloured by fluoroscopy |
| `thal` | categorical | Thalassemia type |

After binarization of the target (described in Section 2.1), the class distribution is 509 disease cases (55.3%) and 411 healthy cases (44.7%) — reasonably balanced, with no need for oversampling or class-weight adjustment.

A prominent data quality issue is heavy missingness in several columns: `ca` is missing in 66.4% of rows, `thal` in 52.8%, and `slope` in 33.6%, with smaller amounts across five other features. The imputation strategy and its implications are discussed in Section 2.2.

The `id` and `dataset` columns are dropped before any modeling. `id` is a meaningless row identifier. `dataset` encodes the collection site; including it would allow models to exploit geographic and institutional confounders rather than clinical features — a form of data leakage relative to the intended clinical prediction task.

**Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

## 1.3 Methodology and shared infrastructure

The repository is organized into four top-level directories: `common/` (shared utilities), `hw3/`, `hw4/`, and `hw5/` (per-homework algorithm implementations and evaluation notebooks), and `report/` (this document and supporting figures).

The `common/` module is the backbone of the project. `data_loader.py` handles CSV loading, target binarization, missing-value imputation, and categorical encoding — and is the single location where dataset-specific column names are hardcoded, keeping all algorithm code fully general. `preprocessing.py` provides a `StandardScaler` (fit-on-train, transform-on-test, with zero-variance protection) and a `Binner` (uniform or quantile discretization for Naive Bayes). `split.py` provides a stratified train/test split and stratified K-Fold cross-validation, both implemented from scratch using `np.random.default_rng`. `metrics.py` provides confusion matrix, accuracy, precision, recall, F1, and a formatted classification report, supporting binary, macro, micro, and per-class averaging modes.

All seven algorithm classes expose a consistent `fit(X, y)` / `predict(X)` interface. Where probabilistic output is meaningful, `predict_proba(X)` is also implemented. This uniformity allows the evaluation notebooks to swap classifiers with a single line change.

Reproducibility is enforced by `random_state=42` throughout, using the modern `np.random.default_rng(42)` API rather than the legacy `np.random.seed`. A dedicated verification pass (documented in `VERIFICATION.md`) confirmed bit-exact reproduction of all 16 headline metrics across two independent notebook runs, and confirmed that all seven algorithms run without modification on a synthetic non-heart-disease dataset — satisfying the course brief's "general and reusable" requirement.

The evaluation strategy is consistent across homeworks: an 80/20 stratified train/test split provides the headline test metrics; 5-fold stratified cross-validation on the training set provides variance estimates; and hyperparameter sweeps (grid search on CV accuracy) identify best configurations per algorithm. Because the same split object is used in HW3, HW4, and HW5, the 736-sample training set and 184-sample test set are identical across all experiments.

---

# 2. Preprocessing

## 2.1 Target binarization

The original target column `num` encodes disease severity on a 0–4 scale, where 0 means no disease and 1–4 represent increasing severity grades. The course brief permits either binary or multi-class framing. The decision here is to binarize:

$$\text{target} = \begin{cases} 1 & \text{if } \texttt{num} > 0 \\ 0 & \text{otherwise} \end{cases}$$

The primary reason is consistency. HW4 logistic regression is binary-only by specification, so binary is the only target formulation that works unchanged across all three homeworks. Using the same target throughout ensures that accuracy numbers from HW3, HW4, and HW5 are directly comparable — differences in performance reflect the algorithms, not the task.

A secondary benefit is interpretability: the binarized task ("disease present or absent") corresponds to a clinically meaningful screening question. The resulting class distribution — 55.3% disease, 44.7% healthy — is close to balanced, so standard accuracy is a fair primary metric and no class-imbalance adjustments are needed.

## 2.2 Missing value imputation

The dataset has substantial missingness across ten features:

| Feature | Type | Missing | Strategy |
|---------|------|---------|----------|
| `ca` | categorical | 66.4% | Mode |
| `thal` | categorical | 52.8% | Mode |
| `slope` | categorical | 33.6% | Mode |
| `fbs` | categorical | 9.8% | Mode |
| `oldpeak` | continuous | 6.7% | Median |
| `trestbps` | continuous | 6.4% | Median |
| `thalch` | continuous | 6.0% | Median |
| `exang` | categorical | 6.0% | Mode |
| `chol` | continuous | 3.3% | Median |
| `restecg` | categorical | 0.2% | Mode |

Three strategies were considered. Dropping high-missingness columns (`ca`, `thal`, `slope`) would remove clinically informative features — even at 66.4% missingness, the observed values in `ca` plausibly carry signal that would be lost by dropping the column. Dropping rows with any missing value would shrink the dataset to roughly 300 samples drawn predominantly from Cleveland, losing the multi-site diversity that gives this dataset its generalization value. Imputation was therefore chosen as the least-damaging option that preserves both sample size and feature set. Note that `ca` is technically an ordinal count variable (number of major vessels, 0–3); it is treated as categorical for imputation purposes as a simplification, which downstream algorithms tolerate.

For continuous features, median imputation is used rather than mean because several features (`chol`, `oldpeak`) have right-skewed distributions; the median is more robust to extreme values and does not require distributional assumptions. For categorical features, mode imputation assigns the most frequent category. Ties are broken deterministically by selecting the smallest value, ensuring reproducibility regardless of row order.

A caveat applies to the heavily imputed categorical columns. For `slope` (33.6% imputed to `flat`), the dominant imputed value shows near-equal conditional class probabilities, attenuating the feature's discriminative signal. For `thal` (52.8% imputed to `normal`), the `reversable defect` value retains its strong discriminative ratio in the observed rows, but the overall feature signal is diluted by mass-imputed `normal` values. This limitation is revisited in the algorithm-specific analyses in Sections 3 and 5.

## 2.3 Categorical encoding

Eight features are categorical: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, and `thal`. Of these, `fbs` and `exang` are already binary numeric in the raw data and require no encoding. The remaining six are encoded using a stable integer scheme: each category is mapped to an integer based on alphabetical order of its unique values (0, 1, 2, …). The mapping is computed at fit time on the training set, stored in a dictionary, and applied identically to test data. Alphabetical sorting guarantees the same integer assignment regardless of the order in which rows appear, making the encoding fully reproducible.

One-hot encoding was considered but not used, for two reasons. First, it increases dimensionality from 13 to roughly 25 features, which complicates covariance estimation in LDA/QDA and makes the decision tree harder to interpret. Second, the course brief emphasizes minimal, understandable implementations — integer codes satisfy that goal without introducing a preprocessing layer that itself requires explanation.

The known limitation of integer encoding is that it imposes an implicit ordinal interpretation on nominal categories. For kNN, Manhattan distance treats integer-coded levels as equidistant ordinal steps, which is technically incorrect for unordered categories like `thal` (normal, fixed defect, reversable defect). This limitation is acknowledged in Section 3 (kNN analysis). For Naive Bayes, each integer is treated as an independent discrete symbol, so ordinality does not enter the computation. For the linear models and LDA, the coefficients absorb any misaligned scaling. For the decision tree, only rank order is used at split points, so the encoding's ordinal assumption is harmless.

## 2.4 Train/test split

The dataset of 920 rows is split 80/20 into 736 training samples and 184 test samples using a stratified split with `random_state=42`. Stratification by the binary target preserves class proportions in both subsets: approximately 55.3% disease and 44.7% healthy in each partition, matching the overall dataset balance.

The same split is reused across HW3, HW4, and HW5. This is a deliberate methodological choice: because the training set and test set are identical in all three homeworks, any difference in test accuracy reflects algorithmic differences on this specific split rather than differences in sampling. The 5-fold cross-validation results reported in Sections 3–5 provide variance estimates that further calibrate these comparisons. The split is implemented from scratch in `common/split.py` and produces bit-identical indices on every run.

## 2.5 Algorithm-specific preprocessing

A single global preprocessing pipeline (impute → encode → split) is shared by all algorithms. What differs is the transformation applied to the split features before fitting:

- **kNN** uses `StandardScaler` on all features (continuous and integer-encoded categorical), fit on the training set only. Standardization is essential because kNN distance metrics are dominated by features with large absolute ranges: `chol` spans up to ~600 mg/dl while binary features like `sex` span {0, 1}. Without scaling, Euclidean and Manhattan distances are almost entirely determined by `chol`, `trestbps`, and `thalch`, effectively ignoring the categorical features.

- **Naive Bayes** requires discrete inputs. Continuous features are discretized using `Binner(n_bins=5, strategy="quantile")` fit on training data; categorical features are passed as integer codes unchanged. Quantile binning is chosen over uniform binning because several continuous features are right-skewed — uniform bins would leave the upper bins sparsely populated, producing noisy conditional probability estimates. Integer-coded categoricals are already discrete and do not need binning.

- **Least-Squares and Logistic Regression** both use `StandardScaler` on all features. For least-squares, scaling improves the condition number of $\tilde{X}^\top \tilde{X}$, though with this dataset the normal equation is well-conditioned even without regularization. For logistic regression, scaling is critical for gradient descent convergence: the experiments show that `lr=0.1` converges in approximately 742 epochs on scaled features, whereas `lr=0.001` on the same scaled data had not converged by 10,000 epochs — a direct consequence of the condition number of the feature matrix.

- **LDA and QDA** use `StandardScaler` on all features. Both methods estimate class-conditional covariance matrices; without scaling, features with large variance (e.g., `chol`) dominate off-diagonal covariance entries, distorting the Gaussian fit and making the estimated covariances numerically ill-conditioned.

- **Decision Tree** uses no scaling. Decision trees split on raw feature thresholds and are invariant to monotone transformations of feature values. Preserving the original scale also keeps the printed tree interpretable: a split reads as `thalch ≤ 141` rather than `thalch_scaled ≤ −0.34`, which is more meaningful in a clinical context.

The preprocessing differences are not arbitrary — each algorithm's mathematical assumptions dictate what its inputs should look like. A single universal pipeline would either degrade Naive Bayes (which needs discrete inputs) or sacrifice decision tree interpretability.
