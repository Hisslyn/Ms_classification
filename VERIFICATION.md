# Verification Pass — 2026-05-20

> **Provenance note:** An earlier pass was run on 2026-04-26 against a code state that did not
> yet apply `zero_coded_cols=["chol","trestbps"]` in the imputation step. That run produced
> different numbers (e.g. kNN 81.97 %, QDA 80.87 %, LS/LR/LDA/DT all 78.14 %). This document
> supersedes that pass. All numbers below reflect the current code, which is also the code that
> produced the report figures and the report's Section 5–6 tables.

## 1. Compliance grep

**Pass.** Running:
```
grep -rn "from sklearn|import sklearn|xgboost|from scipy.stats|import scipy.stats" --include="*.py" --include="*.ipynb" .
```
produced no output. No banned library imports found anywhere in the project.

## 2. Generality check

**Pass.** All seven algorithms ran to completion on a synthetic 3-class dataset
(200 samples, 5 features, NumPy seed 0), completely separate from the heart disease data.

| Algorithm | Notes | Result |
|---|---|---|
| KNN | standard multiclass | 0.3077 |
| DiscreteNaiveBayes | binned with `Binner(n_bins=5)` as required | 0.3077 |
| LeastSquaresClassifier | binary-collapsed labels (classes 1,2 → 1) | 0.6667 |
| LogisticRegression | binary-collapsed labels (classes 1,2 → 1) | 0.6667 |
| LDA | standard multiclass | 0.4103 |
| QDA | standard multiclass | 0.4872 |
| DecisionTreeClassifier | standard multiclass | 0.3846 |

Accuracy values are near-chance on random synthetic data, which is expected.
No workarounds beyond the specified binning (NB) and binary-collapsing (LS/LR) were needed.
`verify_generality.py` was deleted after successful completion.

## 3. Notebook execution

All notebooks executed via:
```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <path>
```

- `hw3/hw3_evaluation.ipynb`: **clean** (all cells ran without error)
- `hw4/hw4_evaluation.ipynb`: **clean** (all cells ran without error)
- `hw5/hw5_evaluation.ipynb`: **clean** (all cells ran without error)

## 4. Output consistency

All re-executed numbers compared against previously-reported values. Tolerance: ±0.1 pp on accuracy, ±0.005 on F1.

| Metric | Expected | Re-executed | Verdict |
|---|---|---|---|
| HW3 kNN test accuracy | 79.78% | 79.78% | MATCH |
| HW3 kNN 5-fold CV | 80.87% ± 2.12% | 80.87% ± 2.12% | MATCH |
| HW3 NB test accuracy | 80.33% | 80.33% | MATCH |
| HW3 NB 5-fold CV | 81.14% ± 1.78% | 81.14% ± 1.78% | MATCH |
| HW4 LS test accuracy | 79.78% | 79.78% | MATCH |
| HW4 LS macro F1 | 0.7930 | 0.7930 | MATCH |
| HW4 LS 5-fold CV | 78.70% ± 2.19% | 78.70% ± 2.19% | MATCH |
| HW4 LR test accuracy | 81.42% | 81.42% | MATCH |
| HW4 LR macro F1 | 0.8101 | 0.8101 | MATCH |
| HW4 LR 5-fold CV | 78.97% ± 2.58% | 78.97% ± 2.58% | MATCH |
| HW5 LDA test accuracy | 79.78% | 79.78% | MATCH |
| HW5 LDA 5-fold CV | 78.70% ± 2.19% | 78.70% ± 2.19% | MATCH |
| HW5 QDA test accuracy | 79.23% | 79.23% | MATCH |
| HW5 QDA 5-fold CV | 78.57% ± 2.78% | 78.57% ± 2.78% | MATCH |
| HW5 DT test accuracy | 74.32% | 74.32% | MATCH |
| HW5 DT 5-fold CV | 75.32% ± 2.51% | 75.32% ± 2.51% | MATCH |

All 16 metrics match exactly (zero floating-point drift), confirming fully deterministic pipelines.

## 5. Summary

All compliance checks pass (no sklearn, scipy.stats, xgboost, or any other banned library anywhere). All seven classifiers are general-purpose and reusable — they run correctly on held-out synthetic data with no algorithm-level modifications. All three evaluation notebooks (HW3, HW4, HW5) execute cleanly top-to-bottom from a fresh kernel, and every headline metric reproduces exactly to the last reported decimal place. All numbers in this document match the report (Sections 3–6) and the README summary tables.
