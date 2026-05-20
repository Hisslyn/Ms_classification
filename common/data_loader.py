"""Heart Disease dataset loading, imputation, and encoding utilities."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset-specific metadata — the ONLY place column names are hardcoded.
# All transformation functions below are general and work on any DataFrame.
# ---------------------------------------------------------------------------

def get_feature_types():
    """Return continuous and categorical feature lists for the Heart Disease dataset.

    This is the single source of truth for dataset-specific column knowledge.
    All transformation functions (impute_missing, encode_categoricals) are
    general; callers retrieve column lists from here and pass them in.

    Returns:
        dict with keys "continuous", "categorical", and "zero_coded_missing",
        each a list of str. "zero_coded_missing" identifies columns where 0
        encodes a missing value rather than a real measurement.
    """
    return {
        "continuous": ["age", "trestbps", "chol", "thalch", "oldpeak"],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        # Columns where 0 is physiologically impossible and encodes a missing value
        # in the UCI CSV rather than an actual measurement.
        "zero_coded_missing": ["chol", "trestbps"],
    }


# ---------------------------------------------------------------------------
# Core loading function
# ---------------------------------------------------------------------------

def load_heart_disease(path="data/heart_disease_uci.csv"):
    """Load and minimally prepare the UCI Heart Disease dataset.

    Steps performed:
    1. Read the CSV.
    2. Binarize the target: target = 1 if num > 0 else 0 (disease present vs absent).
       Original `num` column is retained for reference.
    3. Drop `id` (meaningless row identifier) and `dataset` (source-site indicator
       that would act as a data-leakage proxy for geographic confounders).

    No imputation or encoding is applied here; those are separate steps so
    callers retain full control.

    Args:
        path: Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame with columns: all original feature columns (except id,
        dataset), the original `num`, and the new binary `target` column.

    Raises:
        FileNotFoundError: If the CSV is not found at `path`.
    """
    df = pd.read_csv(path)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["id", "dataset"])
    return df


# ---------------------------------------------------------------------------
# General-purpose imputation and encoding
# ---------------------------------------------------------------------------

def impute_missing(df, continuous_cols, categorical_cols, zero_coded_cols=None):
    """Impute missing values in a DataFrame without mutating the input.

    Strategy:
    - Columns in zero_coded_cols: replace 0 with NaN first (0 encodes a missing
      value in some datasets rather than a real measurement).
    - Continuous columns: replace NaN with the column median.
    - Categorical columns: replace NaN with the column mode (most frequent value).
      If multiple modes exist, the smallest (first alphabetically/numerically)
      is chosen for determinism.

    This function is intentionally general: pass any column lists; it works on
    any DataFrame, not just the Heart Disease dataset.

    Args:
        df: pd.DataFrame (not mutated).
        continuous_cols: List of column names to median-impute.
        categorical_cols: List of column names to mode-impute.
        zero_coded_cols: Optional list of column names where 0 means missing.

    Returns:
        New pd.DataFrame with missing values filled.
    """
    df = df.copy()
    if zero_coded_cols:
        for col in zero_coded_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
    for col in continuous_cols:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
    return df


def encode_categoricals(df, categorical_cols):
    """Map categorical string/object columns to stable integer codes.

    Mapping rule: sorted unique values of each column → 0, 1, 2, ...
    "Sorted" uses Python's default sort (lexicographic for strings, numeric
    for numbers), ensuring the mapping is deterministic and reproducible
    regardless of row order.

    This function is general and works on any DataFrame.

    Args:
        df: pd.DataFrame (not mutated).
        categorical_cols: List of column names to encode.

    Returns:
        Tuple (encoded_df, mapping_dict) where:
        - encoded_df is a new DataFrame with the listed columns replaced by ints.
        - mapping_dict maps column name → {original_value: int_code, ...},
          so the mapping can be reused on test data or inverted.
    """
    df = df.copy()
    mapping = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        unique_vals = sorted(df[col].dropna().unique())
        val_to_code = {v: i for i, v in enumerate(unique_vals)}
        mapping[col] = val_to_code
        df[col] = df[col].map(val_to_code)
    return df, mapping


if __name__ == "__main__":
    # Sanity check using a tiny synthetic DataFrame (no real CSV needed)
    data = {
        "age": [52.0, np.nan, 45.0, 61.0],
        "chol": [200.0, 230.0, np.nan, 210.0],
        "sex": ["Male", "Female", "Male", np.nan],
        "cp": ["typical angina", "atypical angina", "typical angina", "non-anginal"],
        "num": [0, 2, 0, 3],
    }
    df_raw = pd.DataFrame(data)
    # Simulate target binarization
    df_raw["target"] = (df_raw["num"] > 0).astype(int)

    print("Raw DataFrame:")
    print(df_raw, "\n")

    df_imp = impute_missing(df_raw, continuous_cols=["age", "chol"], categorical_cols=["sex", "cp"])
    print("After imputation:")
    print(df_imp, "\n")

    df_enc, mapping = encode_categoricals(df_imp, categorical_cols=["sex", "cp"])
    print("After encoding:")
    print(df_enc)
    print("\nMapping:")
    for col, m in mapping.items():
        print(f"  {col}: {m}")

    # Verify no mutations of the original
    assert df_raw["age"].isna().any(), "Original should still have NaN"
    print("\nOriginal DataFrame not mutated: OK")
