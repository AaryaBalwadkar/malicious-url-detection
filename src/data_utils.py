"""Data utilities for ISCX-URL2016 mirror (36707 rows, 79 lexical features + label)."""
import pandas as pd
import numpy as np

LABEL_COL = "URL_Type_obf_Type"
RANDOM_STATE = 42

# Binary mapping: benign=0, everything else=1
BINARY_MAP = {"benign": 0, "spam": 1, "phishing": 1, "malware": 1, "Defacement": 1,
              "defacement": 1, "Benign": 0}

def load_raw(path="data/ISCX-URL2016_All.csv"):
    df = pd.read_csv(path)
    return df

def clean(df):
    df = df.copy()
    # drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    # replace inf with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    # numeric columns except label
    num_cols = [c for c in df.columns if c != LABEL_COL]
    # impute NaN with median (fit on whole df here; train-only median applied in pipeline via SimpleImputer)
    # keep -1 as informative "absent" code (e.g. no query) per dataset docs
    medians = df[num_cols].median(numeric_only=True)
    df[num_cols] = df[num_cols].fillna(medians)
    # drop rows with missing label
    df = df.dropna(subset=[LABEL_COL])
    # normalise label casing: dataset uses 'Defacement' capitalised, rest lower
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].replace({"Defacement": "defacement", "Benign": "benign"})
    return df, {"duplicates_removed": int(before - after), "rows": len(df)}

def encode_labels(df):
    df = df.copy()
    classes_sorted = sorted(df[LABEL_COL].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    df["label_multi"] = df[LABEL_COL].map(class_to_idx)
    df["label_binary"] = df[LABEL_COL].map(lambda x: BINARY_MAP.get(x, 1))
    return df, class_to_idx

def correlation_prune(X_train, threshold=0.75):
    """Return list of columns to keep (drop one of each highly correlated pair)."""
    corr = X_train.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    keep = [c for c in X_train.columns if c not in to_drop]
    return keep, to_drop
