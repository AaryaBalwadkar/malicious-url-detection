"""Predict labels for new feature rows stored in a CSV file.

The CSV must contain the same 79 lexical feature columns as
data/ISCX-URL2016_All.csv (label column optional, ignored if present).

Run:
    python src/predict_csv.py --input my_urls_features.csv --model XGBoost --output predictions.csv
"""
import argparse
import pathlib
import pandas as pd
import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LABEL_COL = "URL_Type_obf_Type"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV with feature columns")
    ap.add_argument("--model", default="XGBoost")
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    bundle = joblib.load(MODELS / f"{args.model}.joblib")
    clf, scaler, keep = bundle["model"], bundle["scaler"], bundle["features"]
    idx_to_class = {v: k for k, v in bundle["classes"].items()}

    df = pd.read_csv(args.input)
    if LABEL_COL in df.columns:
        df = df.drop(columns=[LABEL_COL])
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing {len(missing)} required columns, e.g. {missing[:5]}")
    X = df[keep].values
    if scaler is not None:
        X = scaler.transform(X)
    pred = clf.predict(X)
    try:
        proba = clf.predict_proba(X)
        proba_max = proba.max(axis=1)
    except Exception:
        proba_max = [float("nan")] * len(pred)

    out = pd.DataFrame({
        "predicted_label": [idx_to_class[int(p)] for p in pred],
        "confidence": proba_max,
    })
    out.to_csv(args.output, index=False)
    print(f"Model: {args.model}")
    print(f"Wrote {len(out)} predictions to {args.output}")
    print(out["predicted_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
