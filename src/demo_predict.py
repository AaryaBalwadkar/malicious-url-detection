"""Demo: load best model and predict a test example with explanation.
Run: .venv/Scripts/python.exe src/demo_predict.py [--model XGBoost] [--row 0]
"""
import argparse, pathlib
import pandas as pd
import joblib

ROOT = pathlib.Path(".")
MODELS = ROOT / "models"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XGBoost")
    ap.add_argument("--row", type=int, default=0,
                    help="test row index to demo (uses held-out test split row)")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split
    import sys
    sys.path.insert(0, "src")
    from data_utils import load_raw, clean, encode_labels, LABEL_COL
    df = encode_labels(clean(load_raw())[0])[0]
    feature_cols = [c for c in df.columns if c not in (LABEL_COL, "label_multi", "label_binary")]
    X = df[feature_cols]; y = df["label_multi"].values
    _, X_test_full, _, y_test = train_test_split(X, y, test_size=0.15,
        stratify=y, random_state=42)
    bundle = joblib.load(MODELS / f"{args.model}.joblib")
    clf, scaler, keep = bundle["model"], bundle["scaler"], bundle["features"]
    idx_to_class = {v: k for k, v in bundle["classes"].items()}
    row = X_test_full.iloc[args.row][keep]
    Xt = scaler.transform([row.values]) if scaler is not None else [row.values]
    pred = int(clf.predict(Xt)[0])
    proba = clf.predict_proba(Xt)[0]
    true = idx_to_class[int(y_test[args.row])]
    print(f"Model: {args.model}")
    print(f"True label: {true}")
    print(f"Predicted: {idx_to_class[pred]}")
    print("Probabilities:")
    for i, p in enumerate(proba):
        print(f"  {idx_to_class[i]:12s}: {p:.4f}")
    print("\nTop input feature values for this row:")
    print(row.sort_values(ascending=False).head(10).to_string())

if __name__ == "__main__":
    main()
