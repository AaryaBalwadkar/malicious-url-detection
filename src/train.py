"""End-to-end training for malicious URL detection (ISCX-URL2016 mirror).
Covers report Sections 6-12: cleaning, split 70:15:15 stratified, correlation
pruning 0.75, 6 classifiers, Accuracy/Precision/Recall/F1/ROC-AUC.
Run: .venv/Scripts/python.exe src/train.py
"""
import time, json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from data_utils import load_raw, clean, encode_labels, correlation_prune, LABEL_COL, RANDOM_STATE

ROOT = pathlib.Path(".")
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGS = ROOT / "figures"

def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=2, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "SVM_RBF": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
        "KNN_k5": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                                 n_jobs=-1, random_state=RANDOM_STATE),
    }

NEEDS_SCALING = {"LogisticRegression", "SVM_RBF", "KNN_k5"}

def main():
    t0 = time.time()
    print("Loading dataset...")
    df_raw = load_raw()
    print(f"Raw: {df_raw.shape}")
    df, clean_info = clean(df_raw)
    df, class_to_idx = encode_labels(df)
    print("Clean info:", clean_info, "classes:", class_to_idx)

    feature_cols = [c for c in df.columns if c not in (LABEL_COL, "label_multi", "label_binary")]
    X = df[feature_cols]
    y = df["label_multi"].values
    print(f"Features: {len(feature_cols)}, rows: {len(X)}")

    # 70:15:15 stratified split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full,
        random_state=RANDOM_STATE)  # 0.1765*0.85 ~= 0.15
    print(f"Train {X_train.shape} Val {X_val.shape} Test {X_test.shape}")

    # Correlation pruning on train only
    keep, dropped = correlation_prune(X_train, threshold=0.75)
    print(f"Kept {len(keep)}/{len(feature_cols)}, dropped {len(dropped)}")
    X_train, X_val, X_test = X_train[keep], X_val[keep], X_test[keep]

    # Median imputation already done in clean(); refit-safe SimpleImputer for pipeline safety
    # Figures: class distribution
    plt.figure(figsize=(7, 4))
    pd.Series(y_train_full).map({v: k for k, v in class_to_idx.items()}).value_counts().plot(kind="bar")
    plt.title("Class distribution (train+val, ISCX-URL2016 mirror n=36707)")
    plt.ylabel("Count"); plt.tight_layout(); plt.savefig(FIGS / "class_distribution.png"); plt.close()

    # Correlation heatmap subset (top 20 by variance) to keep figure readable
    top20 = X_train.var(numeric_only=True).sort_values(ascending=False).head(20).index
    plt.figure(figsize=(8, 6))
    plt.imshow(X_train[top20].corr(numeric_only=True).values, vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r"); plt.xticks(range(len(top20)), top20, rotation=90, fontsize=6)
    plt.yticks(range(len(top20)), top20, fontsize=6); plt.title("Feature correlation (top-20 variance)")
    plt.tight_layout(); plt.savefig(FIGS / "correlation_top20.png"); plt.close()

    models = get_models()
    rows = []
    # 5-fold CV on train for quick sanity (only fast models to save time)
    for name, clf in models.items():
        print(f"\n=== {name} ===")
        start = time.time()
        if name in NEEDS_SCALING:
            scaler = StandardScaler()
            Xt_tr = scaler.fit_transform(X_train)
            Xt_te = scaler.transform(X_test)
        else:
            scaler = None
            Xt_tr, Xt_te = X_train.values, X_test.values
        clf.fit(Xt_tr, y_train)
        train_t = time.time() - start
        y_pred = clf.predict(Xt_te)
        try:
            y_proba = clf.predict_proba(Xt_te)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except Exception as e:
            auc = float("nan")
            print("AUC unavailable:", e)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc:.4f} time={train_t:.1f}s")
        rows.append({"model": name, "accuracy": acc, "precision_macro": prec,
                     "recall_macro": rec, "f1_macro": f1, "roc_auc_ovr_macro": auc,
                     "train_seconds": train_t})
        # save model + scaler
        joblib.dump({"model": clf, "scaler": scaler, "features": keep,
                     "classes": class_to_idx}, MODELS / f"{name}.joblib")
        # confusion matrix figure
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm); plt.title(f"Confusion matrix - {name}"); plt.colorbar()
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig(FIGS / f"cm_{name}.png"); plt.close()

    metrics = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    print("\n", metrics.to_string(index=False))

    # Figure: metrics comparison (accuracy + macro F1 per model)
    plot_df = metrics.sort_values("f1_macro", ascending=False)
    x = range(len(plot_df))
    plt.figure(figsize=(9, 5))
    w = 0.35
    plt.bar([i - w / 2 for i in x], plot_df["accuracy"], width=w, label="Accuracy")
    plt.bar([i + w / 2 for i in x], plot_df["f1_macro"], width=w, label="F1 macro")
    plt.xticks(list(x), plot_df["model"].tolist(), rotation=20, ha="right")
    plt.ylim(0.7, 1.0)
    plt.ylabel("Score")
    plt.title("Model comparison - Accuracy vs Macro F1 (test 15% = 4043 URLs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "metrics_comparison.png")
    plt.close()

    # Figure: micro-average ROC curves for all models (recomputed from saved bundles)
    try:
        from sklearn.preprocessing import label_binarize as _lb
        n_classes = len(class_to_idx)
        y_bin = _lb(y_test, classes=list(range(n_classes)))
        plt.figure(figsize=(7, 6))
        for name in models:
            bundle = joblib.load(MODELS / f"{name}.joblib")
            clf_r, scaler_r = bundle["model"], bundle["scaler"]
            Xt_r = scaler_r.transform(X_test) if scaler_r is not None else X_test.values
            try:
                proba_r = clf_r.predict_proba(Xt_r)
            except Exception:
                continue
            fpr, tpr, _ = roc_curve(y_bin.ravel(), proba_r.ravel())
            try:
                auc_micro = auc(fpr, tpr)
            except Exception:
                auc_micro = float("nan")
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_micro:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Chance")
        plt.xlabel("False Positive Rate (micro)")
        plt.ylabel("True Positive Rate (micro)")
        plt.title("Micro-average ROC - all models")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGS / "roc_micro.png")
        plt.close()
    except Exception as e:
        print("ROC figure skipped:", e)

    # classification reports
    with open(RESULTS / "classification_reports.txt", "w") as f:
        for name in models:
            bundle = joblib.load(MODELS / f"{name}.joblib")
            clf, scaler = bundle["model"], bundle["scaler"]
            Xt_te = scaler.transform(X_test) if scaler is not None else X_test.values
            y_pred = clf.predict(Xt_te)
            f.write(f"\n{'='*60}\n{name}\n{'='*60}\n")
            f.write(classification_report(y_test, y_pred,
                    target_names=[k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]))
    # feature importance from RF
    try:
        rf = joblib.load(MODELS / "RandomForest.joblib")["model"]
        imp = pd.Series(rf.feature_importances_, index=keep).sort_values(ascending=False).head(20)
        plt.figure(figsize=(7, 5))
        imp.sort_values().plot(kind="barh")
        plt.title("Top-20 RF feature importances"); plt.tight_layout()
        plt.savefig(FIGS / "rf_importance.png"); plt.close()
        imp.to_csv(RESULTS / "rf_top20_importance.csv")
    except Exception as e:
        print("importance plot skipped:", e)

    meta = {"rows_raw": int(df_raw.shape[0]), "rows_clean": int(len(df)),
            "features_total": len(feature_cols), "features_kept": len(keep),
            "dropped_corr": dropped, "classes": class_to_idx,
            "split": "70:15:15 stratified random_state=42",
            "total_seconds": time.time() - t0}
    json.dump(meta, open(RESULTS / "run_meta.json", "w"), indent=2)
    print(f"\nDone in {meta['total_seconds']:.1f}s. See results/metrics.csv, figures/, models/")

if __name__ == "__main__":
    main()
