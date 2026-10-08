# Malicious URL Detection using Machine Learning

**Group 19 — Symbiosis Institute of Technology, Academic Year 2026–27**
**Members:** Aarya Balwadkar, Afifa Bintul Hasan, Aparna Nair, Eccha Bansal

**GitHub repository:** https://github.com/AaryaBalwadkar/malicious-url-detection
> The ZIP submitted to the professor already contains everything in this repo plus
> `data/`, `models/` and `report/` so it runs offline.

Lexical-feature ML pipeline on the ISCX-URL2016 mirror (36707 rows, 79 numeric
features, 5 classes: benign / defacement / malware / phishing / spam).
6 classifiers are trained with a 70:15:15 stratified split, correlation pruning
(threshold 0.75 → 34 kept features), and evaluated on Accuracy, macro
Precision / Recall / F1 and ROC-AUC (OVR).

---

## 1. What is in this ZIP (and the GitHub repo)

```
MaliciousURLDetection_Submission/
│  README.md               <- this file (how to execute + GitHub link)
│  HOW_TO_RUN.txt           <- 1-page short version for the evaluator
│  requirements.txt        <- pinned Python dependencies
│  .gitignore              <- keeps .venv / cache out of GitHub
│  run_train.bat           <- Windows double-click executable: full training
│  run_demo.bat            <- Windows double-click executable: demo prediction
│  src/
│    train.py              <- end-to-end training (Sections 6-12 of report)
│    data_utils.py         <- loading / cleaning / label encoding / pruning
│    demo_predict.py       <- demo: predict one held-out test row
│    predict_csv.py        <- predict new feature CSV rows -> predictions.csv
│  data/
│    ISCX-URL2016_All.csv  <- 11.6 MB, 36707 rows x 80 cols (mirror, see §2)
│  models/
│    *.joblib (x6)         <- trained model + scaler + feature list + class map
│  results/
│    metrics.csv, classification_reports.txt,
│    rf_top20_importance.csv, run_meta.json
│  figures/
│    class_distribution.png, correlation_top20.png, cm_*.png (x6),
│    rf_importance.png, metrics_comparison.png, roc_micro.png
│  report/
│    CaseStudy_Report_Group19.pdf / .docx   <- main case-study report
│    IEEE_Conference_Paper_Group19.pdf      <- IEEE-format paper
```

Total unzipped ≈ 88 MB (models ≈ 67 MB, data ≈ 11 MB), zipped ≈ 70–75 MB —
inside the 100 MB upload limit. Nothing else is needed to run.

---

## 2. Dataset provenance

- File used: `data/ISCX-URL2016_All.csv` (mirror of ISCX-URL2016).
- Public mirror:
  `https://raw.githubusercontent.com/quickheaven/scs-3253-machine-learning/master/datasets/ISCX-URL2016_All.csv`
- Original: Mamun et al. 2016, Canadian Institute for Cybersecurity, UNB
  (`https://www.unb.ca/cic/datasets/url-2016.html`), full set ~114400 URLs.
  This mirror is a balanced subset: benign 7781, defacement 7930,
  phishing 7586, malware 6712, spam 6698 (before dedup).
- After exact-duplicate removal: **26953 rows** (honest count in
  `results/run_meta.json`). Label column: `URL_Type_obf_Type`.
- No internet is required — the CSV is already in the ZIP.

---

## 3. Requirements

- Windows 10/11, Python **3.10** (3.10.10 used here). Any 3.10+ works.
- ~2 GB free disk, ~5 minutes for full training (SVM is the slowest step).
- All Python packages are pinned in `requirements.txt`:
  pandas 2.0.3, numpy 1.26.4, scikit-learn 1.3.2, xgboost 2.0.3,
  matplotlib 3.8.2, tldextract 5.1.2, scipy 1.11.4, joblib 1.3.2.

---

## 4. How to execute (fresh machine, step by step)

### Option A — fastest for the evaluator (no training, uses saved models)

```powershell
# 1. Unzip, then open the folder in PowerShell or CMD
cd MaliciousURLDetection_Submission

# 2. (One time) create isolated env and install deps — nothing installed globally
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\pip.exe install -r requirements.txt

# 3. Demo prediction (no training needed, loads models/*.joblib)
.\.venv\Scripts\python.exe src\demo_predict.py --model XGBoost --row 0
.\.venv\Scripts\python.exe src\demo_predict.py --model RandomForest --row 5

# 4. Predict your own feature CSV (same 79 columns as data file)
.\.venv\Scripts\python.exe src\predict_csv.py --input data\ISCX-URL2016_All.csv --model XGBoost --output predictions.csv
```

Or simply **double-click** in Windows Explorer (no terminal needed):

- `run_demo.bat` — demo prediction with XGBoost, row 0
  (`run_demo.bat RandomForest 5` from terminal for other models/rows)
- `run_train.bat` — full retraining (see Option B)

### Option B — full retraining (reproduces models/, results/, figures/)

```powershell
cd MaliciousURLDetection_Submission
.\.venv\Scripts\python.exe src\train.py
```

This rewrites `models/*.joblib`, `results/metrics.csv`,
`results/classification_reports.txt`, `results/rf_top20_importance.csv`,
`results/run_meta.json` and all `figures/*.png`
(including `metrics_comparison.png` and `roc_micro.png`).

Expected runtime ≈ 5 min (SVM_RBF ≈ 170 s dominates). All numbers below
were produced by this exact script — no hand-edited values.

---

## 5. Results (test 15% = 4043 URLs, 70:15:15 stratified, corr-prune 0.75 → 34 features)

| model | acc | prec_macro | rec_macro | f1_macro | auc_ovr |
|---|---|---|---|---|---|
| XGBoost | 0.9693 | 0.9713 | 0.9526 | 0.9611 | 0.9974 |
| RandomForest | 0.9636 | 0.9705 | 0.9408 | 0.9536 | 0.9968 |
| KNN_k5 | 0.9204 | 0.9125 | 0.8999 | 0.9048 | 0.9786 |
| DecisionTree | 0.9159 | 0.8996 | 0.8861 | 0.8920 | 0.9511 |
| SVM_RBF | 0.9001 | 0.9081 | 0.8391 | 0.8605 | 0.9821 |
| LogisticRegression | 0.8046 | 0.7898 | 0.7205 | 0.7329 | 0.9319 |

- Source: `results/metrics.csv`. Per-class details: `results/classification_reports.txt`.
- Figures: `figures/cm_*.png` (confusion matrices), `figures/roc_micro.png`
  (micro-average ROC), `figures/metrics_comparison.png` (accuracy vs F1),
  `figures/rf_importance.png` + `results/rf_top20_importance.csv` (top features:
  NumberofDotsinURL, ArgUrlRatio, domain_token_count, …).
- Interpretation: see report Sections 14–16. Best model: **XGBoost**.

---

## 6. About "exe files"

This is a **Python project — there is no compiled `.exe` by design**
(Python is interpreted; the professor runs `python src/train.py`).
To satisfy the "source + exe in one ZIP" submission rule this package provides:

1. **All source files** (`src/*.py`) — original, commented code.
2. **Windows executables**: `run_train.bat` and `run_demo.bat` — double-click
   to run without typing commands (Windows treats `.bat` as executable).
3. **Pre-trained executables of the models**: `models/*.joblib` — these are the
   runnable artefacts loaded by the demo (equivalent to `.exe` output of a
   compiled project).
4. **Reproducible outputs**: `results/` + `figures/` prove the `.bat`/`.py`
   files actually execute.

If the evaluator strictly requires a single-file `predict.exe`, it can be
built (not included because it adds ~40 MB and would push the ZIP over 100 MB):

```
.\.venv\Scripts\pip.exe install pyinstaller
.\.venv\Scripts\pyinstaller.exe --onefile src\demo_predict.py
REM -> dist\demo_predict.exe (needs models\ + data\ next to it)
```

---

## 7. GitHub link

Public repo (source + results + figures + report; models/data also present —
see `.gitignore` note about Git LFS if push warns on the 57 MB RandomForest file):

**https://github.com/AaryaBalwadkar/malicious-url-detection**

This README is the submission version — re-zip this folder before uploading
to the professor portal.

---

## 8. Verification checklist (for the evaluator)

- [ ] `requirements.txt` installs without error
- [ ] `src/demo_predict.py --model XGBoost --row 0` prints True=defacement,
      Predicted=defacement (or matching pair for other rows)
- [ ] `results/metrics.csv` matches the table in §5
- [ ] `src/train.py` completes and rewrites `results/` + `figures/`
- [ ] ZIP size < 100 MB, single file, contains README + GitHub link
