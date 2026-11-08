# # Imports and settings
import json
import sys

import joblib                             # saves the trained pipelines to disk for the app
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model.meta_data import (DATA_FILE, FREQ_COL, FREQ_LEVELS, ID_COL, META_FILE, MODEL_DIR, SAMPLE_CSV, SEED,
                             TARGET, TARGET_CLASS_ORDER, TEST_FRACTION, column_groups, ensure_dirs, model_file)
from model.scoring import METRIC_NAMES, metrics

# # Model registry: name -> (classifier, hyper-parameter grid). To add a model, add one entry here.
MODELS = {
    # Linear model on scaled features. C is the inverse regularisation strength (smaller = stronger penalty).
    "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=SEED),
                            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    # Axis-aligned splits; needs no scaling. Depth and leaf size control over-fitting.
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"clf__max_depth": [3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 5, 10, 20],
                       "clf__criterion": ["gini", "entropy"]}),
    # Distance-based, so the scaling inside the pipeline matters. algorithm is pinned to ball_tree: with text
    # labels, recent scikit-learn versions crash in the automatic brute-force path (uniform + Manhattan).
    "kNN": (KNeighborsClassifier(algorithm="ball_tree"),
            {"clf__n_neighbors": [3, 5, 9, 15, 25], "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]}),                                    # 1 = Manhattan, 2 = Euclidean
    # Assumes features are independent given the class; var_smoothing adds a floor to the variances.
    "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": np.logspace(-9, -1, 9)}),
    # probability=True is required so predict_proba (and therefore AUC) is available.
    "Support Vector Machine": (SVC(probability=True, random_state=SEED),
                               [{"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]},
                                {"clf__kernel": ["rbf"], "clf__C": [0.5, 1, 5, 10],
                                 "clf__gamma": ["scale", 0.01, 0.05]}]),
}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def build_preprocessor(numeric_features, nominal_features) -> ColumnTransformer:
    """Fresh (unfitted) preprocessing block; kept inside each pipeline so there is no train/test leakage."""
    return ColumnTransformer([
        ("numeric", Pipeline([("fill_median", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric_features),
        ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
        ("ordinal", Pipeline([("order", OrdinalEncoder(categories=[FREQ_LEVELS])),
                              ("scale", StandardScaler())]), [FREQ_COL]),
    ])


def tune_evaluate(name, classifier, param_grid, features, split):
    """Grid-search on the training rows (scored by macro-F1), evaluate once on the test rows, save the pipeline."""
    numeric_features, nominal_features = features
    X_train, X_test, y_train, y_test = split
    pipe = Pipeline([("prep", build_preprocessor(numeric_features, nominal_features)), ("clf", classifier)])
    search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=CV, n_jobs=-1, refit=True,
                          error_score="raise")      # never let a failed fit pass silently as NaN
    search.fit(X_train, y_train)
    best = search.best_estimator_

    row = metrics(y_test, best.predict(X_test), best.predict_proba(X_test), best.classes_)
    row["CV macro-F1 (train)"] = search.best_score_
    row["Train macro-F1"] = f1_score(y_train, best.predict(X_train), average="macro")
    params = {k.replace("clf__", ""): (v.item() if isinstance(v, np.generic) else v)     # plain numbers, not np.float64(...)
              for k, v in search.best_params_.items()}
    joblib.dump(best, MODEL_DIR / model_file(name))

    print(f"{name}\n  best params : {params}")
    print(f"  CV macro-F1 : {search.best_score_:.4f} | test macro-F1: {row['F1']:.4f} | test MCC: {row['MCC']:.4f}\n")
    return best, row, params


def to_builtin(o):                      # numpy values aren't JSON-serialisable
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main() -> None:
    ensure_dirs()
    raw = pd.read_csv(DATA_FILE)
    numeric_cols, nominal_cols = column_groups(raw)

    # 1. Clean-up
    fn_data = raw.drop(columns=[ID_COL])                                    # identifier, no predictive meaning
    fn_data["Late_Night_Usage"] = fn_data["Late_Night_Usage"].astype(int)   # True/False -> 1/0

    # # Sanity ranges - reports problems, does not silently delete rows
    checks = {
        "Age 10-60": fn_data["Age"].between(10, 60),
        "Sleep hours 0-14": fn_data["Sleep_Duration_Hours"].between(0, 14),
        "GPA 0-4 (or missing)": fn_data["Academic_Performance_GPA"].between(0, 4) | fn_data["Academic_Performance_GPA"].isna(),
    }
    for label, ok in checks.items():
        print(f"{label:25s} violations: {(~ok).sum()}")
    print("\nRemaining missing values (filled inside the model pipeline):")
    print(fn_data.isna().sum()[lambda s: s > 0].to_string())
    print("Remaining columns after clean-up:", fn_data.shape[1] - 1, "features + target")

    # 2. Stratified train / test split
    X = fn_data.drop(columns=[TARGET])
    y = fn_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED)
    print("Training Set:", X_train.shape, "| Test Set:", X_test.shape)
    print("Test set class counts:", y_test.value_counts().to_dict())
    pd.concat([X_test, y_test], axis=1).to_csv(S
                                               AMPLE_CSV, index=False)     # test set, used by evaluate.py and the app
    print(f"\nSaved {SAMPLE_CSV.name}\n")

    numeric_features = [c for c in numeric_cols if c in X.columns]          # numbers + the 0/1 late-night flag
    nominal_features = [c for c in nominal_cols if c in X.columns]
    features = (numeric_features, nominal_features)
    split = (X_train, X_test, y_train, y_test)

    # 3. Tune, evaluate and save every model in the registry
    fitted, results, best_params = {}, {}, {}
    for name, (clf, grid) in MODELS.items():
        fitted[name], results[name], best_params[name] = tune_evaluate(name, clf, grid, features, split)

    # 4. Run metadata: everything the evaluate script and the app need, so they never hard-code it
    meta = {
        "seed": SEED, "test_fraction": TEST_FRACTION,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
        "target": TARGET, "id_col": ID_COL,
        "classes": list(next(iter(fitted.values())).classes_),              # predict_proba column order
        "class_display_order": TARGET_CLASS_ORDER,
        "features": {"numeric": numeric_features, "nominal": nominal_features, "ordinal": {FREQ_COL: FREQ_LEVELS}},
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in numeric_features},
        "medians": {c: float(X[c].median()) for c in numeric_features},
        "categories": {c: sorted(X[c].dropna().unique().tolist()) for c in nominal_features},
        "models": {n: {"file": model_file(n), "best_params": best_params[n],
                       "cv_macro_f1": results[n]["CV macro-F1 (train)"],
                       "train_macro_f1": results[n]["Train macro-F1"],
                       "test_metrics": {k: results[n][k] for k in METRIC_NAMES}} for n in fitted},
    }
    META_FILE.write_text(json.dumps(meta, indent=2, default=to_builtin))
    print(f"Saved {len(fitted)} models and {META_FILE.name}")
    
if __name__ == "__main__":      # required on Windows: GridSearchCV(n_jobs=-1) starts worker processes
    main()