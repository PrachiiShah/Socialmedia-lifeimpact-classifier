# # Imports and settings
from pathlib import Path
import joblib                             # saves the trained model data to disk for use by app

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

# Data Manipulation
import pandas as pd
import seaborn as sns

# Classification ML Modelling Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Preprocessing, Feature Selection & Model Performance
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                             roc_curve)

# Pipeline Building & Model Selection
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import sys, sklearn, pandas, numpy, matplotlib, seaborn, joblib
print(sys.version)
for m in (sklearn, pandas, numpy, matplotlib, seaborn, joblib):
    print(f"{m.__name__.replace('sklearn', 'scikit-learn')}=={m.__version__}")


HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT = HERE.parent if HERE.name == "model" else HERE
PROJECT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = PROJECT / "data" / "Social_media_impact_on_life.csv"
DATA_FILE = "social-impact-classifier/data/Social_media_impact_on_life.csv"
MODEL_DIR = PROJECT / "model"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR = PROJECT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42                      # change to your own number; keep it fixed afterwards
TEST_FRACTION = 0.20
TARGET = "Overall_Impact"
ID_COL = "Student_ID"
TARGET_CLASS_ORDER = ["Negative", "Neutral", "Beneficial"]
FREQ_COL = "Social_Comparison_Frequency"
FREQ_LEVELS = ["Never", "Rarely", "Sometimes", "Frequently", "Always"]

# Plot Styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

def save_show(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.show()

# Step 1 - Dataset description, Data Quality checks, EDA and clean-up
## Target: `Overall_Impact` (Beneficial / Neutral / Negative)

# 1. Load the data
raw = pd.read_csv(DATA_FILE)
print("Rows x columns:", raw.shape)
print(raw.head().to_string())

# 2. Dataset description
print("\n---Column Data Types ---")
print(raw.dtypes.to_string())

n_rows, n_cols = raw.shape
print(f"\nRows: {n_rows} | Columns: {n_cols} | Predictive features (minus ID and target): {n_cols - 2}")

# 3. Data Quality Checks 
# # Missing Values
print("\n--- Missing Values ---")
miss = raw.isna().sum()
print(miss[miss > 0].to_string() if miss.any() else "none")

# # Duplicate Values
print("\n--- Duplicate Values ---")
print("Duplicate rows:", raw.duplicated().sum(), "| duplicate student ids:", raw[ID_COL].duplicated().sum())

# # Outlier Detection - May need to remove - check later
print("\n--- Outlier Detection (using IQR) ---")
outlier_cols = [c for c in raw.select_dtypes(include=["number"]).columns if c != TARGET]
for col in outlier_cols:
    Q1  = raw[col].quantile(0.25)
    Q3  = raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((raw[col] < lower) | (raw[col] > upper)).sum()
    print(f'{col:<30} Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}  '
          f'Bounds=[{lower:.1f}, {upper:.1f}]  Outliers={n_out:,}')
    
# # Impossible/ inaccurate values
# # # Negative Age - invalid
print(f'\nNegative age: ', (raw['Age'] < 0).sum())

# # # Daily Usage Hours < 24 hours and > 0 hours - i.e. hours shouldn't be negative and not more than 24 hrs for dialy use.
print('\nDaily Usage Limit < 24:', ((raw['Daily_Usage_Hours'] > 24) | (raw['Daily_Usage_Hours'] < 0)).sum())

# # # Weekend_Extra_Hours - should not be negative
print(f'\nWeekend_Extra_Hours < 0: ', (raw['Weekend_Extra_Hours'] < 0).sum())

# # # Sleep_Duration_Hours > 0
print(f'\nSleep_Duration_Hours < 0: ', (raw['Sleep_Duration_Hours'] < 0).sum())

# # # Sleep_Quality_Score shoudl not be negative
print(f'\nSleep_Quality_Score < 0: ', (raw['Sleep_Quality_Score'] < 0).sum())

# # # Perceived_Stress_Score shoudl not be negative
print(f'\nPerceived_Stress_Score < 0: ', (raw['Perceived_Stress_Score'] < 0).sum())

# # # Mental_Health_Index shoudl not be negative
print(f'\nMental_Health_Index < 0: ', (raw['Mental_Health_Index'] < 0).sum())

# # # Academic_Performance_GPA > 0
print(f'\nAcademic_Performance_GPA < 0: ', (raw['Academic_Performance_GPA'] < 0).sum())

# 4. Target Distribution
print("\n--- Target Distribution ---")
dist = raw[TARGET].value_counts().reindex(TARGET_CLASS_ORDER)
print(pd.DataFrame({"rows": dist, "share_%": (dist / dist.sum() * 100).round(2)}).to_string())
plt.bar(dist.index, dist.values,color=['#34a4eb','#3480eb','#3452eb'], edgecolor='white', width=0.5)

plt.title('Target Variable Distribution: Social Media Life Impact', fontsize=12)
plt.xlabel('Overall Impact Category')
plt.ylabel('No. of students impacted')
save_show("01_target_distribution")

# 4. Data Preprocessing
##Grouping the columns by datatype (used by the model pipelines later)
numeric_cols = [c for c in raw.select_dtypes(include=["number", "bool"]).columns if c != TARGET]
nominal_cols = [c for c in raw.columns                      # every remaining text column
                if c not in numeric_cols + [TARGET, ID_COL, FREQ_COL]]
print("Numeric/binary :", numeric_cols)
print("Nominal text   :", nominal_cols)
print("Ordinal        :", [FREQ_COL], "->", FREQ_LEVELS)

# # Numeric Data Summary
print("\n--- Numeric Data Summary ---")
print(raw[numeric_cols].describe().T.round(2).to_string(),"\n")

# # Category values distribution
for col in nominal_cols + [FREQ_COL]:
    print(f"{col}: {raw[col].value_counts().to_dict()}")

# 6. Exploratory Data Analysis (EDA)

# # Target balance and missing values
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
sns.barplot(x=dist.index, y=dist.values, hue=dist.index, palette="crest", legend=False, ax=axes[0])
axes[0].set_title("Overall_Impact class counts")
axes[0].set_xlabel("Category")

miss_pct = (raw.isna().mean() * 100)[lambda s: s > 0]
sns.barplot(x=miss_pct.index, y=miss_pct.values, color="#c9783a", ax=axes[1])
axes[1].set_title("Missing values (%)")
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_xlabel("Missing Value Columns")
save_show("02_target_and_missing")

# # EDA - distributions of numeric features
plt_cols = [c for c in numeric_cols if raw[c].nunique() > 2]
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax, col in zip(axes.ravel(), plt_cols):
    sns.histplot(raw[col].dropna(), kde=True, ax=ax, color="#3b7d8c")
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
for ax in axes.ravel()[len(plt_cols):]:
    ax.axis("off")
save_show("03_numeric_distributions")

# # EDA - numeric features split by target class (which features separate the classes?)
key_cols = ["Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA",
            "Daily_Usage_Hours", "Sleep_Duration_Hours", "Sleep_Quality_Score"]
key_cols = [c for c in key_cols if c in raw.columns]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes.ravel(), key_cols):
    sns.boxplot(data=raw, x=TARGET, y=col, order=TARGET_CLASS_ORDER, hue=TARGET, palette="crest",
                legend=False, ax=ax)
    ax.set_xlabel("")
save_show("04_features_by_target")

# # Correlation Heatmap for Numeric Features
corr = raw[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation of numeric features")
save_show("05_correlation_heatmap")

# # Categorical features vs target (row-wise share of each class)
fig, axes = plt.subplots(1, len(nominal_cols) + 1, figsize=(4 * (len(nominal_cols) + 1), 3.6))
for ax, col in zip(axes, nominal_cols + [FREQ_COL]):
    share = pd.crosstab(raw[col], raw[TARGET], normalize="index")[TARGET_CLASS_ORDER] * 100
    if col == FREQ_COL:
        share = share.reindex(FREQ_LEVELS)
    share.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", legend=False)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
axes[0].legend(TARGET_CLASS_ORDER, fontsize=7)
save_show("06_categoricals_vs_target")

# 7. Clean-up
fn_data = raw.copy()
fn_data = fn_data.drop(columns=[ID_COL])                        # identifier, no predictive meaning
fn_data["Late_Night_Usage"] = fn_data["Late_Night_Usage"].astype(int)   # True/False -> 1/0 (Boolean to binary int)

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

# 8. Stratified train / test split
X = fn_data.drop(columns=[TARGET])
y = fn_data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED)
print("Training Set:", X_train.shape, "| Test Set:", X_test.shape)
print("Test set class counts:", y_test.value_counts().to_dict())

pd.concat([X_test, y_test], axis=1).to_csv(PROJECT / "test_data.csv", index=False)   # Test set needed by the app later
print("\nSaved test_data.csv")

# 9. Preprocessing & Model Eva;uation Functions (kept inside the pipeline so there is no train/test leakage)
numeric_features = [c for c in numeric_cols if c in X.columns]   # numbers + the 0/1 late-night flag
nominal_features = [c for c in nominal_cols if c in X.columns]

# # Preprocessor Pipeline Function
def build_preprocessor() -> ColumnTransformer:
    """Fresh (unfitted) preprocessing block; every model pipeline gets its own copy."""
    return ColumnTransformer([
        ("numeric", Pipeline([("fill_median", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric_features),
        ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
        ("ordinal", Pipeline([("order", OrdinalEncoder(categories=[FREQ_LEVELS])),
                              ("scale", StandardScaler())]), [FREQ_COL]),
    ])

# # Model Evaluation Metric Function
CLASS_LABELS = sorted(y.unique())            # alphabetical = column order of predict_proba
def metrics(y_true, y_pred, y_proba) -> dict:
    """The six metrics for model evaluation."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=CLASS_LABELS),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# # Model Tuning and Evaluation Function
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
RESULTS, BEST_PARAMS, FITTED = {}, {}, {}

def tune_evaluate(name: str, classifier, param_grid) -> Pipeline:
    """Grid-search on the training rows (scored by macro-F1), then evaluate once on the test rows."""
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", classifier)])
    search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=CV, n_jobs=-1, refit=True,
                          error_score="raise")      # never let a failed fit pass silently as NaN
    search.fit(X_train, y_train)
    best = search.best_estimator_

    row = metrics(y_test, best.predict(X_test), best.predict_proba(X_test))
    row["CV macro-F1 (train)"] = search.best_score_
    row["Train macro-F1"] = f1_score(y_train, best.predict(X_train), average="macro")
    RESULTS[name] = row
    BEST_PARAMS[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    FITTED[name] = best
    joblib.dump(best, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    print(f"{name}\n  best params : {BEST_PARAMS[name]}")
    print(f"  CV macro-F1 : {search.best_score_:.4f} | test macro-F1: {row['F1']:.4f} | test MCC: {row['MCC']:.4f}\n")
    return best

# 10. Indvidual Model Tuning
# # Model 1 - Logistic Regression
## Linear model on scaled features. C is the inverse regularisation strength (smaller = stronger penalty).
logreg = tune_evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=3000, random_state=SEED),
    {"clf__C": [0.01, 0.1, 1, 10, 100]},
)

# # Model 2 - Decision Tree Classifier
## Axis-aligned splits; needs no scaling. Depth and leaf size control over-fitting.
tree = tune_evaluate(
    "Decision Tree",
    DecisionTreeClassifier(random_state=SEED),
    {"clf__max_depth": [3, 4, 5, 6, 8, None],
     "clf__min_samples_leaf": [1, 5, 10, 20],
     "clf__criterion": ["gini", "entropy"]},
)

# # Model 3 - K-Nearest Neighbour Classifier
# Distance-based, so the scaling inside the pipeline matters. k, vote weighting and distance type are tuned.
# algorithm is pinned to ball_tree: with text labels, recent scikit-learn versions crash in the automatic
# brute-force path for (uniform weights + Manhattan distance) once there are ~20+ features.
knn = tune_evaluate(
    "kNN",
    KNeighborsClassifier(algorithm="ball_tree"),
    {"clf__n_neighbors": [3, 5, 9, 15, 25],
     "clf__weights": ["uniform", "distance"],
     "clf__p": [1, 2]},                       # 1 = Manhattan, 2 = Euclidean
)

# # Model 4 - Naive Bayes (Gaussian)
# Assumes features are independent given the class and Gaussian within each class.
# var_smoothing adds a floor to the variances for numerical stability.
nbayes = tune_evaluate(
    "Naive Bayes",
    GaussianNB(),
    {"clf__var_smoothing": np.logspace(-9, -1, 9)},
)

# # Model 5 - Support Vector Machine
# probability=True is required so predict_proba (and therefore AUC) is available.
svm = tune_evaluate(
    "Support Vector Machine",
    SVC(probability=True, random_state=SEED),
    [{"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]},
     {"clf__kernel": ["rbf"], "clf__C": [0.5, 1, 5, 10], "clf__gamma": ["scale", 0.01, 0.05]}],
)

# 11. Model Evaluation Metrics

# # Comparison table - the six required metrics checked on the test rows
table = pd.DataFrame(RESULTS).T
table.index.name = "ML Model Name"
metrics_table = table[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].round(4)
print(metrics_table.to_string())
print("\nBest model per metric:")
print(metrics_table.idxmax().to_string(), "\n")

# # Over-fitting check (train vs cross-validation vs test)
print(table[["Train macro-F1", "CV macro-F1 (train)", "F1"]].round(4)
      .rename(columns={"F1": "Test macro-F1"}).to_string())

# # Bar chart of all six metrics
long = metrics_table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 4))
sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="crest")
plt.ylim(0.70, 1.0)
plt.title("Test-set metrics by model (y-axis starts at 0.70)")
plt.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
save_show("07_bar_chart_of_metrics")

# # Confusion matrices (counts) for all five models
print("\nTest-set support:", y_test.value_counts().reindex(TARGET_CLASS_ORDER).to_dict())
fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
for ax, (name, mdl) in zip(axes, FITTED.items()):
    cm = confusion_matrix(y_test, mdl.predict(X_test), labels=TARGET_CLASS_ORDER)
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako_r", cbar=False, ax=ax,
                xticklabels=TARGET_CLASS_ORDER, yticklabels=TARGET_CLASS_ORDER)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
save_show("08_confusion_matrices")

# # Per-class classification reports
for name, mdl in FITTED.items():
    print(f"\n=================== {name} ===================")
    print(classification_report(y_test, mdl.predict(X_test), digits=3, zero_division=0))

# # One-vs-rest ROC curves per class (one panel per model)
fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=True)
for ax, (name, mdl) in zip(axes, FITTED.items()):
    proba = mdl.predict_proba(X_test)
    for i, cls in enumerate(mdl.classes_):
        fpr, tpr, _ = roc_curve((y_test == cls).astype(int), proba[:, i])
        ax.plot(fpr, tpr, label=cls)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].legend(fontsize=8)
save_show("09_roc_curves")

# # Rare-class check - per-class recall for every model
print(f"\n=================== Rare Class Check per Model ===================")
per_class_recall = pd.DataFrame(
    {n: recall_score(y_test, m.predict(X_test), labels=TARGET_CLASS_ORDER, average=None, zero_division=0)
     for n, m in FITTED.items()}, index=TARGET_CLASS_ORDER).T.round(3)
print(per_class_recall.to_string())

