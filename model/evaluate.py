import json, joblib
import pandas as pd, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_curve
from model.meta_data import META_FILE, MODEL_DIR, SAMPLE_CSV, TARGET, TARGET_CLASS_ORDER
from model.plotting import plt, save_fig
from model.scoring import metrics

meta = json.loads(META_FILE.read_text())

if any("test_metrics" not in m or "train_macro_f1" not in m for m in meta["models"].values()):
    raise SystemExit("run_meta.json comes from an older train.py. Re-run: python -m model.train")

feat = meta["features"]
FEATURES = feat["numeric"] + feat["nominal"] + list(feat["ordinal"])
test = pd.read_csv(SAMPLE_CSV)
X_test, y_test = test[FEATURES], test[TARGET]
FITTED = {n: joblib.load(MODEL_DIR / m["file"]) for n, m in meta["models"].items()}
RESULTS = {n: {**metrics(y_test, m.predict(X_test), m.predict_proba(X_test), m.classes_),
               "CV macro-F1 (train)": meta["models"][n]["cv_macro_f1"],
               "Train macro-F1": meta["models"][n]["train_macro_f1"]} for n, m in FITTED.items()}

for n, m in meta["models"].items():
    if abs(RESULTS[n]["F1"] - m["test_metrics"]["F1"]) > 1e-4:
        print(f"WARNING: {n} F1 differs from training ({RESULTS[n]['F1']:.4f} vs {m['test_metrics']['F1']:.4f}); "
              f"test_data.csv or the model file may have changed since train.py ran.")
        
# Model Evaluation Metrics

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
floor = max(0.0, long["Score"].min() - 0.05)
plt.ylim(floor, 1.0)
plt.title(f"Test-set metrics by model (y-axis starts at {floor:.2f})")
plt.xlabel("")
plt.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
save_fig("07_bar_chart_of_metrics")

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
save_fig("08_confusion_matrices")

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
save_fig("09_roc_curves")

# # Rare-class check - per-class recall for every model
print("\n=================== Rare Class Check per Model ===================")
per_class_recall = pd.DataFrame(
    {n: recall_score(y_test, m.predict(X_test), labels=TARGET_CLASS_ORDER, average=None, zero_division=0)
     for n, m in FITTED.items()}, index=TARGET_CLASS_ORDER).T.round(3)
print(per_class_recall.to_string())