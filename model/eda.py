import pandas as pd
import seaborn as sns
from model.meta_data import (DATA_FILE, TARGET, ID_COL, TARGET_CLASS_ORDER, FREQ_COL, FREQ_LEVELS,
                             ensure_dirs, column_groups)
from model.plotting import plt, save_fig

ensure_dirs()

# Step 1 - Dataset description, Data Quality checks, EDA and clean-up
## Target: `Overall_Impact` (Beneficial / Neutral / Negative)

# 1. Load the data
raw = pd.read_csv(DATA_FILE)
numeric_cols, nominal_cols = column_groups(raw)                       # numbers + 0/1 flag | text columns
continuous_cols = [c for c in numeric_cols if raw[c].nunique() > 2]   # excludes the 0/1 late-night flag
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
for col in continuous_cols:
    Q1 = raw[col].quantile(0.25)
    Q3 = raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((raw[col] < lower) | (raw[col] > upper)).sum()
    print(f'{col:<30} Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}  '
          f'Bounds=[{lower:.1f}, {upper:.1f}]  Outliers={n_out:,}')

# # Impossible/ inaccurate values
print("\n--- Impossible Values ---")
print("Daily_Usage_Hours outside 0-24:", ((raw["Daily_Usage_Hours"] < 0) | (raw["Daily_Usage_Hours"] > 24)).sum())
for col in ["Age", "Weekend_Extra_Hours", "Sleep_Duration_Hours", "Sleep_Quality_Score",
            "Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA"]:
    print(f"{col} < 0:", (raw[col] < 0).sum())

# 4. Target Distribution
print("\n--- Target Distribution ---")
dist = raw[TARGET].value_counts().reindex(TARGET_CLASS_ORDER)
print(pd.DataFrame({"rows": dist, "share_%": (dist / dist.sum() * 100).round(2)}).to_string())
plt.bar(dist.index, dist.values, color=['#34a4eb', '#3480eb', '#3452eb'], edgecolor='white', width=0.5)
plt.title('Target Variable Distribution: Social Media - Student Life Impact', fontsize=12)
plt.xlabel('Overall Impact Category')
plt.ylabel('# Students')
save_fig("01_target_distribution")

# 5. Data Preprocessing
## Column groups come from column_groups() in meta_data.py (the same lists train.py uses)
print("Numeric/binary :", numeric_cols)
print("Nominal text   :", nominal_cols)
print("Ordinal        :", [FREQ_COL], "->", FREQ_LEVELS)

# # Numeric Data Summary
print("\n--- Numeric Data Summary ---")
print(raw[numeric_cols].describe().T.round(2).to_string(), "\n")

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
if miss_pct.empty:
    axes[1].text(0.5, 0.5, "No missing values", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_axis_off()
else:
    sns.barplot(x=miss_pct.index, y=miss_pct.values, color="#c9783a", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_xlabel("Missing Value Columns")
axes[1].set_title("Missing values (%)")
save_fig("02_target_and_missing")

# # EDA - distributions of numeric features
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax, col in zip(axes.ravel(), continuous_cols):
    sns.histplot(raw[col].dropna(), kde=True, ax=ax, color="#3b7d8c")
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
for ax in axes.ravel()[len(continuous_cols):]:
    ax.axis("off")
save_fig("03_numeric_distributions")

# # EDA - numeric features split by target class (which features separate the classes?)
key_cols = ["Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA",
            "Daily_Usage_Hours", "Sleep_Duration_Hours", "Sleep_Quality_Score"]
key_cols = [c for c in key_cols if c in raw.columns]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes.ravel(), key_cols):
    sns.boxplot(data=raw, x=TARGET, y=col, order=TARGET_CLASS_ORDER, hue=TARGET, palette="crest",
                legend=False, ax=ax)
    ax.set_xlabel("")
save_fig("04_features_by_target")

# # Correlation Heatmap for Numeric Features
corr = raw[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation of numeric features")
save_fig("05_correlation_heatmap")

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
save_fig("06_categoricals_vs_target")
print("\nSaved figures 01-06 to", "figures/")