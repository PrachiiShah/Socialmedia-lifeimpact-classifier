# # Imports and settings
import html
import warnings
from pathlib import Path
import joblib                             # saves the trained model data to disk for use by app
import streamlit as st

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

# Data Manipulation
import pandas as pd
import seaborn as sns

# Preprocessing, Feature Selection & Model Performance
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                             roc_curve)

import sys, sklearn, pandas, numpy, matplotlib, seaborn, joblib, streamlit
print(sys.version)
for m in (sklearn, pandas, numpy, matplotlib, seaborn, joblib, streamlit):
    print(f"{m.__name__.replace('sklearn', 'scikit-learn')}=={m.__version__}")

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT = HERE.parent if HERE.name == "model" else HERE      # always the repo root
DATA_FILE = PROJECT / "data" / "Social_media_impact_on_life.csv"
SAMPLE_CSV = HERE / "test_data.csv"
MODEL_DIR = PROJECT / "model"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Overall_Impact"
ID_COL = "Student_ID"
TARGET_CLASS_ORDER = ["Negative", "Neutral", "Beneficial"]
FREQ_COL = "Social_Comparison_Frequency"
FREQ_LEVELS = ["Never", "Rarely", "Sometimes", "Frequently", "Always"]
NUMERIC_FEATURES = ["Age", "Daily_Usage_Hours", "Weekend_Extra_Hours", "Sleep_Duration_Hours",
                    "Sleep_Quality_Score", "Late_Night_Usage", "Perceived_Stress_Score",
                    "Mental_Health_Index", "Academic_Performance_GPA"]
NOMINAL_FEATURES = ["Gender", "Academic_Level", "Primary_Platform", "Device_Type"]
FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + [FREQ_COL]

MODEL_FILES = {                       # dropdown label -> saved pipeline
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
MODEL_BLURB = {
    "Logistic Regression": "Linear model on scaled features; fast and easy to interpret.",
    "Decision Tree": "Rule-based splits; no scaling needed but prone to over-fitting.",
    "kNN": "Classifies by the closest training students; sensitive to scaling and dimensions.",
    "Naive Bayes": "Probabilistic model assuming features are independent given the class.",
    "Support Vector Machine": "Finds the widest-margin boundary between classes.",
}

st.set_page_config(page_title="Screen-Time Impact Lab", page_icon="📱", layout="wide")

THEME_CSS = """
<style>
/* page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f9ff 0%, #e8f1ff 45%, #d9e8ff 100%);
}
[data-testid="stHeader"] { background: transparent; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #dcebff 0%, #c4dcff 55%, #adcdff 100%);
    border-right: 1px solid #a9c8f5;
}
[data-testid="stSidebar"] h1 { color: #0b3d91; }

/* hero banner */
.hero {
    background: linear-gradient(120deg, #0b3d91 0%, #1f6feb 55%, #4aa3ff 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem 1.4rem 2rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 6px 18px rgba(31, 111, 235, 0.28);
}
.hero h1 { color: #ffffff !important; margin: 0 0 .45rem 0; padding: 0; font-size: 2.2rem; line-height: 1.2; }
.hero p  { color: #eaf3ff !important; margin: 0; font-size: 1.05rem; line-height: 1.5; }

/* data status line */
.data-strip {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #b7d3fb;
    border-left: 5px solid #1f6feb;
    border-radius: 10px;
    padding: .55rem 1rem;
    margin-bottom: 1rem;
    color: #22456f;
    font-size: .93rem;
}

/* metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border-left: 5px solid #1f6feb;
    border-radius: 12px;
    padding: .7rem 1rem;
    box-shadow: 0 2px 10px rgba(11, 61, 145, 0.10);
}
[data-testid="stMetricLabel"] p { color: #4a6585; font-weight: 600; }
[data-testid="stMetricValue"] { color: #0b3d91; }

/* tabs */
button[data-baseweb="tab"] { font-weight: 600; color: #3a5a85; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1f6feb; }
div[data-baseweb="tab-highlight"] { background-color: #1f6feb; }

/* section headings and buttons */
h2, h3 { color: #0b3d91; }
.stDownloadButton button {
    background: linear-gradient(90deg, #1f6feb, #4aa3ff);
    color: #ffffff; border: none; border-radius: 10px; font-weight: 600;
}
.stDownloadButton button:hover { filter: brightness(1.08); color: #ffffff; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def load_model(label: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[label])


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame | None:
    return pd.read_csv(SAMPLE_CSV) if SAMPLE_CSV.exists() else None


def prepare(frame: pd.DataFrame):
    """The uploaded file should same as the training format. Returns (clean_frame, problems, notes)."""
    problems, notes = [], []
    frame = frame.copy()
    if "Student_ID" in frame.columns:
        frame = frame.drop(columns="Student_ID")
        notes.append("Ignored the Student_ID column.")

    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        return frame, [f"Missing required column(s): {', '.join(missing)}"], notes

    # True/False (original Kaggle format) or 1/0 -> 1/0
    late = frame["Late_Night_Usage"]
    frame["Late_Night_Usage"] = late.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0})
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    bad_late = frame["Late_Night_Usage"].isna().sum()
    if bad_late:
        problems.append(f"Late_Night_Usage has {bad_late} value(s) that are not True/False or 1/0.")

    bad_freq = ~frame[FREQ_COL].isin(FREQ_LEVELS)
    if bad_freq.any():
        problems.append(f"{FREQ_COL} must be one of {FREQ_LEVELS}; found {bad_freq.sum()} invalid/blank value(s).")

    blanks = frame[NUMERIC_FEATURES].isna().sum().sum()
    if blanks:
        notes.append(f"{int(blanks)} blank numeric cell(s) will be filled with the training median, as in training.")
    return frame, problems, notes


def metrics(y_true, y_pred, y_proba, classes) -> dict:
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores["AUC"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=list(classes))
    except ValueError:                       # the uploaded file does not contain every class
        pass
    return scores


def fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📱 Controls")
    model_name = st.selectbox("1 · Choose a model", list(MODEL_FILES))
    st.caption(MODEL_BLURB[model_name])

    st.markdown("**2 · Provide test data**")
    upload = st.file_uploader("Upload a CSV", type="csv",
                              help="Use test_data.csv from the repo, or any file with the same columns.")
    use_sample = st.checkbox("Use the bundled test_data.csv", value=True, disabled=upload is not None)
    pct_view = st.toggle("Confusion matrix as row %", value=False)

    st.divider()
    with st.expander("Required columns"):
        st.write(", ".join(FEATURES))
        st.caption(f"Optional: {TARGET} (true labels, needed for metrics).")

# ----------------------------------------------------------------------------- header + data
st.markdown(
    """
    <div class="hero">
      <h1>Social Media - Life Impact Analysis</h1>
      <p>How does the social media impact a student's life? Predict <b>Overall Impact</b>
      (<i>Beneficial / Neutral / Negative</i>) and compare five classic ML classifiers on the same data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if upload is not None:
    source_df, source_label = pd.read_csv(upload), f"uploaded file “{upload.name}”"
elif use_sample and load_sample() is not None:
    source_df, source_label = load_sample(), "bundled test_data.csv"
else:
    source_df, source_label = None, ""

data, problems, notes = None, [], []
if source_df is not None:
    prepared, problems, notes = prepare(source_df)
    data = None if problems else prepared

if data is not None:                         # data status line, shown above the tabs
    status = f"Data: {source_label} · {len(data):,} rows" + (" · " + " ".join(notes) if notes else "")
    st.markdown(f'<div class="data-strip">{html.escape(status)}</div>', unsafe_allow_html=True)

tab_perf, tab_compare, tab_rows, tab_whatif = st.tabs(
    ["📊 Performance", "🏁 Compare all models", "🔎 Predictions", "🧪 What-if student"])

model = load_model(model_name)
classes = list(model.classes_)

if source_df is None:
    for tab in (tab_perf, tab_compare, tab_rows):
        with tab:
            st.info("Upload a CSV or tick “Use the bundled test_data.csv” in the sidebar to begin.")
    data = None
else:
    data, problems, notes = prepare(source_df)
    if problems:
        for tab in (tab_perf, tab_compare, tab_rows):
            with tab:
                st.error("Cannot score this file:\n\n- " + "\n- ".join(problems))
        data = None
    else:
        with tab_perf:
            st.caption(f"Data: {source_label} · {len(data):,} rows" + (" · " + " ".join(notes) if notes else ""))

if data is not None:
    X_in = data[FEATURES]
    has_truth = TARGET in data.columns
    unknown_labels = sorted(set(data[TARGET].dropna()) - set(classes)) if has_truth else []
    if unknown_labels:
        has_truth = False                    # cannot score labels the models were never trained on
    pred = model.predict(X_in)
    proba = model.predict_proba(X_in)

    # ------------------------------------------------------------------ tab 1: performance
    with tab_perf:
        if unknown_labels:
            st.error(f"{TARGET} contains label(s) the models were never trained on: {unknown_labels}. "
                     f"Expected: {CLASS_ORDER}.")
        elif not has_truth:
            st.warning(f"No “{TARGET}” column in this file, so only predictions are available (see the Predictions tab).")
        else:
            truth = data[TARGET]
            scores = metrics(truth, pred, proba, classes)
            for col, (name, val) in zip(st.columns(6), scores.items()):
                col.metric(name, fmt(val))
            st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest (macro).")

            order = [c for c in CLASS_ORDER if c in classes]
            left, right = st.columns([1, 1.15])
            with left:
                st.subheader("Confusion matrix")
                cm = confusion_matrix(truth, pred, labels=order)
                row_share = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)      # 0-1 per actual class
                labels_txt = np.array([[f"{v * 100:.1f}%" if pct_view else f"{c:d}"
                                        for c, v in zip(cm_row, share_row)]
                                       for cm_row, share_row in zip(cm, row_share)])
                fig, ax = plt.subplots(figsize=(4.8, 3.9))
                sns.heatmap(row_share, annot=labels_txt, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                            xticklabels=order, yticklabels=order, linewidths=1, linecolor="white", ax=ax)
                ax.set_yticklabels(order, rotation=0)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close(fig)
            with right:
                st.subheader("Classification report")
                report = pd.DataFrame(classification_report(truth, pred, output_dict=True, zero_division=0)).T
                report.loc["accuracy", "support"] = report.loc["macro avg", "support"]   # total rows
                report["support"] = report["support"].astype(int)
                st.dataframe(report.round(3), width="stretch")
                if "Negative" in report.index and report.loc["Negative", "support"] > 0:
                    r = report.loc["Negative"]
                    st.info(f"Rare class **Negative**: recall {r['recall']:.2f}, precision {r['precision']:.2f} "
                            f"(only {int(r['support'])} rows), so it moves the macro scores the most.")

    # ------------------------------------------------------------------ tab 2: compare
    with tab_compare:
        if not has_truth:
            st.warning(f"A valid “{TARGET}” column is needed to compare models.")
        else:
            live = {}
            for label in MODEL_FILES:
                m = load_model(label)
                live[label] = metrics(truth, m.predict(X_in), m.predict_proba(X_in), list(m.classes_))
            board = pd.DataFrame(live).T
            board.index.name = "ML Model Name"
            st.subheader("All five models on this data")
            st.dataframe(board.round(4).style.highlight_max(axis=0, color="#b9d7ff").format("{:.4f}"),
                         width="stretch")
            pick = st.selectbox("Chart a metric", list(board.columns), index=4)
            fig2, ax2 = plt.subplots(figsize=(8, 3.2))
            best_model = board[pick].idxmax()
            bar_colors = ["#1f6feb" if m == best_model else "#a9c9f7" for m in board.index]
            ax2.bar(board.index, board[pick], color=bar_colors)
            for x, v in enumerate(board[pick]):
                ax2.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#0b3d91")
            ax2.set_ylabel(pick)
            ax2.set_xlabel("")
            ax2.set_ylim(max(0, float(board[pick].min()) - 0.05), 1.02)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.setp(ax2.get_xticklabels(), rotation=15)
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption(f"Best on this file by {pick}: **{board[pick].idxmax()}**")

    # ------------------------------------------------------------------ tab 3: predictions
    with tab_rows:
        out = data.copy()
        out["Predicted_Impact"] = pred
        for i, c in enumerate(classes):
            out[f"P({c})"] = proba[:, i].round(3)
        if has_truth:
            out["Correct"] = out[TARGET] == out["Predicted_Impact"]
            if st.checkbox("Show only misclassified rows"):
                out = out[~out["Correct"]]
        st.caption(f"{len(out):,} rows shown · model: {model_name}")
        st.dataframe(out, width="stretch", height=380)
        st.download_button("⬇️ Download predictions as CSV", out.to_csv(index=False).encode(),
                           file_name="predictions.csv", mime="text/csv")

# ----------------------------------------------------------------------------- tab 4: what-if (independent of uploads)
with tab_whatif:
    st.subheader("Try a single student")
    ref = load_sample()
    if ref is None or any(c not in ref.columns for c in FEATURES):
        st.info("A valid test_data.csv (with all feature columns) is needed to set the input ranges for this tab.")
    else:
        st.caption(f"Adjust the inputs; the selected model ({model_name}) predicts the overall impact.")
        c1, c2, c3 = st.columns(3)
        values = {}
        for i, col in enumerate(NUMERIC_FEATURES):
            target_col = (c1, c2, c3)[i % 3]
            s = ref[col].dropna()
            if col == "Late_Night_Usage":
                values[col] = int(target_col.checkbox("Late-night usage", value=bool(round(s.mean()))))
            elif col in ("Age", "Sleep_Quality_Score"):
                values[col] = target_col.slider(col.replace("_", " "), int(s.min()), int(s.max()), int(s.median()))
            else:
                values[col] = target_col.slider(col.replace("_", " "), float(s.min()), float(s.max()),
                                                float(round(s.median(), 1)), step=0.1)
        for i, col in enumerate(NOMINAL_FEATURES):
            opts = sorted(ref[col].dropna().unique())
            values[col] = (c1, c2, c3)[i % 3].selectbox(col.replace("_", " "), opts)
        values[FREQ_COL] = c3.select_slider("Social comparison frequency", FREQ_LEVELS, value="Sometimes")

        one = pd.DataFrame([values])[FEATURES]
        probs = model.predict_proba(one)[0]
        top = classes[int(np.argmax(probs))]
        st.markdown(f"### Predicted impact: **{top}**")
        chart = pd.Series(probs, index=classes).reindex([c for c in CLASS_ORDER if c in classes])
        st.bar_chart(chart, y_label="probability", color="#1f6feb", height=260)
        st.caption("Educational demo built on a survey dataset; not a diagnostic tool.")
