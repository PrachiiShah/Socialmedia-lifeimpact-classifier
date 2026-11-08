# # Imports and settings
import html
import json

import joblib                             # loads the pipelines saved by model/train.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.meta_data import META_FILE, MODEL_DIR, SAMPLE_CSV     # paths only: the app never creates folders
from model.scoring import metrics                                 # same metric code as train.py / evaluate.py

st.set_page_config(page_title="Social Media - Life Impact Analysis", page_icon="📱", layout="wide",
                   initial_sidebar_state="collapsed")            # must be the first Streamlit call

# # ------------------- everything about the data and models comes from run_meta.json -------------------
if not META_FILE.exists():
    st.error(f"Metadata file not found at: {META_FILE}. Run `python -m model.train` and commit the model/ folder.")
    st.stop()
META = json.loads(META_FILE.read_text())

TARGET = META["target"]
ID_COL = META["id_col"]
TARGET_CLASS_ORDER = META["class_display_order"]
NUMERIC_FEATURES = META["features"]["numeric"]
NOMINAL_FEATURES = META["features"]["nominal"]
FREQ_COL, FREQ_LEVELS = next(iter(META["features"]["ordinal"].items()))
FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + [FREQ_COL]
MODEL_FILES = {name: m["file"] for name, m in META["models"].items()}    # dropdown label -> saved pipeline

missing_files = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).exists()]
if missing_files:
    st.error(f"Missing model file(s) in model/: {', '.join(missing_files)}. "
             f"Run `python -m model.train` and commit the model/ folder.")
    st.stop()
TRAINED_WITH = META["versions"]["scikit-learn"]

MODEL_BLURB = {
    "Logistic Regression": "Linear model on scaled features; fast and easy to interpret.",
    "Decision Tree": "Rule-based splits; no scaling needed but prone to over-fitting.",
    "kNN": "Classifies by the closest training students; sensitive to scaling and dimensions.",
    "Naive Bayes": "Probabilistic model assuming features are independent given the class.",
    "Support Vector Machine": "Finds the widest-margin boundary between classes.",
}

# --- presentation-only constants (design) ---
LOW_CONF = 0.85                                   # threshold for the "highlight low confidence" option
CLASS_COLOR = {"Negative": "#f43f5e", "Neutral": "#f59e0b", "Beneficial": "#10b981"}
CLASS_FACE = {"Negative": "😞", "Neutral": "😐", "Beneficial": "😊"}
METRIC_COLOR = {"Accuracy": "#6366f1", "AUC": "#6366f1", "Precision": "#3b82f6",
                "Recall": "#06b6d4", "F1": "#a855f7", "MCC": "#10b981"}
BRAND = "#4f46e5"


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ---------- base ---------- */
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stAppViewContainer"] { background: #f8fafc; color-scheme: light; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { max-width: 1280px; padding-top: 1.2rem; padding-bottom: 3rem; }
.stApp, .stApp p, .stApp label { color: #1e293b; }
[data-testid="stCaptionContainer"] p, .stApp small { color: #64748b; font-size: .78rem; }
[data-testid="stAlert"] { border-radius: 12px; }

/* ---------- top bar ---------- */
.topbar {
    display: flex; align-items: center; gap: .75rem;
    background: rgba(255,255,255,.85); border: 1px solid #e2e8f0; border-radius: 16px;
    padding: .7rem 1.2rem; margin-bottom: 1.1rem;
}
.topbar .logo {
    width: 2rem; height: 2rem; border-radius: .6rem; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(45deg, #4f46e5, #818cf8); box-shadow: 0 4px 10px rgba(79,70,229,.25); font-size: 1rem;
}
.topbar .t1 { font-weight: 700; color: #0f172a; font-size: 1rem; line-height: 1.1; }
.topbar .t2 { color: #94a3b8; font-size: .75rem; margin-top: .15rem; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 24px; color: #fff;
    background: linear-gradient(90deg, #4f46e5 0%, #4f46e5 40%, #1d4ed8 100%);
    padding: 2rem 2.2rem; margin-bottom: 1.2rem; box-shadow: 0 16px 30px rgba(79,70,229,.14);
}
.hero::after {
    content: ""; position: absolute; right: -2.5rem; bottom: -2.5rem; width: 16rem; height: 16rem;
    background: rgba(255,255,255,.10); border-radius: 50%; filter: blur(24px); pointer-events: none;
}
.hero-inner { position: relative; z-index: 1; max-width: 48rem; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.2);
    color: #e0e7ff !important; font-size: .75rem; font-weight: 500; padding: .25rem .75rem; border-radius: 999px;
}
.hero-title { color: #fff !important; font-size: 2rem; font-weight: 800; letter-spacing: -.02em; margin: .75rem 0 .5rem 0; line-height: 1.15; }
.hero-text { color: #f1f5f9 !important; font-size: .92rem; line-height: 1.6; margin: 0; opacity: .92; }
.hero-text strong { color: #6ee7b7 !important; }

/* ---------- cards (st.container(key="card_*")) ---------- */
[class*="st-key-card_"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.4rem 1.5rem; box-shadow: 0 1px 2px rgba(15,23,42,.06);
}
[class*="st-key-box_"] {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: .7rem 1rem;
}
.card-head {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
    border-bottom: 1px solid #f1f5f9; padding-bottom: .75rem; margin-bottom: .25rem;
}
.card-head .left { display: flex; align-items: center; gap: .55rem; }
.card-head .step {
    width: 1.5rem; height: 1.5rem; border-radius: 50%; background: rgba(99,102,241,.2); color: #4f46e5;
    font-size: .75rem; font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
}
.card-title { font-weight: 700; color: #0f172a !important; font-size: 1rem; }
.card-sub { color: #64748b !important; font-size: .75rem; margin-top: .1rem; }
.pill {
    font-size: .75rem; font-weight: 600; padding: .25rem .65rem; background: #f1f5f9; color: #475569 !important;
    border-radius: 8px;
}
.chip {
    display: inline-flex; align-items: center; gap: .4rem; font-size: .75rem; color: #64748b !important;
    background: #f1f5f9; border: 1px solid #e2e8f0; padding: .35rem .75rem; border-radius: 10px;
}
.lbl { font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #94a3b8 !important; margin-bottom: .3rem; }

/* ---------- inputs ---------- */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #e2e8f0; border-radius: 16px; background: rgba(248,250,252,.6);
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #6366f1; }
[data-baseweb="select"] > div { background: #f8fafc; border-color: #e2e8f0; border-radius: 12px; font-weight: 600; }
.stApp [data-baseweb="select"]:focus-within > div { border-color: #6366f1; box-shadow: 0 0 0 2px rgba(99,102,241,.35); }
[data-testid="stTextInput"] input { background: #f8fafc; border-radius: 12px; }
[data-testid="stSlider"] [role="slider"] { background-color: #4f46e5; }
[data-testid="stCheckbox"] input:checked + div, [data-testid="stToggle"] input:checked + div { background-color: #4f46e5; }

/* ---------- metric strip ---------- */
.metric-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1rem; margin: 1.1rem 0 .3rem 0; }
@media (max-width: 1100px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 640px)  { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
.metric-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.06);
}
.metric-card .k { font-size: .75rem; font-weight: 600; color: #94a3b8 !important; margin-bottom: .25rem; }
.metric-card .v { font-size: 1.5rem; font-weight: 800; color: #0f172a !important; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.metric-card .track { width: 100%; height: 6px; background: #f1f5f9; border-radius: 999px; margin-top: .5rem; overflow: hidden; }
.metric-card .fill { height: 100%; border-radius: 999px; }

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] { gap: .6rem; }
button[data-baseweb="tab"] { font-weight: 600; color: #64748b; padding: .75rem 1rem; }
button[data-baseweb="tab"] p { color: inherit; font-size: .88rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: #4f46e5; }
div[data-baseweb="tab-highlight"] { background-color: #4f46e5; }
div[data-baseweb="tab-border"] { background-color: #e2e8f0; }

/* ---------- confusion matrix ---------- */
.cm-grid { display: grid; gap: .5rem; align-items: center; margin: 1rem 0 .5rem 0; font-size: .78rem; }
.cm-colh { text-align: center; font-weight: 700; color: #94a3b8; padding-bottom: .15rem; }
.cm-rowh { text-align: right; font-weight: 700; color: #475569; padding-right: .5rem; }
.cm-cell { padding: 1rem .25rem; border-radius: 12px; text-align: center; font-weight: 600; background: #f1f5f9; color: #475569; }
.cm-diag { color: #fff; font-weight: 700; box-shadow: 0 1px 2px rgba(15,23,42,.15); }

/* ---------- what-if result ---------- */
.sim-result { border-radius: 16px; text-align: center; padding: 1.4rem 1rem; }
.sim-badge {
    display: inline-block; color: #fff !important; font-weight: 700; font-size: 1.05rem;
    padding: .5rem 1.1rem; border-radius: 999px; margin: .6rem 0 .8rem 0;
}
.sim-conf-label { font-size: .75rem; color: #94a3b8 !important; }
.sim-conf { font-size: 1.6rem; font-weight: 700; color: #0f172a !important; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

/* ---------- export buttons ---------- */
.stDownloadButton button {
    background: #f1f5f9; color: #334155; border: none; border-radius: 12px; font-weight: 600; font-size: .8rem;
    padding: .55rem 1rem;
}
.stDownloadButton button:hover { background: #e2e8f0; color: #0f172a; }
.stDownloadButton button p { color: inherit; }
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
    if ID_COL in frame.columns:
        frame = frame.drop(columns=ID_COL)
        notes.append(f"Ignored the {ID_COL} column.")

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


@st.cache_data(show_spinner=False)
def score_all(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Metrics of every saved model on the same rows (cached, so toggles don't re-score five models)."""
    rows = {}
    for name in MODEL_FILES:
        m = load_model(name)
        rows[name] = metrics(y, m.predict(X), m.predict_proba(X), list(m.classes_))
    return pd.DataFrame(rows).T


def fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"


# ----------------------------------------------------------------------------- html builders (design only)
def card_head(title: str, subtitle: str = "", badge: str = "") -> str:
    sub = f'<div class="card-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    pill = f'<span class="pill">{html.escape(badge)}</span>' if badge else ""
    return (f'<div class="card-head"><div><div class="card-title">{html.escape(title)}</div>{sub}</div>{pill}</div>')


def step_head(number: int, title: str, chip: str = "") -> str:
    chip_html = f'<span class="chip">🗄️ {html.escape(chip)}</span>' if chip else ""
    return (f'<div class="card-head"><div class="left"><span class="step">{number}</span>'
            f'<span class="card-title">{html.escape(title)}</span></div>{chip_html}</div>')


def metric_strip_html(scores: dict) -> str:
    cards = []
    for name, val in scores.items():
        width = 0.0 if pd.isna(val) else min(max(float(val), 0.0), 1.0) * 100
        cards.append(
            f'<div class="metric-card"><div class="k">{html.escape(name)}</div><div class="v">{fmt(val)}</div>'
            f'<div class="track"><div class="fill" style="width:{width:.1f}%;background:{METRIC_COLOR[name]}"></div></div></div>')
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def confusion_html(cell_text, order) -> str:
    n = len(order)
    out = [f'<div class="cm-grid" style="grid-template-columns: 6rem repeat({n}, 1fr);"><div></div>']
    out += [f'<div class="cm-colh">Pred: {html.escape(c)}</div>' for c in order]
    for i, actual in enumerate(order):
        out.append(f'<div class="cm-rowh">{html.escape(actual)}</div>')
        for j in range(n):
            if i == j:
                color = CLASS_COLOR.get(actual, BRAND)
                out.append(f'<div class="cm-cell cm-diag" style="background:{color}">{html.escape(cell_text[i][j])}</div>')
            else:
                out.append(f'<div class="cm-cell">{html.escape(cell_text[i][j])}</div>')
    out.append("</div>")
    return "".join(out)


# ----------------------------------------------------------------------------- header + hero
st.markdown(
    '<div class="topbar"><div class="logo">🧠</div><div><div class="t1">Social Impact Intelligence</div>'
    '<div class="t2">Machine Learning Student Wellbeing Evaluator</div></div></div>'
    '<div class="hero"><div class="hero-inner">'
    '<span class="hero-badge">✨ ML Behavioral Insights Project</span>'
    '<div class="hero-title">Social Media — Life Impact Analysis</div>'
    '<p class="hero-text">How does the social media impact a student\'s life? Predict <strong>Overall Impact</strong> '
    '(<em>Beneficial / Neutral / Negative</em>) and compare five classic ML classifiers on the same data.</p>'
    '</div></div>',
    unsafe_allow_html=True,
)
if TRAINED_WITH != sklearn.__version__:
    st.warning(f"The models were trained with scikit-learn {TRAINED_WITH} but this app runs {sklearn.__version__}; "
               f"results may be unreliable. Pin scikit-learn=={TRAINED_WITH} in requirements.txt.")

# ----------------------------------------------------------------------------- 1 · provide test data
with st.container(key="card_data"):
    data_head = st.empty()                   # filled in once the data has been read (needs row count / notes)
    up_col, use_col = st.columns(2, vertical_alignment="center")
    with up_col:
        upload = st.file_uploader("Upload a CSV", type="csv", label_visibility="collapsed",
                                  help="Use test_data.csv from the repo, or any file with the same columns.")
    with use_col:
        with st.container(key="box_bundled"):
            use_sample = st.checkbox("Use the bundled test_data.csv", value=True, disabled=upload is not None)
            st.caption("Use test_data.csv from the repo, or upload any file with the same columns.")
    with st.expander("Required columns"):
        st.write(", ".join(FEATURES))
        st.caption(f"Optional: {TARGET} (true labels, needed for metrics).")

# ----------------------------------------------------------------------------- 2 · classifier & display settings
with st.container(key="card_controls"):
    st.markdown(step_head(2, "Classifier & Display Settings"), unsafe_allow_html=True)
    ctl_model, ctl_cm, ctl_low = st.columns(3)
    with ctl_model:
        st.markdown('<div class="lbl">1 · Choose a model</div>', unsafe_allow_html=True)
        model_name = st.selectbox("Choose a model", list(MODEL_FILES), label_visibility="collapsed")
        st.caption(MODEL_BLURB.get(model_name, ""))
        tuned = ", ".join(f"{k}={v}" for k, v in META["models"][model_name]["best_params"].items())
        st.caption(f"Tuned parameters: {tuned}")
    with ctl_cm:
        st.markdown('<div class="lbl">Display options</div>', unsafe_allow_html=True)
        with st.container(key="box_cm"):
            pct_view = st.toggle("Confusion matrix as row %", value=False)
    with ctl_low:
        st.markdown('<div class="lbl">Threshold highlights</div>', unsafe_allow_html=True)
        with st.container(key="box_low"):
            low_conf = st.toggle(f"Highlight low confidence (<{LOW_CONF:.0%})", value=False)

# ----------------------------------------------------------------------------- data
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

if data is not None:                         # data status chip, shown in the header of card 1
    status = f"Data: {source_label} · {len(data):,} rows" + (" · " + " ".join(notes) if notes else "")
    data_head.markdown(step_head(1, "Provide Test Data", status), unsafe_allow_html=True)
else:
    data_head.markdown(step_head(1, "Provide Test Data"), unsafe_allow_html=True)

model = load_model(model_name)
classes = list(model.classes_)

# ----------------------------------------------------------------------------- predictions + metrics
has_truth, unknown_labels, truth, scores = False, [], None, None
if data is not None:
    X_in = data[FEATURES]
    has_truth = TARGET in data.columns
    unknown_labels = sorted(set(data[TARGET].dropna()) - set(classes)) if has_truth else []
    if unknown_labels:
        has_truth = False                    # cannot score labels the models were never trained on
    pred = model.predict(X_in)
    proba = model.predict_proba(X_in)
    if has_truth:
        truth = data[TARGET]
        scores = metrics(truth, pred, proba, classes)

if scores is not None:                       # six metric cards, above the tabs
    st.markdown(metric_strip_html(scores), unsafe_allow_html=True)
    st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest (macro).")

# ----------------------------------------------------------------------------- tabs
tab_perf, tab_compare, tab_rows, tab_whatif = st.tabs(
    [":material/bar_chart: Performance", ":material/compare_arrows: Compare all models",
     ":material/table_rows: Predictions", ":material/tune: What-if student"])

export_df = None

if source_df is None:
    for tab in (tab_perf, tab_compare, tab_rows):
        with tab:
            st.info("Upload a CSV or tick “Use the bundled test_data.csv” above to begin.")
elif problems:
    for tab in (tab_perf, tab_compare, tab_rows):
        with tab:
            st.error("Cannot score this file:\n\n- " + "\n- ".join(problems))

if data is not None:
    # ------------------------------------------------------------------ tab 1: performance
    with tab_perf:
        if unknown_labels:
            st.error(f"{TARGET} contains label(s) the models were never trained on: {unknown_labels}. "
                     f"Expected: {TARGET_CLASS_ORDER}.")
        elif not has_truth:
            st.warning(f"No “{TARGET}” column in this file, so only predictions are available (see the Predictions tab).")
        else:
            order = [c for c in TARGET_CLASS_ORDER if c in classes]
            left, right = st.columns(2)
            with left:
                with st.container(key="card_cm"):
                    st.markdown(card_head("Confusion matrix", "Actual vs predicted classification breakdown",
                                          "Row %" if pct_view else "Counts"), unsafe_allow_html=True)
                    cm = confusion_matrix(truth, pred, labels=order)
                    row_share = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)      # 0-1 per actual class
                    labels_txt = [[f"{v * 100:.1f}%" if pct_view else f"{c:d}"
                                   for c, v in zip(cm_row, share_row)]
                                  for cm_row, share_row in zip(cm, row_share)]
                    st.markdown(confusion_html(labels_txt, order), unsafe_allow_html=True)
                    st.caption("Coloured diagonal cells are correct predictions; the rest are misclassifications.")
            with right:
                with st.container(key="card_report"):
                    st.markdown(card_head("Classification report", "Per-class precision, recall and F1"),
                                unsafe_allow_html=True)
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
            with st.container(key="card_compare"):
                st.markdown(card_head("Comparative classifier breakdown", "All five models on this data"),
                            unsafe_allow_html=True)
                board = score_all(X_in, truth)
                board.index.name = "ML Model Name"
                st.dataframe(board.round(4).style.highlight_max(axis=0, color="#e0e7ff").format("{:.4f}"),
                             width="stretch")
                pick = st.selectbox("Chart a metric", list(board.columns), index=4)
                fig2, ax2 = plt.subplots(figsize=(8, 3.4))
                best_model = board[pick].idxmax()
                bar_colors = [BRAND if m == best_model else "#c7d2fe" for m in board.index]
                ax2.bar(board.index, board[pick], color=bar_colors)
                for x, v in enumerate(board[pick]):
                    ax2.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#3730a3")
                ax2.set_ylabel(pick)
                ax2.set_xlabel("")
                ax2.set_ylim(max(0, float(board[pick].min()) - 0.05), 1.02)
                ax2.spines[["top", "right"]].set_visible(False)
                plt.setp(ax2.get_xticklabels(), rotation=15)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                st.caption(f"Best on this file by {pick}: **{board[pick].idxmax()}**")

    # ------------------------------------------------------------------ tab 3: predictions
    with tab_rows:
        out = data.copy()
        out["Predicted_Impact"] = pred
        for i, c in enumerate(classes):
            out[f"P({c})"] = proba[:, i].round(3)
        with st.container(key="card_pred"):
            st.markdown(card_head("Row-level predictions", "Predicted class and class probabilities for every row"),
                        unsafe_allow_html=True)
            f_search, f_wrong = st.columns([2, 1], vertical_alignment="center")
            query = f_search.text_input("Search records", placeholder="Search records…", label_visibility="collapsed")
            if has_truth:
                out["Correct"] = out[TARGET] == out["Predicted_Impact"]
                if f_wrong.checkbox("Show only misclassified rows"):
                    out = out[~out["Correct"]]
            if query:
                hit = out.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False, regex=False))
                out = out[hit.any(axis=1)]
            st.caption(f"{len(out):,} rows shown · model: {model_name}")
            if low_conf:
                conf = pd.Series(proba.max(axis=1), index=data.index)
                is_low = (conf.loc[out.index] < LOW_CONF).to_numpy()
                shown = out.style.apply(
                    lambda row: ["background-color: #fef3c7"] * len(row) if is_low[out.index.get_loc(row.name)]
                    else [""] * len(row), axis=1)
                st.dataframe(shown, width="stretch", height=380)
                st.caption(f"{int(is_low.sum()):,} row(s) with top-class probability below {LOW_CONF:.0%} highlighted.")
            else:
                st.dataframe(out, width="stretch", height=380)
        export_df = out

# ----------------------------------------------------------------------------- tab 4: what-if (independent of uploads)
with tab_whatif:
    col_in, col_out = st.columns([7, 5])
    with col_in:
        with st.container(key="card_sim_in"):
            st.markdown(card_head("Try a single student",
                                  f"Adjust the inputs; the selected model ({model_name}) predicts the overall impact."),
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            values = {}
            for i, col in enumerate(NUMERIC_FEATURES):
                target_col = (c1, c2)[i % 2]
                lo, hi = META["ranges"][col]                  # min/max seen in training
                mid = META["medians"][col]                    # starting value
                if col == "Late_Night_Usage":
                    values[col] = int(target_col.checkbox("Late-night usage", value=bool(round(mid))))
                elif col in ("Age", "Sleep_Quality_Score"):
                    values[col] = target_col.slider(col.replace("_", " "), int(lo), int(hi), int(mid))
                else:
                    start = min(max(round(mid, 1), lo), hi)   # rounding must not push the default outside the range
                    values[col] = target_col.slider(col.replace("_", " "), float(lo), float(hi), float(start), step=0.1)
            for i, col in enumerate(NOMINAL_FEATURES):
                values[col] = (c1, c2)[i % 2].selectbox(col.replace("_", " "), META["categories"][col])
            values[FREQ_COL] = c2.select_slider("Social comparison frequency", FREQ_LEVELS, value="Sometimes")

    one = pd.DataFrame([values])[FEATURES]
    probs = model.predict_proba(one)[0]
    top = classes[int(np.argmax(probs))]
    with col_out:
        with st.container(key="card_sim_out"):
            tone = CLASS_COLOR.get(top, BRAND)
            st.markdown(
                f'<div class="sim-result" style="background:{tone}1a;border:1px solid {tone}4d">'
                f'<div class="lbl">Predicted life impact</div>'
                f'<div class="sim-badge" style="background:{tone}">{CLASS_FACE.get(top, "")} {html.escape(str(top))}</div>'
                f'<div class="sim-conf-label">Confidence score</div>'
                f'<div class="sim-conf">{float(np.max(probs)):.1%}</div></div>',
                unsafe_allow_html=True)
            chart = pd.Series(probs, index=classes).reindex([c for c in TARGET_CLASS_ORDER if c in classes])
            st.bar_chart(chart, y_label="probability", color=BRAND, height=240)
    st.caption("Educational demo built on a survey dataset; not a diagnostic tool.")

# ----------------------------------------------------------------------------- export (bottom of the page)
with st.container(key="card_export"):
    ex_title, ex_csv, ex_json = st.columns([3, 1.3, 1.5], vertical_alignment="center")
    ex_title.markdown(
        '<div class="card-title">Export analysis results</div>'
        '<div class="card-sub">Download the classified rows and a summary of the selected model’s metrics</div>',
        unsafe_allow_html=True)
    ex_csv.download_button(":material/table_view: Predictions (CSV)",
                           export_df.to_csv(index=False).encode() if export_df is not None else b"",
                           file_name="predictions.csv", mime="text/csv", disabled=export_df is None,
                           width="stretch")
    summary = {
        "model": model_name,
        "data_source": source_label,
        "rows": int(len(data)) if data is not None else 0,
        "classes": [str(c) for c in classes],
        "metrics": ({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in scores.items()}
                    if scores is not None else None),
    }
    ex_json.download_button(":material/data_object: Model summary (JSON)", json.dumps(summary, indent=2).encode(),
                            file_name="model_summary.json", mime="application/json", disabled=data is None,
                            width="stretch")