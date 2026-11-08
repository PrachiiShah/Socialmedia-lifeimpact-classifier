from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]          # repo root (this file lives in model/)
DATA_FILE  = PROJECT / "data" / "Social_media_impact_on_life.csv"
SAMPLE_CSV = PROJECT / "test_data.csv"
MODEL_DIR  = PROJECT / "model"
FIG_DIR    = PROJECT / "figures"
META_FILE  = MODEL_DIR / "run_meta.json"

SEED = 42
TEST_FRACTION = 0.20
TARGET = "Overall_Impact"
ID_COL = "Student_ID"
TARGET_CLASS_ORDER = ["Negative", "Neutral", "Beneficial"]
FREQ_COL = "Social_Comparison_Frequency"
FREQ_LEVELS = ["Never", "Rarely", "Sometimes", "Frequently", "Always"]

def ensure_dirs():                                     # call from eda.py / train.py only, never from the app
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def model_file(name: str) -> str:
    return f"{name.lower().replace(' ', '_')}.joblib"

def column_groups(raw):                                # one definition, used by eda.py and train.py
    numeric = [c for c in raw.select_dtypes(include=["number", "bool"]).columns if c not in (TARGET, ID_COL)]
    nominal = [c for c in raw.columns if c not in numeric + [TARGET, ID_COL, FREQ_COL]]
    return numeric, nominal