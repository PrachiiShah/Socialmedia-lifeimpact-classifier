import matplotlib
matplotlib.use("Agg")                  # must come before pyplot is imported
import matplotlib.pyplot as plt
import seaborn as sns
from model.meta_data import FIG_DIR

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

def save_fig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)     # add this line
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close("all")