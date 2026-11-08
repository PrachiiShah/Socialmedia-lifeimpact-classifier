"""Shared evaluation metrics, used by train.py, evaluate.py and app.py so the numbers always agree."""
import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


def metrics(y_true, y_pred, y_proba, classes) -> dict:
    """The six evaluation metrics.

    Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest (macro).
    `classes` must be the model's `classes_`, i.e. the column order of `y_proba`.
    AUC is NaN when y_true does not contain every class (e.g. a small uploaded file).
    """
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
    except ValueError:
        pass
    return scores