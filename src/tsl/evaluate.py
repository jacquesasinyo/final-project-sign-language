"""Metrics (EVAL-02). Used for validation during development; the test split is
evaluated once per frozen configuration in Phase 4 (EVAL-03)."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, label_names):
    labels = list(range(len(label_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, name in enumerate(label_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def macro_f1(y_true, y_pred, n_classes):
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def most_confused(cm, label_names, k=5):
    cm = np.asarray(cm)
    pairs = [(int(cm[i, j]), label_names[i], label_names[j])
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    return sorted(pairs, reverse=True)[:k]
