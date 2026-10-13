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


def proportion_ci(k, n, level=0.95):
    """Exact (Clopper-Pearson) confidence interval for k successes out of n."""
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def live_summary(trials, label_names):
    """Metrics for logged live trials (spec LIVE-04).

    trials: dicts with target, predicted, status, latency_s and optionally signer/condition.
    Too-short recordings are not trials. A trial without a prediction (no hand detected) counts
    as wrong and is reported separately.
    """
    idx = {name: i for i, name in enumerate(label_names)}
    trials = [t for t in trials if t["status"] != "too_short"]
    y_true = np.array([idx[t["target"]] for t in trials])
    y_pred = np.array([idx.get(t["predicted"], -1) for t in trials])  # -1 = no prediction
    m = compute_metrics(y_true, y_pred, label_names)
    k, n = int((y_true == y_pred).sum()), len(trials)
    lat = np.array([float(t["latency_s"]) for t in trials if t.get("latency_s") not in ("", None)])
    out = dict(
        n_trials=n, n_correct=k, accuracy=m["accuracy"], accuracy_ci95=proportion_ci(k, n),
        macro_f1=m["macro_f1"], per_class=m["per_class"], confusion_matrix=m["confusion_matrix"],
        most_confused=most_confused(m["confusion_matrix"], label_names),
        no_prediction=sum(t["status"] != "ok" for t in trials),
        latency_s=dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)), n=len(lat)) if len(lat) else None,
    )
    for key in ("signer", "condition"):
        groups = sorted({t[key] for t in trials if key in t})
        if len(groups) > 1:
            out[f"accuracy_by_{key}"] = {g: float(np.mean([y_true[i] == y_pred[i] for i, t in enumerate(trials)
                                                           if t[key] == g])) for g in groups}
    return out
