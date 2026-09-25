"""Probability calibration of the matcher output, fit on out-of-fold predictions only.

Why: LightGBM scores are ranked well but are not probabilities. Two consumers need real
probabilities: (1) the expected-F0.5 decoder (decode.py) and (2) any threshold transfer to a
split whose positive/negative mix differs from training (the test split has ~2x the decoy
density). Only the OOF predictions (train.py) are used to fit anything here; the model
that produced a prediction never saw the record.

Methods
  platt    : p' = sigmoid(a * logit(p) + b), two parameters fit by Newton iterations on the
             log-loss (the standard Platt scaling, with logit(p) as the score).
  isotonic : monotone step function (PAV); more flexible, may overfit small folds.
  prior    : prior-shift correction of a calibrated probability when the positive rate moves
             from pi_train to pi_test:  logit(p') = logit(p) + log(odds_test / odds_train).
             It is applied on top of platt/isotonic at prediction time (see shift_odds).

Metrics: log loss, Brier score, expected calibration error (ECE, 15 equal-width bins) and
a reliability table. `nested_calibration_report` fits on K-1 folds and evaluates on the
held-out fold, so the reported numbers are themselves out-of-sample.
"""
import json

import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ Platt
def fit_platt(p, y, iters=50, l2=1e-4):
    """Newton's method on the 2-parameter logistic regression y ~ sigmoid(a*logit(p)+b)."""
    x = logit(p)
    y = np.asarray(y, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = sigmoid(a * x + b)
        w = q * (1 - q) + 1e-12
        g = np.array([np.sum((q - y) * x) + l2 * (a - 1), np.sum(q - y)])
        h = np.array([[np.sum(w * x * x) + l2, np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return {"method": "platt", "a": float(a), "b": float(b)}


# ------------------------------------------------------------------ isotonic (PAV)
def fit_isotonic(p, y, n_bins=2000):
    """Pool-adjacent-violators on score-quantile bins (bins keep it O(n) in memory)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p)
    p, y = p[order], y[order]
    edges = np.linspace(0, len(p), n_bins + 1).astype(int)
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            xs.append(p[a:b].mean())
            ys.append(y[a:b].mean())
            ws.append(b - a)
    # PAV
    vals, wts, lo = [], [], []
    for x, v, w in zip(xs, ys, ws):
        vals.append(v)
        wts.append(w)
        lo.append(x)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            vals[-2:] = [v2]
            wts[-2:] = [wts[-2] + wts[-1]]
            lo[-2:] = [lo[-2]]
    return {"method": "isotonic", "x": [float(v) for v in lo], "y": [float(v) for v in vals]}


def apply(cal, p):
    p = np.asarray(p, dtype=np.float64)
    if cal is None or cal.get("method") == "identity":
        return p
    if cal["method"] == "platt":
        return sigmoid(cal["a"] * logit(p) + cal["b"])
    if cal["method"] == "isotonic":
        x, y = np.asarray(cal["x"]), np.asarray(cal["y"])
        idx = np.clip(np.searchsorted(x, p, side="right") - 1, 0, len(y) - 1)
        return y[idx]
    raise ValueError(cal["method"])


def shift_odds(p, odds_ratio):
    """Prior-shift correction: multiply the odds by odds_ratio = (pi_te/(1-pi_te)) / (pi_tr/(1-pi_tr))."""
    return sigmoid(logit(p) + np.log(odds_ratio))


# ------------------------------------------------------------------ metrics
def calibration_metrics(p, y, n_bins=15):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    table, ece = [], 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            conf, acc, n = float(p[m].mean()), float(y[m].mean()), int(m.sum())
            ece += abs(conf - acc) * n / len(p)
            table.append({"bin": b, "n": n, "mean_pred": round(conf, 4), "frac_pos": round(acc, 4)})
    return {"log_loss": ll, "brier": brier, "ece": float(ece), "reliability": table}


def nested_calibration_report(p, y, folds, methods=("platt", "isotonic")):
    """Fit each method on K-1 folds of the OOF predictions, evaluate on the held-out fold."""
    out = {"uncalibrated": calibration_metrics(p, y)}
    for m in methods:
        q = np.empty_like(p, dtype=np.float64)
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            cal = fit_platt(p[tr], y[tr]) if m == "platt" else fit_isotonic(p[tr], y[tr])
            q[te] = apply(cal, p[te])
        out[m] = calibration_metrics(q, y)
    for k, v in out.items():
        v.pop("reliability", None) if k != "uncalibrated" and k != "platt" else None
    return out


def save(cal, path):
    with open(path, "w") as f:
        json.dump(cal, f)


def load(path):
    with open(path) as f:
        return json.load(f)
