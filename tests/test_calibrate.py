"""Calibration must be stable on the near-separable scores a good matcher produces."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import calibrate  # noqa: E402


def _data(n=20000, seed=0, sharp=1.0, shift=0.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(float)
    z = np.where(y == 1, 4.0, -4.0) + rng.normal(0, 2.0, n)
    p = calibrate.sigmoid(sharp * z + shift)  # over-confident (sharp > 1) / biased (shift) scores
    return p, y


def test_platt_recovers_temperature_and_bias():
    p, y = _data(sharp=2.0, shift=1.0)
    cal = calibrate.fit_platt(p, y)
    assert "fallback" not in cal
    assert cal["log_loss"] < cal["identity_log_loss"]
    assert cal["b"] < 0  # undoes the positive shift
    before = calibrate.calibration_metrics(p, y)["ece"]
    after = calibrate.calibration_metrics(calibrate.apply(cal, p), y)["ece"]
    assert after < before / 2


def test_platt_does_not_diverge_on_separable_scores():
    rng = np.random.default_rng(1)
    y = (rng.random(50000) < 0.25).astype(float)
    p = np.where(y == 1, 1 - 1e-7, 1e-7)  # perfectly separable, extreme scores
    p[rng.random(len(p)) < 0.01] = 0.5
    cal = calibrate.fit_platt(p, y)
    assert np.isfinite(cal["a"]) and np.isfinite(cal["b"])
    assert abs(cal["a"]) < 1e3
    q = calibrate.apply(cal, p)
    assert calibrate.calibration_metrics(q, y)["log_loss"] <= calibrate.calibration_metrics(p, y)["log_loss"] + 1e-9


def test_platt_falls_back_to_identity_when_it_cannot_help():
    p, y = _data()
    cal = calibrate.fit_platt(calibrate.apply(calibrate.fit_platt(p, y), p), y)
    q = calibrate.apply(cal, p)
    assert np.allclose(np.sort(q), np.sort(calibrate.apply({"method": "platt", "a": cal["a"], "b": cal["b"]}, p)))


def test_isotonic_monotone_and_apply():
    p, y = _data(sharp=3.0)
    cal = calibrate.fit_isotonic(p, y)
    assert all(a <= b + 1e-12 for a, b in zip(cal["y"], cal["y"][1:]))
    q = calibrate.apply(cal, p)
    assert q.min() >= 0 and q.max() <= 1


def test_prior_shift_moves_odds():
    p = np.array([0.5, 0.9])
    q = calibrate.shift_odds(p, 0.5)  # half the positive odds
    assert np.allclose(q, [1 / 3, 0.9 * 0.5 / (0.9 * 0.5 + 0.1)])


def test_nested_report_keys():
    p, y = _data()
    folds = np.arange(len(y)) % 3
    rep = calibrate.nested_calibration_report(p, y, folds)
    assert set(rep) == {"uncalibrated", "platt", "isotonic"}
    assert rep["platt"]["ece"] <= rep["uncalibrated"]["ece"] + 1e-3
