"""The decision policy of the matcher: one global acceptance threshold on the calibrated probability,
plus the decision-rule extras shared by every link (margin over the runner-up candidate and the
contradiction penalty, see thresholds.accept_mask / model.assign).

    policy = ThresholdPolicy(0.76, margin=0.0, contra_penalty=0.0)
    policy.save(path); policy = ThresholdPolicy.load(path)
    links = assign_with_policy(meta, p, policy, contra)

Per-class thresholds (one threshold per country / source / script ...) were evaluated in
docs/REPORT.md section 13 and never beat the global threshold outside the fold-to-fold noise, so
the machinery was removed (v5.1); the global, decoy-density-adjusted threshold of
select_threshold.py is the policy every run ships with.
"""
import json
import os

import numpy as np

POLICY_VERSION = 3  # 3: global threshold + decision rule only


class ThresholdPolicy:
    def __init__(self, default, scale="calibrated", name=None, fit=None, margin=0.0, contra_penalty=0.0, **_ignored):
        self.default = float(default)
        self.scale = scale
        self.name = name or "global"
        self.fit = fit or {}
        self.margin = float(margin or 0.0)
        self.contra_penalty = float(contra_penalty or 0.0)

    @property
    def threshold(self):
        return self.default

    @property
    def rule(self):
        return {"margin": self.margin, "contra_penalty": self.contra_penalty}

    def to_dict(self):
        return {"version": POLICY_VERSION, "name": self.name, "scheme": "global", "scale": self.scale, "default": self.default,
                "margin": self.margin, "contra_penalty": self.contra_penalty, "fit": self.fit}

    @classmethod
    def from_dict(cls, d):
        return cls(d["default"], d.get("scale", "calibrated"), d.get("name"), d.get("fit"), d.get("margin", 0.0), d.get("contra_penalty", 0.0))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def describe(self):
        extra = f" (margin {self.margin:.2f}, contradiction penalty {self.contra_penalty:.2f})" if (self.margin or self.contra_penalty) else ""
        return f"{self.name}: global threshold {self.default:.3f}{extra}"


def assign_with_policy(meta, p, policy, contra=None):
    """Links (t_rid, s_rid, p) accepted under a policy (the same rule as model.assign)."""
    from model import assign  # local import: model.py imports lightgbm, which the tools do not need
    return assign(meta, np.asarray(p), policy.default, policy.margin, policy.contra_penalty, contra)
