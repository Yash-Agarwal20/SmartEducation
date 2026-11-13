"""Combined student risk score.

Each module yields a risk in [0, 1]:
  performance : expected backlog share = 1 - (subjects predicted to be cleared / subjects registered)
  dropout     : P(Dropout) from the dropout network
  attendance  : P(At-Risk) from the BiLSTM
  sentiment   : P(Negative) of the student's feedback (averaged if several comments)

The overall score is a TRANSPARENT weighted average (weights are a design choice, editable in
the dashboard), renormalised over the modules that are available. It is a decision-support
heuristic, not a fifth trained model.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def performance_risk(pred_cleared: float, registered: float) -> float:
    """Expected backlog share: the fraction of registered subjects the student is NOT expected to clear.
    0 = clears everything, 1 = clears nothing."""
    if registered <= 0:
        return 0.0
    return float(np.clip(1.0 - pred_cleared / registered, 0.0, 1.0))


def band(score: float) -> str:
    for upper, name in C.RISK_BANDS:
        if score < upper:
            return name
    return C.RISK_BANDS[-1][1]


def combine(components: dict, weights: dict | None = None) -> dict:
    """components: {name: risk in [0,1] or None}. Returns score, band and per-module contribution."""
    w = dict(C.DEFAULT_RISK_WEIGHTS if weights is None else weights)
    avail = {k: float(v) for k, v in components.items() if v is not None and k in w and w[k] > 0}
    if not avail:
        return {"score": None, "band": "n/a", "contributions": {}, "weights_used": {}}
    tot = sum(w[k] for k in avail)
    used = {k: w[k] / tot for k in avail}
    contrib = {k: used[k] * avail[k] for k in avail}
    score = float(sum(contrib.values()))
    return {"score": score, "band": band(score), "contributions": contrib,
            "weights_used": used, "module_risks": avail}


def recommendations(components: dict) -> list[str]:
    """Rule-based, human-readable suggestions for the advisor (based on the module risks)."""
    tips = []
    if (components.get("attendance") or 0) >= 0.5:
        tips.append("Attendance risk is high: contact the student early, check commute / work / health barriers.")
    if (components.get("performance") or 0) >= 0.5:
        tips.append("Several subjects are expected to remain uncleared: arrange tutoring or a study plan before exams.")
    if (components.get("dropout") or 0) >= 0.4:
        tips.append("Elevated dropout probability: schedule a mentoring conversation and review fees / financial aid.")
    if (components.get("sentiment") or 0) >= 0.5:
        tips.append("Feedback is negative: follow up on the concerns the student raised.")
    return tips or ["No module signals high risk. Keep monitoring."]
