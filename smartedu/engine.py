"""Prediction engine shared by every dashboard page (no Streamlit code in here, so it is testable).

A *profile* is a plain dict {raw column name: value} describing ONE student. The engine turns a
profile + a "stage" (how much we know) into predictions from the trained models.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import glossary as G
from .risk import performance_risk

BUNDLE_FILES = {
    "performance_start": "performance_start", "performance_sem1": "performance_sem1",
    "dropout_start": "dropout_start", "dropout_sem1": "dropout_sem1", "dropout_sem2": "dropout_sem2",
    "attendance": "attendance", "sentiment": "sentiment_bilstm",
}
# stage -> Module A / B model keys (Module A cannot forecast sem 2 once sem 2 results exist)
PERF_KEY = {"start": "performance_start", "sem1": "performance_sem1"}
DROP_KEY = {"start": "dropout_start", "sem1": "dropout_sem1", "sem2": "dropout_sem2"}


@lru_cache(maxsize=16)
def load_bundle(key: str):
    return joblib.load(C.MODELS_DIR / f"{BUNDLE_FILES[key]}.joblib")


def available(key: str) -> bool:
    return (C.MODELS_DIR / f"{BUNDLE_FILES[key]}.joblib").exists()


def default_profile() -> dict:
    """Typical student = training-set median / most common value, from the richest model."""
    b = load_bundle("dropout_sem2")
    out = {}
    for c in b.feature_cols:
        v = b.defaults[c]
        out[c] = int(float(v)) if c in b.cat_cols else (int(v) if b.ui_spec[c].get("int") else float(v))
    return out


def row_for(profile: dict, bundle) -> pd.DataFrame:
    return pd.DataFrame([{c: profile.get(c, bundle.defaults[c]) for c in bundle.feature_cols}])


def forecast_subjects(profile: dict, stage: str) -> dict | None:
    """Module A. Returns None when not applicable (semester-2 results are already known)."""
    key = PERF_KEY.get(stage)
    if key is None or not available(key):
        return None
    b = load_bundle(key)
    registered = float(profile.get(C.PERFORMANCE_REGISTERED, b.defaults[C.PERFORMANCE_REGISTERED]))
    cleared = float(np.clip(b.predict(row_for(profile, b))[0], 0, registered))
    err = b.meta.get("metrics", {}).get("rmse")
    return {"registered": registered, "cleared": cleared, "backlog": registered - cleared,
            "risk": performance_risk(cleared, registered), "typical_error": err, "bundle": key}


def dropout_outcome(profile: dict, stage: str) -> dict | None:
    key = DROP_KEY.get(stage)
    if key is None or not available(key):
        return None
    b = load_bundle(key)
    p = b.predict_proba(row_for(profile, b))[0]
    return {"probs": p, "risk": float(p[C.DROPOUT_LABELS.index("Dropout")]), "bundle": key,
            "most_likely": C.DROPOUT_LABELS[int(np.argmax(p))]}


def what_if(profile: dict, stage: str, changes: dict) -> dict:
    """Re-run both forecasts with some fields changed -> {'before': ..., 'after': ...}."""
    new = {**profile, **changes}
    return {"before": {"subjects": forecast_subjects(profile, stage), "dropout": dropout_outcome(profile, stage)},
            "after": {"subjects": forecast_subjects(new, stage), "dropout": dropout_outcome(new, stage)}}


def sweep(profile: dict, stage: str, field: str, values) -> pd.DataFrame:
    """Dropout probability and expected backlog as one field varies (for the interactive curves)."""
    rows = []
    for v in values:
        pr = {**profile, field: v}
        d, f = dropout_outcome(pr, stage), forecast_subjects(pr, stage)
        rows.append({"value": v, "dropout_risk": d["risk"] if d else np.nan,
                     "backlog": f["backlog"] if f else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ example students
PRESETS = {
    "Steady performer": {
        "Course": 9119, "Age at enrollment": 19, "Daytime/evening attendance": 1, "Displaced": 1,
        "Application order": 1, "Application mode": 1, "Previous qualification": 1,
        "Tuition fees up to date": 1, "Debtor": 0, "Scholarship holder": 1,
        C.SEM1_COLS[0]: 0, C.SEM1_COLS[1]: 6, C.SEM1_COLS[2]: 7, C.SEM1_COLS[3]: 6, C.SEM1_COLS[4]: 13.5, C.SEM1_COLS[5]: 0,
        C.SEM2_REGISTRATION_COLS[0]: 0, C.SEM2_REGISTRATION_COLS[1]: 6,
        C.SEM2_RESULT_COLS[0]: 7, C.SEM2_RESULT_COLS[1]: 6, C.SEM2_RESULT_COLS[2]: 13.8, C.SEM2_RESULT_COLS[3]: 0,
    },
    "Struggling after semester 1": {
        "Course": 9254, "Age at enrollment": 22, "Daytime/evening attendance": 1, "Displaced": 1,
        "Application order": 3, "Application mode": 17, "Previous qualification": 1,
        "Tuition fees up to date": 0, "Debtor": 1, "Scholarship holder": 0,
        C.SEM1_COLS[0]: 0, C.SEM1_COLS[1]: 6, C.SEM1_COLS[2]: 8, C.SEM1_COLS[3]: 2, C.SEM1_COLS[4]: 10.8, C.SEM1_COLS[5]: 2,
        C.SEM2_REGISTRATION_COLS[0]: 0, C.SEM2_REGISTRATION_COLS[1]: 6,
        C.SEM2_RESULT_COLS[0]: 4, C.SEM2_RESULT_COLS[1]: 1, C.SEM2_RESULT_COLS[2]: 10.0, C.SEM2_RESULT_COLS[3]: 3,
    },
    "Working mature student (evening)": {
        "Course": 9991, "Age at enrollment": 31, "Daytime/evening attendance": 0, "Displaced": 0,
        "Application order": 1, "Application mode": 39, "Previous qualification": 1,
        "Tuition fees up to date": 1, "Debtor": 0, "Scholarship holder": 0,
        C.SEM1_COLS[0]: 0, C.SEM1_COLS[1]: 5, C.SEM1_COLS[2]: 5, C.SEM1_COLS[3]: 3, C.SEM1_COLS[4]: 12.0, C.SEM1_COLS[5]: 1,
        C.SEM2_REGISTRATION_COLS[0]: 0, C.SEM2_REGISTRATION_COLS[1]: 5,
        C.SEM2_RESULT_COLS[0]: 5, C.SEM2_RESULT_COLS[1]: 3, C.SEM2_RESULT_COLS[2]: 11.8, C.SEM2_RESULT_COLS[3]: 1,
    },
    "Good marks but fee problems": {
        "Course": 9500, "Age at enrollment": 20, "Daytime/evening attendance": 1, "Displaced": 1,
        "Application order": 1, "Application mode": 1, "Previous qualification": 1,
        "Tuition fees up to date": 0, "Debtor": 1, "Scholarship holder": 0,
        C.SEM1_COLS[0]: 0, C.SEM1_COLS[1]: 6, C.SEM1_COLS[2]: 7, C.SEM1_COLS[3]: 5, C.SEM1_COLS[4]: 12.5, C.SEM1_COLS[5]: 0,
        C.SEM2_REGISTRATION_COLS[0]: 0, C.SEM2_REGISTRATION_COLS[1]: 6,
        C.SEM2_RESULT_COLS[0]: 6, C.SEM2_RESULT_COLS[1]: 4, C.SEM2_RESULT_COLS[2]: 12.2, C.SEM2_RESULT_COLS[3]: 0,
    },
}


# ------------------------------------------------------------------ looking inside the network
def architecture(bundle) -> dict:
    """Layer sizes and parameter count of a tabular MLP bundle (for the 'inside the network' page)."""
    m = bundle.model
    out = m.n_classes if m.task == "classification" else 1
    sizes = [m.in_dim_, *m.hidden, out]
    params = sum(p.numel() for p in m.model.parameters())
    return {"sizes": sizes, "params": int(params), "dropout": m.dropout, "task": m.task}


def trace_mlp(bundle, profile: dict) -> dict:
    """Run ONE student through the MLP and keep the numbers after every hidden block."""
    import torch
    x = bundle.transform(row_for(profile, bundle))
    names = list(bundle.prep.get_feature_names_out())
    net = bundle.model.model.net.eval()
    h = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    blocks = []
    with torch.no_grad():
        for layer in net:
            h = layer(h)
            if isinstance(layer, torch.nn.ReLU):
                blocks.append(h[0].numpy().copy())
        logits = h[0].numpy().copy()
    if bundle.task == "classification":
        e = np.exp(logits - logits.max())
        result = e / e.sum()
    else:
        result = logits * bundle.model.y_std_ + bundle.model.y_mean_
    return {"names": names, "inputs": np.asarray(x)[0], "blocks": blocks, "logits": logits, "result": result}


# ------------------------------------------------------------------ real cohort for comparisons
@lru_cache(maxsize=1)
def cohort() -> pd.DataFrame:
    from .data_loaders import load_university
    try:
        return load_university("real")
    except FileNotFoundError:
        return load_university("synthetic")


@lru_cache(maxsize=8)
def cohort_dropout_risk(stage: str) -> np.ndarray:
    """P(Dropout) for every student in the real cohort (what 'typical' looks like at this stage)."""
    b = load_bundle(DROP_KEY[stage])
    df = cohort()
    X = df[[c for c in b.feature_cols if c in df.columns]]
    return b.predict_proba(X)[:, C.DROPOUT_LABELS.index("Dropout")]


@lru_cache(maxsize=1)
def test_cohort() -> pd.DataFrame:
    """The 664 real students the dropout models NEVER saw during training (same split as training).
    Used for the class view so that nothing shown is an in-sample flattery."""
    from .utils import split_indices
    df = cohort()
    y = df[C.DROPOUT_TARGET].map({l: i for i, l in enumerate(C.DROPOUT_LABELS)}).to_numpy()
    _, _, ite = split_indices(len(df), y)
    return df.iloc[ite].reset_index(drop=True)


def predict_students(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Predictions for MANY students (one row each, raw column names). Missing columns take typical values.
    Returns: P(drops out / still studying / graduates), expected subjects cleared + backlog (when the stage
    allows), and the combined risk (weighted like the risk card)."""
    from .risk import combine
    bd = load_bundle(DROP_KEY[stage])
    p = bd.predict_proba(df)
    out = pd.DataFrame({"p_drop": p[:, 0], "p_enrolled": p[:, 1], "p_graduate": p[:, 2]}, index=df.index)
    key = PERF_KEY.get(stage)
    if key and available(key):
        bp = load_bundle(key)
        reg = pd.to_numeric(df[C.PERFORMANCE_REGISTERED], errors="coerce") if C.PERFORMANCE_REGISTERED in df \
            else pd.Series(bp.defaults[C.PERFORMANCE_REGISTERED], index=df.index)
        reg = reg.fillna(bp.defaults[C.PERFORMANCE_REGISTERED]).to_numpy(float)
        cleared = np.clip(bp.predict(df), 0, reg)
        out["registered"], out["cleared"], out["backlog"] = reg, cleared, reg - cleared
        out["backlog_share"] = np.where(reg > 0, 1 - cleared / np.maximum(reg, 1e-9), 0.0)
    risks = []
    for i in range(len(out)):
        comp = {"dropout": float(out["p_drop"].iloc[i])}
        if "backlog_share" in out:
            comp["performance"] = float(out["backlog_share"].iloc[i])
        risks.append(combine(comp)["score"])
    out["overall_risk"] = risks
    return out
