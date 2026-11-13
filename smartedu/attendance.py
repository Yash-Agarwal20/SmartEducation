"""Module C - Attendance risk prediction (early warning).

Input  : the first ``OBSERVED_WEEKS`` weeks of (attendance %, assignment rate, LMS logins)
         + static student features.
Target : At-Risk = final attendance below 75% (simulated data) or Fail/Withdrawn (OULAD).
Models : Logistic Regression / Random Forest / XGBoost on engineered aggregates (baselines),
         MLP on the flattened sequence, and a BiLSTM over the weekly sequence (deep model).
The decision threshold is tuned on the validation set for F2 (recall-weighted), because
missing an at-risk student is worse than a false alarm.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as C
from . import glossary as G
from .data_loaders import load_attendance
from .nn import TorchSeqRisk, TorchTabular
from .tabular import (build_preprocessor, make_defaults, make_ui_spec, prepare_frame,
                      split_columns)
from .utils import (best_threshold, clf_metrics, plot_confusion, plot_history,
                    plot_importance, save_json, set_seed, split_indices)

LABELS = C.ATTENDANCE_LABELS


# --------------------------------------------------------------- features
def engineered_features(seq: np.ndarray) -> pd.DataFrame:
    """Aggregates of the weekly sequence for the tree / linear baselines."""
    n, T, ch = seq.shape
    a = seq[:, :, 0]
    x = np.arange(T) - (T - 1) / 2.0
    slope = ((a - a.mean(1, keepdims=True)) * x).sum(1) / (x ** 2).sum()
    f = {
        "att_mean": a.mean(1), "att_min": a.min(1), "att_last": a[:, -1],
        "att_last3": a[:, -3:].mean(1), "att_slope": slope,
        "att_weeks_below_75": (a < C.ATTENDANCE_THRESHOLD).sum(1),
    }
    for c in range(1, ch):
        f[f"ch{c}_mean"] = seq[:, :, c].mean(1)
        f[f"ch{c}_last3"] = seq[:, -3:, c].mean(1)
    return pd.DataFrame(f)


# ----------------------------------------------------------------- bundle
class AttendanceBundle:
    """Static-feature preprocessor + BiLSTM + tuned threshold, with a raw-input API."""

    def __init__(self, prep, model, threshold, channels, seq_len, static_cols, num_cols,
                 cat_cols, defaults, ui_spec, seq_means, meta=None):
        self.prep, self.model, self.threshold = prep, model, threshold
        self.channels, self.seq_len = list(channels), seq_len
        self.static_cols, self.num_cols, self.cat_cols = list(static_cols), list(num_cols), list(cat_cols)
        self.defaults, self.ui_spec, self.seq_means = defaults, ui_spec, seq_means
        self.meta = meta or {}
        self.labels = LABELS

    def static_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.static_cols:
            if c not in df.columns:
                df[c] = self.defaults[c]
        out = prepare_frame(df, self.static_cols, self.cat_cols)
        for c in self.num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def predict_proba(self, seq, static_df) -> np.ndarray:
        """P(At-Risk) per student."""
        st = self.prep.transform(self.static_frame(static_df))
        return self.model.predict_proba(np.asarray(seq, float), st)[:, 1]

    def predict(self, seq, static_df) -> np.ndarray:
        return (self.predict_proba(seq, static_df) >= self.threshold).astype(int)

    def occlusion(self, seq1: np.ndarray, static1: pd.DataFrame) -> pd.DataFrame:
        """Per-input sensitivity for ONE student: change in P(At-Risk) when a value is
        replaced by the training average. Positive = this value pushes risk UP."""
        base = float(self.predict_proba(seq1, static1)[0])
        rows = []
        for c, ch in enumerate(self.channels):          # weekly signals
            for w in range(self.seq_len):
                s2 = np.array(seq1, float, copy=True)
                s2[0, w, c] = self.seq_means[w, c]
                rows.append((f"{ch} (week {w + 1})", base - float(self.predict_proba(s2, static1)[0])))
        for col in self.static_cols:                    # static features
            d2 = static1.copy()
            d2[col] = self.defaults[col]
            rows.append((col, base - float(self.predict_proba(seq1, d2)[0])))
        return pd.DataFrame(rows, columns=["feature", "effect"])

    def occlusion_groups(self, seq1: np.ndarray, static1: pd.DataFrame) -> pd.DataFrame:
        """Like ``occlusion`` but removes a WHOLE weekly signal at once (all 8 weeks of attendance, of
        assignments, of logins) plus each static fact - a clearer answer to 'what drives this risk?'."""
        base = float(self.predict_proba(seq1, static1)[0])
        rows = []
        for c, ch in enumerate(self.channels):
            s2 = np.array(seq1, float, copy=True)
            s2[0, :, c] = self.seq_means[:, c]
            rows.append((f"{ch} (all weeks)", base - float(self.predict_proba(s2, static1)[0])))
        for col in self.static_cols:
            d2 = static1.copy()
            d2[col] = self.defaults[col]
            rows.append((col, base - float(self.predict_proba(seq1, d2)[0])))
        return pd.DataFrame(rows, columns=["feature", "effect"])


# ------------------------------------------------------------------- run
def _row(m):
    pc = m["per_class"]["At-Risk"]
    return {"accuracy": m["accuracy"], "precision": pc["precision"], "recall": pc["recall"],
            "f1": pc["f1"], "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc")}


def run(source="synthetic", quick=False, shap_rows=0, verbose=True) -> dict:
    set_seed()
    static, seq, y, info = load_attendance(source)
    n, T, ch = seq.shape
    itr, iva, ite = split_indices(n, y)
    num, cat = split_columns(static)
    prep = build_preprocessor(num, cat, min_frequency=10).fit(prepare_frame(static.iloc[itr], static.columns, cat))
    st = lambda i: prep.transform(prepare_frame(static.iloc[i], static.columns, cat))  # noqa: E731
    Str, Sva, Ste = st(itr), st(iva), st(ite)
    if verbose:
        print(f"  [Attendance] source={info['source']}  n={n}  seq={seq.shape}  "
              f"at-risk rate={y.mean():.2%}  label: {info['label']}")

    eng = lambda i: np.hstack([engineered_features(seq[i]).to_numpy(), st(i)])  # noqa: E731
    flat = lambda i: np.hstack([seq[i].reshape(len(i), -1), st(i)])  # noqa: E731
    # standardise flattened inputs with TRAIN statistics (MLP only)
    f_tr = flat(itr)
    f_mu, f_sd = f_tr.mean(0), f_tr.std(0) + 1e-6
    flat_s = lambda i: (flat(i) - f_mu) / f_sd  # noqa: E731

    ytr, yva, yte = y[itr], y[iva], y[ite]
    table, probs = {}, {}
    sw = compute_sample_weight("balanced", ytr)

    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=150 if quick else 400, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=C.SEED),
        "XGBoost": XGBClassifier(
            n_estimators=150 if quick else 400, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=C.SEED, n_jobs=4, verbosity=0),
    }
    models_val = {}
    for name, m in baselines.items():
        eng_tr = eng(itr)
        mu, sd = eng_tr.mean(0), eng_tr.std(0) + 1e-6
        scale = (lambda a, mu=mu, sd=sd: (a - mu) / sd) if name == "Logistic Regression" else (lambda a: a)
        if name == "XGBoost":
            m.fit(scale(eng_tr), ytr, sample_weight=sw)
        else:
            m.fit(scale(eng_tr), ytr)
        models_val[name] = (m.predict_proba(scale(eng(iva)))[:, 1], m.predict_proba(scale(eng(ite)))[:, 1])

    kw = dict(max_epochs=60 if quick else 200, patience=10 if quick else 20)
    t0 = time.time()
    mlp = TorchTabular("classification", n_classes=2, hidden=(128, 64), dropout=0.3, lr=1e-3,
                       weight_decay=1e-3, batch_size=64, **kw).fit(flat_s(itr), ytr, flat_s(iva), yva)
    models_val["MLP (flattened)"] = (mlp.predict_proba(flat_s(iva))[:, 1], mlp.predict_proba(flat_s(ite))[:, 1])
    t_mlp = time.time() - t0
    t0 = time.time()
    lstm = TorchSeqRisk(hidden=48, dropout=0.3, lr=2e-3, batch_size=64,
                        max_epochs=50 if quick else 150, patience=8 if quick else 15)
    lstm.fit(seq[itr], Str, ytr, seq[iva], Sva, yva)
    models_val["BiLSTM + static (deep learning)"] = (
        lstm.predict_proba(seq[iva], Sva)[:, 1], lstm.predict_proba(seq[ite], Ste)[:, 1])
    t_lstm = time.time() - t0

    thresholds = {}
    for name, (pv, pt) in models_val.items():
        thr = best_threshold(yva, pv, beta=2.0)
        thresholds[name] = thr
        met = clf_metrics(yte, (pt >= thr).astype(int), np.stack([1 - pt, pt], 1), LABELS)
        table[name] = {**_row(met), "threshold": thr}
        probs[name] = pt
        if verbose:
            r = table[name]
            print(f"    {name:32s} recall {r['recall']:.3f}  precision {r['precision']:.3f}  "
                  f"F1 {r['f1']:.3f}  AUC {r['roc_auc']:.3f}  (thr {thr:.2f})")
    table["MLP (flattened)"]["train_seconds"] = round(t_mlp, 1)
    table["BiLSTM + static (deep learning)"]["train_seconds"] = round(t_lstm, 1)
    table["BiLSTM + static (deep learning)"]["best_epoch"] = lstm.history_["best_epoch"]

    deep = "BiLSTM + static (deep learning)"
    thr = thresholds[deep]
    plot_history(lstm.history_, C.FIG_DIR / "attendance_training.png", "BiLSTM training")
    plot_confusion(yte, (probs[deep] >= thr).astype(int), LABELS,
                   C.FIG_DIR / "attendance_confusion_bilstm.png", "Attendance BiLSTM - test set")

    seq_means = seq[itr].mean(0)                       # (T, channels)
    bundle = AttendanceBundle(
        prep, lstm, thr, info["channels"], T, list(static.columns), num, cat,
        make_defaults(static.iloc[itr], num, cat), make_ui_spec(static.iloc[itr], num, cat),
        seq_means, meta={"metrics": table[deep], "source": info["source"], "label": info["label"]})

    # global occlusion sensitivity (mean |effect| per input group on a test sample)
    rng = np.random.default_rng(C.SEED)
    sample = rng.choice(ite, size=min(40 if quick else 120, len(ite)), replace=False)
    acc: dict[str, list] = {}
    for i in sample:
        occ = bundle.occlusion(seq[i:i + 1], static.iloc[i:i + 1])
        occ["group"] = occ["feature"].str.replace(r" \(week \d+\)", "", regex=True)
        for g, v in occ.groupby("group")["effect"].apply(lambda s: float(np.abs(s).sum())).items():
            acc.setdefault(g, []).append(v)
    imp = (pd.DataFrame({"feature": list(acc), "importance": [np.mean(v) for v in acc.values()]})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    plot_importance(imp.assign(feature=imp["feature"].map(G.label)), C.FIG_DIR / "attendance_occlusion_importance.png",
                    "Occlusion sensitivity (sum of |change in P(At-Risk)|)",
                    xlabel="mean summed |effect|")

    joblib.dump(bundle, C.MODELS_DIR / "attendance.joblib")
    out = {"module": "attendance", "source": info["source"], "label_definition": info["label"],
           "n_rows": n, "at_risk_rate": float(y.mean()), "observed_weeks": T,
           "n_train": len(itr), "n_val": len(iva), "n_test": len(ite),
           "test": table, "importance_top10": imp.head(10).to_dict("records")}
    save_json(out, C.REPORTS_DIR / "metrics_attendance.json")
    return out


# ---------------------------------------------- wide <-> (static, sequence) helpers
def to_wide(static: pd.DataFrame, seq: np.ndarray, channels) -> pd.DataFrame:
    """Flat one-row-per-student table: static columns + '<channel>_w<k>' columns (for CSV I/O)."""
    cols = {f"{ch}_w{w + 1}": seq[:, w, c] for c, ch in enumerate(channels) for w in range(seq.shape[1])}
    return pd.concat([static.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


def from_wide(df: pd.DataFrame, bundle: AttendanceBundle):
    """Inverse of ``to_wide`` using a trained bundle's channel names / window length.
    Returns (static_df, seq) or raises ValueError listing the missing columns."""
    need = [f"{ch}_w{w + 1}" for ch in bundle.channels for w in range(bundle.seq_len)]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing {len(miss)} weekly column(s), e.g. {miss[:4]}. "
                         f"Expected '<channel>_w<k>' for channels {bundle.channels}, weeks 1..{bundle.seq_len}.")
    seq = np.stack([df[[f"{ch}_w{w + 1}" for w in range(bundle.seq_len)]].to_numpy(float)
                    for ch in bundle.channels], axis=2)
    static = df[[c for c in bundle.static_cols if c in df.columns]].copy()
    return static, seq
