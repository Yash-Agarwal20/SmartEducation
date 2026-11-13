"""Loaders for REAL datasets (data/raw/) with synthetic fall-backs.

Every loader takes ``source='real'|'synthetic'``. With 'real' and a missing file
you get an error that tells you exactly what to download and where to put it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import synthetic as S

HELP = {
    "university": (
        "UCI 'Predict Students' Dropout and Academic Success' (Realinho et al., 2022)\n"
        "  https://archive.ics.uci.edu/dataset/697\n"
        "  Save the CSV as data/raw/dropout.csv (the original file is called data.csv)\n"
        "  or run:  python download_data.py"),
    "feedback": (
        "A CSV with a text column and a sentiment/rating column, e.g. a Kaggle\n"
        "  'Coursera course reviews' or 'student feedback' dataset.\n"
        "  Save it as data/raw/feedback.csv (columns auto-detected: text/review/comment/...\n"
        "  and label/sentiment/rating/score)  or run:  python download_data.py"),
    "oulad": (
        "OULAD (Kuzilek et al., 2017) https://analyse.kmi.open.ac.uk/open_dataset\n"
        "  Put studentInfo.csv and studentVle.csv into data/raw/oulad/"),
}


def _missing(kind: str, path: Path):
    raise FileNotFoundError(f"\nMissing {path}\n\nHow to get it:\n{HELP[kind]}\n"
                            f"(Or run with --source synthetic to test the pipeline.)")


# ------------------------------------------------- A + B. university student records
def load_university(source="real") -> pd.DataFrame:
    """Real university records (4,424 students) or a schema-identical synthetic stand-in.

    Codes are normalised to the official ones (see glossary.normalize_university) so that
    every later step can show real course / admission-route names.
    """
    from .glossary import normalize_university
    if source == "synthetic":
        return normalize_university(S.make_dropout())
    for fn in ("dropout.csv", "data.csv"):
        p = C.RAW_DIR / fn
        if p.exists():
            df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")   # ';' or ',' + BOM
            df.columns = [c.strip().replace("\t", "") for c in df.columns]
            df[C.DROPOUT_TARGET] = df[C.DROPOUT_TARGET].astype(str).str.strip()
            bad = set(df[C.DROPOUT_TARGET]) - set(C.DROPOUT_LABELS)
            if bad:
                raise ValueError(f"Unexpected values in '{C.DROPOUT_TARGET}': {bad}")
            return normalize_university(df)
    _missing("university", C.RAW_DIR / "dropout.csv")


def load_dropout(source="real") -> pd.DataFrame:
    return load_university(source)


def load_performance(source="real") -> pd.DataFrame:
    """Students who registered for at least one subject in semester 2 (the target needs that)."""
    df = load_university(source)
    return df[df[C.PERFORMANCE_REGISTERED] > 0].reset_index(drop=True)


# ------------------------------------------------------------- C. attendance
def load_attendance(source="synthetic", n_synth=3000):
    """Returns (static_df, seq[n, T, channels], y[n], info_dict).

    'synthetic' -> simulated attendance (documented rules, see synthetic.py)
    'oulad'     -> real engagement data from OULAD (VLE clicks as attendance proxy)
    """
    if source in ("synthetic", "real"):
        if source == "real":
            print("[attendance] No public attendance dataset exists - using the documented "
                  "simulation. Use --attendance-source oulad for real engagement data.")
        static, seq, _full, y = S.make_attendance(n_synth)
        return static, seq, y, {"source": "simulated", "channels": S.SEQ_CHANNELS,
                                "label": f"final attendance < {C.ATTENDANCE_THRESHOLD:.0f}%"}
    if source == "oulad":
        return _load_oulad()
    raise ValueError(source)


def _load_oulad(weeks: int = C.OBSERVED_WEEKS):
    """OULAD adapter: weekly VLE clicks of the first ``weeks`` weeks -> At-Risk
    (final_result in {Withdrawn, Fail}). Engagement proxy, NOT physical attendance."""
    d = C.RAW_DIR / "oulad"
    for fn in ("studentInfo.csv", "studentVle.csv"):
        if not (d / fn).exists():
            _missing("oulad", d / fn)
    info = pd.read_csv(d / "studentInfo.csv")
    vle = pd.read_csv(d / "studentVle.csv", usecols=["code_module", "code_presentation",
                                                      "id_student", "date", "sum_click"])
    key = ["code_module", "code_presentation", "id_student"]
    vle = vle[(vle["date"] >= 0) & (vle["date"] < 7 * weeks)].copy()
    vle["week"] = (vle["date"] // 7).astype(int)
    # per-week total clicks and number of active days
    wk = vle.groupby(key + ["week"]).agg(clicks=("sum_click", "sum"),
                                         active_days=("date", "nunique")).reset_index()
    info = info.drop_duplicates(key).reset_index(drop=True)
    idx = {tuple(r): i for i, r in enumerate(info[key].itertuples(index=False, name=None))}
    seq = np.zeros((len(info), weeks, 2), dtype=float)
    for r in wk.itertuples(index=False):
        i = idx.get((r.code_module, r.code_presentation, r.id_student))
        if i is not None:
            seq[i, r.week, 0] = np.log1p(r.clicks)
            seq[i, r.week, 1] = r.active_days
    keep = info["final_result"].isin(["Pass", "Distinction", "Fail", "Withdrawn"]).to_numpy()
    y = info["final_result"].isin(["Fail", "Withdrawn"]).astype(int).to_numpy()
    static = info[["highest_education", "imd_band", "age_band",
                   "num_of_prev_attempts", "studied_credits", "disability"]].copy()
    static["imd_band"] = static["imd_band"].fillna("unknown")
    return (static[keep].reset_index(drop=True), seq[keep], y[keep],
            {"source": "OULAD", "channels": ["log_clicks", "active_days"],
             "label": "final_result in {Fail, Withdrawn}"})


# -------------------------------------------------------------- D. feedback
_TEXT_CANDIDATES = ["text", "review", "reviews", "comment", "comments", "feedback", "content", "sentence"]
_LABEL_CANDIDATES = ["label", "sentiment", "rating", "score", "stars", "polarity"]


def load_feedback(source="real", n_synth=3000, text_col=None, label_col=None) -> pd.DataFrame:
    """Returns a DataFrame with columns text, label (Negative/Neutral/Positive)."""
    if source == "synthetic":
        return S.make_feedback(n_synth)
    p = C.RAW_DIR / "feedback.csv"
    if not p.exists():
        _missing("feedback", p)
    raw = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
    low = {c.lower().strip(): c for c in raw.columns}
    tc = text_col or next((low[c] for c in _TEXT_CANDIDATES if c in low), None)
    lc = label_col or next((low[c] for c in _LABEL_CANDIDATES if c in low), None)
    if tc is None or lc is None:
        raise ValueError(f"Could not detect text/label columns in {list(raw.columns)}. "
                         f"Pass --text-col and --label-col.")
    df = pd.DataFrame({"text": raw[tc].astype(str), "label": raw[lc]}).dropna()
    df["label"] = _normalise_sentiment(df["label"])
    return df.dropna().reset_index(drop=True)


def _normalise_sentiment(s: pd.Series) -> pd.Series:
    """Maps ratings 1-5 (1-2 neg, 3 neutral, 4-5 pos) or text labels to Negative/Neutral/Positive."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        mx = num.max()
        if mx <= 1.0:                                    # 0/1 polarity
            return num.map({0.0: "Negative", 1.0: "Positive"})
        return pd.cut(num, bins=[-np.inf, 2, 3, np.inf], labels=C.SENTIMENT_LABELS).astype(object)
    m = {"negative": "Negative", "neg": "Negative", "neutral": "Neutral", "neu": "Neutral",
         "positive": "Positive", "pos": "Positive"}
    return s.astype(str).str.strip().str.lower().map(m)
