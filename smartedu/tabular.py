"""Tabular pipeline: preprocessing, the deployable ``TabularBundle`` and explainability.

A bundle = fitted preprocessor + trained network + everything the dashboard needs
(feature lists, default values, UI ranges, a small background sample for SHAP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


# --------------------------------------------------------------- preprocessing
def _to_cat_str(s: pd.Series) -> pd.Series:
    """Integer-coded categories must map to identical strings ('9003' not '9003.0')."""
    if pd.api.types.is_numeric_dtype(s):
        return s.round().astype("Int64").astype(str)
    return s.astype(str)


def split_columns(df: pd.DataFrame, categorical_hint=()):
    """Categorical = explicitly hinted, or anything that is not a number (pandas 3 'str' dtype incl.)."""
    cat = [c for c in df.columns
           if c in categorical_hint or pd.api.types.is_bool_dtype(df[c])
           or not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in df.columns if c not in cat]
    return num, cat


def build_preprocessor(num_cols, cat_cols, min_frequency=10) -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                             sparse_output=False)),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, list(num_cols)))
    if cat_cols:
        transformers.append(("cat", cat_pipe, list(cat_cols)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def prepare_frame(df: pd.DataFrame, cols, cat_cols) -> pd.DataFrame:
    out = df[list(cols)].copy()
    for c in cat_cols:
        out[c] = _to_cat_str(out[c])
    return out


def make_ui_spec(df: pd.DataFrame, num_cols, cat_cols) -> dict:
    spec = {}
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        is_int = bool((s == s.round()).all())
        spec[c] = {"type": "num", "min": float(s.min()), "max": float(s.max()),
                   "default": float(s.median()), "int": is_int}
    for c in cat_cols:
        s = _to_cat_str(df[c])
        spec[c] = {"type": "cat", "options": sorted(s.unique().tolist()),
                   "default": str(s.mode().iloc[0])}
    return spec


# ---------------------------------------------------------------- the bundle
class TabularBundle:
    """Preprocessor + trained model with a raw-DataFrame predict API."""

    def __init__(self, prep, model, feature_cols, num_cols, cat_cols, task, labels,
                 defaults, ui_spec, background, meta=None):
        self.prep, self.model = prep, model
        self.feature_cols, self.num_cols, self.cat_cols = list(feature_cols), list(num_cols), list(cat_cols)
        self.task, self.labels = task, labels
        self.defaults, self.ui_spec, self.background = defaults, ui_spec, background
        self.meta = meta or {}

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing columns with training defaults and fix order / dtypes."""
        df = df.copy()
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = self.defaults[c]
        out = prepare_frame(df, self.feature_cols, self.cat_cols)
        for c in self.num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def transform(self, df):
        return self.prep.transform(self.frame(df))

    def predict_proba(self, df):
        return self.model.predict_proba(self.transform(df))

    def predict(self, df):
        return self.model.predict(self.transform(df))


def make_defaults(df: pd.DataFrame, num_cols, cat_cols) -> dict:
    d = {}
    for c in num_cols:
        d[c] = float(pd.to_numeric(df[c], errors="coerce").median())
    for c in cat_cols:
        d[c] = str(_to_cat_str(df[c]).mode().iloc[0])
    return d


# ----------------------------------------------------------- explainability
def permutation_importance(predict_fn, X: pd.DataFrame, y, loss_fn, n_repeats=5, seed=C.SEED):
    """Raw-feature permutation importance: increase in loss when a feature is shuffled."""
    rng = np.random.default_rng(seed)
    base = loss_fn(y, predict_fn(X))
    rows = []
    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            deltas.append(loss_fn(y, predict_fn(Xp)) - base)
        rows.append((col, float(np.mean(deltas)), float(np.std(deltas))))
    return (pd.DataFrame(rows, columns=["feature", "importance", "std"])
            .sort_values("importance", ascending=False).reset_index(drop=True))


def shap_attributions(bundle: TabularBundle, X_explain: pd.DataFrame, class_index=None,
                      n_background=30, max_evals=None, seed=C.SEED):
    """Model-agnostic SHAP values (permutation explainer) on RAW features.

    Categorical features are integer-coded for the explainer and decoded again
    inside the prediction wrapper, so attributions are per original feature.
    Returns (values[n, p], base_value, feature_names). ``class_index`` selects the
    class probability being explained for classifiers; regressors explain the output.
    """
    import shap
    cols = bundle.feature_cols
    options = {c: list(bundle.ui_spec[c]["options"]) for c in bundle.cat_cols}

    def encode(df: pd.DataFrame) -> np.ndarray:
        d = bundle.frame(df)[cols].copy()
        for c in bundle.cat_cols:
            lookup = {v: i for i, v in enumerate(options[c])}
            d[c] = d[c].map(lambda v, lk=lookup: lk.get(v, -1)).astype(float)
        return d.to_numpy(dtype=float)

    def decode(arr: np.ndarray) -> pd.DataFrame:
        d = pd.DataFrame(np.asarray(arr, dtype=float), columns=cols)
        for c in bundle.cat_cols:
            opts = options[c]
            idx = np.clip(np.round(d[c].to_numpy()).astype(int), 0, len(opts) - 1)
            d[c] = [opts[i] for i in idx]
        return d

    def f(arr):
        d = decode(arr)
        if bundle.task == "classification":
            return bundle.predict_proba(d)[:, class_index]
        return bundle.predict(d)

    bg = bundle.background.sample(min(n_background, len(bundle.background)), random_state=seed)
    masker = shap.maskers.Independent(encode(bg), max_samples=len(bg))
    explainer = shap.Explainer(f, masker, algorithm="permutation", feature_names=cols)
    ex = explainer(encode(X_explain),
                   max_evals=max_evals or (4 * len(cols) + 1), silent=True)
    base = float(np.mean(ex.base_values))
    return np.asarray(ex.values, float), base, cols


def shap_global_table(values: np.ndarray, cols) -> pd.DataFrame:
    return (pd.DataFrame({"feature": cols, "importance": np.abs(values).mean(axis=0)})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# ------------------------------------------------------- convenience helpers
def fit_preprocessor(X: pd.DataFrame, idx_train, categorical_hint=(), min_frequency=10):
    """Fit the preprocessor on TRAIN rows only (no leakage). Returns (prep, num_cols, cat_cols)."""
    num, cat = split_columns(X, categorical_hint)
    prep = build_preprocessor(num, cat, min_frequency).fit(
        prepare_frame(X.iloc[idx_train], X.columns, cat))
    return prep, num, cat


def transform_rows(prep, X: pd.DataFrame, idx, cat_cols) -> np.ndarray:
    return prep.transform(prepare_frame(X.iloc[idx], X.columns, cat_cols))


def make_bundle(prep, model, X: pd.DataFrame, idx_train, num, cat, task, labels,
                meta=None, n_background=100, seed=C.SEED) -> TabularBundle:
    Xtr = X.iloc[idx_train]
    bg = Xtr.sample(min(n_background, len(Xtr)), random_state=seed)
    return TabularBundle(prep, model, X.columns, num, cat, task, labels,
                         make_defaults(Xtr, num, cat), make_ui_spec(Xtr, num, cat), bg, meta)
