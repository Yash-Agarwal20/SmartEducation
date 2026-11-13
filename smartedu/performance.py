"""Module A - Subject-clearance forecast  ("how many of next semester's subjects will this student clear?").

Question answered : A student registers for N subjects in semester 2. How many will they pass?
                    N - prediction = the expected "backlog" (subjects not cleared).
Data              : real university records (same file as Module B), students who registered for at
                    least one subject in semester 2 (4,244 rows).
Target            : "Curricular units 2nd sem (approved)" = subjects cleared in semester 2.
Two situations (``stage``), because *what we know* changes how good the forecast can be:
  start : before any exam result  -> background, money, and the number of subjects registered
  sem1  : after semester 1        -> + the semester-1 record (subjects cleared, grades ...)
Models : naive baseline (typical pass rate x registered subjects), Ridge, Random Forest, XGBoost,
         and the deep model: an MLP (Linear -> BatchNorm -> ReLU -> Dropout blocks).
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from . import config as C
from . import glossary as G
from .data_loaders import load_performance
from .nn import TorchTabular, tune_mlp
from .tabular import (fit_preprocessor, make_bundle, permutation_importance,
                      shap_attributions, shap_global_table, transform_rows)
from .utils import (plot_history, plot_importance, plot_pred_vs_actual, quick_eda,
                    reg_metrics, save_json, set_seed, split_indices)

TARGET, REG = C.PERFORMANCE_TARGET, C.PERFORMANCE_REGISTERED

# the inputs a teacher is asked for in the dashboard (the rest take typical values)
KEY_FEATURES = {"start": C.BACKGROUND_COLS + C.SEM2_REGISTRATION_COLS,
                "sem1": C.BACKGROUND_COLS + C.SEM1_COLS + C.SEM2_REGISTRATION_COLS}


class PassRateBaseline:
    """The 'no model' yardstick: every student clears the average share of what they registered."""

    def fit(self, X, y, registered):
        self.share_ = float(np.sum(y) / max(np.sum(registered), 1))
        return self

    def predict(self, registered):
        return self.share_ * np.asarray(registered, float)


def _clip(pred, registered):
    return np.clip(pred, 0, np.asarray(registered, float))


def _metrics(y, pred, registered):
    pred = _clip(pred, registered)
    m = reg_metrics(y, pred)
    m["within_1_subject"] = float(np.mean(np.abs(y - np.round(pred)) <= 1))
    m["backlog_mae"] = float(np.mean(np.abs((registered - y) - (registered - pred))))   # = MAE, kept explicit
    return m


def _baselines(quick: bool):
    return {
        "Ridge Regression": Ridge(alpha=3.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=120 if quick else 400, min_samples_leaf=3, n_jobs=-1, random_state=C.SEED),
        "XGBoost": XGBRegressor(
            n_estimators=150 if quick else 400, learning_rate=0.04, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=2.0, random_state=C.SEED, n_jobs=4, verbosity=0),
    }


def _fit_mlp(Xtr, ytr, Xva, yva, quick, params=None):
    kw = dict(max_epochs=80 if quick else 250, patience=15 if quick else 25, batch_size=64,
              hidden=(64, 32), dropout=0.3, lr=1e-3, weight_decay=1e-3)
    kw.update(params or {})
    return TorchTabular("regression", **kw).fit(Xtr, ytr, Xva, yva)


def _cross_validate(X, y, reg, quick, n_splits=5):
    """K-fold CV of every model (a second opinion next to the single test split)."""
    n_splits = 3 if quick else n_splits
    scores = {}
    for tr, te in KFold(n_splits, shuffle=True, random_state=C.SEED).split(X):
        inner_tr, inner_va, _ = split_indices(len(tr), stratify=False)
        tr_i, va_i = tr[inner_tr], tr[inner_va]
        prep, num, cat = fit_preprocessor(X, tr_i, C.UNIVERSITY_CATEGORICAL, min_frequency=15)
        Xt = lambda i: transform_rows(prep, X, i, cat)  # noqa: E731
        scores.setdefault("Typical pass rate (no model)", []).append(
            _metrics(y[te], PassRateBaseline().fit(None, y[tr], reg[tr]).predict(reg[te]), reg[te])["rmse"])
        for name, m in _baselines(quick).items():
            m.fit(Xt(tr), y[tr])
            scores.setdefault(name, []).append(_metrics(y[te], m.predict(Xt(te)), reg[te])["rmse"])
        mlp = _fit_mlp(Xt(tr_i), y[tr_i], Xt(va_i), y[va_i], quick)
        scores.setdefault("MLP (deep learning)", []).append(_metrics(y[te], mlp.predict(Xt(te)), reg[te])["rmse"])
    return {k: {"rmse_mean": float(np.mean(v)), "rmse_std": float(np.std(v))} for k, v in scores.items()}


def _run_stage(df, stage, quick, tune, shap_rows, verbose):
    set_seed()
    tag = f"performance_{stage}"
    cols = [c for c in C.STAGES[stage]["cols"] if c in df.columns]
    y = df[TARGET].to_numpy(float)
    reg = df[REG].to_numpy(float)
    X = df[cols].reset_index(drop=True)
    itr, iva, ite = split_indices(len(X), (y >= reg).astype(int))        # stratify: "cleared everything" or not
    prep, num, cat = fit_preprocessor(X, itr, C.UNIVERSITY_CATEGORICAL, min_frequency=15)
    Xt = lambda i: transform_rows(prep, X, i, cat)  # noqa: E731

    table = {}
    nb = PassRateBaseline().fit(None, y[itr], reg[itr])
    table["Typical pass rate (no model)"] = _metrics(y[ite], nb.predict(reg[ite]), reg[ite])
    for name, m in _baselines(quick).items():
        m.fit(Xt(itr), y[itr])
        table[name] = _metrics(y[ite], m.predict(Xt(ite)), reg[ite])
    params = None
    if tune:
        params = tune_mlp(Xt(itr), y[itr], Xt(iva), y[iva], task="regression", n_trials=10 if quick else 30)
    t0 = time.time()
    mlp = _fit_mlp(Xt(itr), y[itr], Xt(iva), y[iva], quick, params)
    pred = _clip(mlp.predict(Xt(ite)), reg[ite])
    table["MLP (deep learning)"] = {**_metrics(y[ite], pred, reg[ite]),
                                    "train_seconds": round(time.time() - t0, 1),
                                    "best_epoch": mlp.history_["best_epoch"]}
    if verbose:
        for k, v in table.items():
            print(f"    {k:30s} RMSE {v['rmse']:.3f}  MAE {v['mae']:.3f}  within-1 {v['within_1_subject']:.2f}")

    plot_history(mlp.history_, C.FIG_DIR / f"{tag}_training.png", f"Subjects-cleared MLP training ({stage})")
    plot_pred_vs_actual(y[ite], pred, C.FIG_DIR / f"{tag}_pred_vs_actual.png",
                        f"Test students: predicted vs actual subjects cleared ({C.STAGES[stage]['label']})")

    bundle = make_bundle(prep, mlp, X, itr, num, cat, "regression", None,
                         meta={"stage": stage, "metrics": table["MLP (deep learning)"],
                               "pass_share": nb.share_})
    Xte_df = X.iloc[ite]
    imp = permutation_importance(
        lambda d: _clip(bundle.predict(d), d[REG].to_numpy(float)), Xte_df, y[ite],
        lambda yt, yp: float(np.sqrt(np.mean((yt - yp) ** 2))), n_repeats=3 if quick else 6)
    imp_plot = imp.assign(feature=imp["feature"].map(G.label))
    plot_importance(imp_plot, C.FIG_DIR / f"{tag}_permutation_importance.png",
                    f"What matters most (shuffle test) - {C.STAGES[stage]['label']}", xlabel="Increase in error (subjects)")
    shap_info = None
    try:
        vals, base, cols_ = shap_attributions(bundle, Xte_df.head(shap_rows))
        g = shap_global_table(vals, cols_)
        plot_importance(g.assign(feature=g["feature"].map(G.label)), C.FIG_DIR / f"{tag}_shap_global.png",
                        f"Average push on the forecast (SHAP) - {C.STAGES[stage]['label']}", xlabel="mean |SHAP| (subjects)")
        shap_info = g.head(10).to_dict("records")
    except Exception as e:  # noqa: BLE001 - explainability must never break training
        print("    [warn] SHAP failed:", e)

    joblib.dump(bundle, C.MODELS_DIR / f"{tag}.joblib")
    cv = _cross_validate(X, y, reg, quick)
    if verbose:
        print("    CV RMSE:", {k: f"{v['rmse_mean']:.2f}+-{v['rmse_std']:.2f}" for k, v in cv.items()})
    return {"label": C.STAGES[stage]["label"], "test": table, "cv": cv, "n_train": len(itr), "n_val": len(iva),
            "n_test": len(ite), "n_inputs": len(cols), "std_target": float(np.std(y[ite])),
            "permutation_importance_top10": imp.head(10).to_dict("records"),
            "shap_top10": shap_info, "tuned_params": params}


def run(source="real", quick=False, tune=False, shap_rows=60, verbose=True) -> dict:
    df = load_performance(source)
    quick_eda(G.pretty_columns(df), "performance", G.label(TARGET))
    out = {"module": "performance", "source": source, "n_rows": len(df), "target": TARGET, "settings": {}}
    for stage in C.PERFORMANCE_STAGES:
        if verbose:
            print(f"  [Subjects cleared] stage '{stage}' ({C.STAGES[stage]['label']})")
        out["settings"][stage] = _run_stage(df, stage, quick, tune, shap_rows, verbose)
    save_json(out, C.REPORTS_DIR / "metrics_performance.json")
    return out
