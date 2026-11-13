"""Module B - Dropout prediction (3 outcomes: Dropout / Still enrolled / Graduate).

Question answered : Given what we know about a student at this point in time, will they drop out,
                    still be studying, or graduate?
Data              : real university records (4,424 students; same file as Module A).
Three situations (``stage``) - the more we know, the better the answer:
  start : before any exam result (background, money, registration only) - a rough first screen
  sem1  : after semester 1  - the realistic EARLY WARNING (time to intervene)
  sem2  : after semester 2  - most accurate, but latest
Inputs  : background, money matters and the semester record(s). Sensitive attributes (gender,
          nationality, marital status ...) and parents' jobs are NOT used - see glossary.EXCLUDED_AND_WHY.
Imbalance handled with class weights (default) or SMOTE. Baselines: Logistic Regression, Random Forest,
XGBoost.  Deep model: MLP.  Headline metric: RECALL of the Dropout class (missing a student who is
about to leave costs more than a false alarm), next to macro-F1 and ROC-AUC.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as C
from . import glossary as G
from .data_loaders import load_dropout
from .nn import TorchTabular, tune_mlp
from .tabular import (fit_preprocessor, make_bundle, permutation_importance,
                      shap_attributions, shap_global_table, transform_rows)
from .utils import (clf_metrics, plot_confusion, plot_history, plot_importance,
                    quick_eda, save_json, set_seed, split_indices)

LABELS = C.DROPOUT_LABELS
FRIENDLY_LABELS = ["Drops out", "Still studying", "Graduates"]
DROPOUT_IDX = LABELS.index("Dropout")


def _smote(Xtr, ytr, seed=C.SEED):
    from imblearn.over_sampling import SMOTE
    k = max(1, min(5, int(np.bincount(ytr).min()) - 1))
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xtr, ytr)


def _row(m):
    pc = m["per_class"]["Dropout"]
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "roc_auc": m.get("roc_auc"),
            "dropout_recall": pc["recall"], "dropout_precision": pc["precision"], "dropout_f1": pc["f1"],
            "per_class": {k: {kk: vv for kk, vv in v.items()} for k, v in m["per_class"].items()}}


def _run_stage(df, stage, quick, tune, imbalance, shap_rows, verbose):
    set_seed()
    tag = f"dropout_{stage}"
    cols = [c for c in C.STAGES[stage]["cols"] if c in df.columns]
    y = df[C.DROPOUT_TARGET].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    X = df[cols].reset_index(drop=True)
    itr, iva, ite = split_indices(len(X), y)
    prep, num, cat = fit_preprocessor(X, itr, C.DROPOUT_CATEGORICAL, min_frequency=15)
    Xt = lambda i: transform_rows(prep, X, i, cat)  # noqa: E731
    Xtr, Xva, Xte = Xt(itr), Xt(iva), Xt(ite)
    ytr, yva, yte = y[itr], y[iva], y[ite]
    if verbose:
        counts = {l: int(c) for l, c in zip(LABELS, np.bincount(ytr))}
        print(f"  [Dropout/{stage}] inputs {len(cols)} -> {Xtr.shape[1]} after encoding | train class counts {counts}")

    Xfit, yfit = (_smote(Xtr, ytr) if imbalance == "smote" else (Xtr, ytr))
    sw = None if imbalance == "smote" else compute_sample_weight("balanced", ytr)
    cw = "balanced" if imbalance == "weights" else None
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5, class_weight=cw),
        "Random Forest": RandomForestClassifier(n_estimators=150 if quick else 500, min_samples_leaf=2,
                                                class_weight=cw, n_jobs=-1, random_state=C.SEED),
        "XGBoost": XGBClassifier(n_estimators=150 if quick else 400, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.7, objective="multi:softprob",
                                 eval_metric="mlogloss", random_state=C.SEED, n_jobs=4, verbosity=0),
    }
    table, full = {}, {}
    for name, m in models.items():
        m.fit(Xfit, yfit, sample_weight=sw) if name == "XGBoost" else m.fit(Xfit, yfit)
        pr = m.predict_proba(Xte)
        met = clf_metrics(yte, pr.argmax(1), pr, LABELS)
        table[name], full[name] = _row(met), met

    params = None
    if tune:
        params = tune_mlp(Xtr, ytr, Xva, yva, "classification", 3, n_trials=10 if quick else 30)
    kw = dict(max_epochs=80 if quick else 300, patience=12 if quick else 25, batch_size=64,
              hidden=(128, 64, 32), dropout=0.3, lr=1e-3, weight_decay=1e-3)
    kw.update(params or {})
    t0 = time.time()
    mlp = TorchTabular("classification", n_classes=3, class_weight=cw, **kw).fit(Xfit, yfit, Xva, yva)
    pr = mlp.predict_proba(Xte)
    met = clf_metrics(yte, pr.argmax(1), pr, LABELS)
    table["MLP (deep learning)"], full["MLP (deep learning)"] = _row(met), met
    table["MLP (deep learning)"]["train_seconds"] = round(time.time() - t0, 1)
    table["MLP (deep learning)"]["best_epoch"] = mlp.history_["best_epoch"]
    if verbose:
        for k, r in table.items():
            print(f"    {k:22s} acc {r['accuracy']:.3f}  macroF1 {r['macro_f1']:.3f}  "
                  f"dropout-recall {r['dropout_recall']:.3f}  AUC {r['roc_auc']:.3f}")

    plot_history(mlp.history_, C.FIG_DIR / f"{tag}_training.png", f"Dropout MLP training ({stage})")
    plot_confusion(yte, pr.argmax(1), FRIENDLY_LABELS, C.FIG_DIR / f"{tag}_confusion_mlp.png",
                   f"Dropout MLP on unseen students - {C.STAGES[stage]['label']}")

    bundle = make_bundle(prep, mlp, X, itr, num, cat, "classification", LABELS,
                         meta={"stage": stage, "metrics": table["MLP (deep learning)"], "imbalance": imbalance})
    Xte_df = X.iloc[ite]
    imp = permutation_importance(
        lambda d: bundle.predict(d), Xte_df, yte,
        lambda yt, yp: 1.0 - clf_metrics(yt, yp)["macro_f1"], n_repeats=3 if quick else 6)
    plot_importance(imp.assign(feature=imp["feature"].map(G.label)), C.FIG_DIR / f"{tag}_permutation_importance.png",
                    f"What matters most (shuffle test) - {C.STAGES[stage]['label']}", xlabel="Drop in quality when shuffled")
    shap_info = None
    try:
        vals, base, cols_ = shap_attributions(bundle, Xte_df.head(shap_rows), class_index=DROPOUT_IDX)
        g = shap_global_table(vals, cols_)
        plot_importance(g.assign(feature=g["feature"].map(G.label)), C.FIG_DIR / f"{tag}_shap_global.png",
                        f"Average push on the dropout probability (SHAP) - {C.STAGES[stage]['label']}",
                        xlabel="mean |SHAP| (probability points)")
        shap_info = g.head(10).to_dict("records")
    except Exception as e:  # noqa: BLE001
        print("    [warn] SHAP failed:", e)

    joblib.dump(bundle, C.MODELS_DIR / f"{tag}.joblib")
    return {"label": C.STAGES[stage]["label"], "n_train": len(itr), "n_val": len(iva), "n_test": len(ite),
            "n_inputs": len(cols), "test": table, "full_mlp_metrics": full["MLP (deep learning)"],
            "permutation_importance_top10": imp.head(10).to_dict("records"),
            "shap_top10": shap_info, "tuned_params": params}


def run(source="real", quick=False, tune=False, imbalance="weights", shap_rows=60, verbose=True) -> dict:
    df = load_dropout(source)
    quick_eda(G.pretty_columns(df), "dropout", G.label(C.DROPOUT_TARGET))
    out = {"module": "dropout", "source": source, "n_rows": len(df), "imbalance": imbalance, "settings": {}}
    for stage in C.DROPOUT_STAGES:
        out["settings"][stage] = _run_stage(df, stage, quick, tune, imbalance, shap_rows, verbose)
    save_json(out, C.REPORTS_DIR / "metrics_dropout.json")
    return out
