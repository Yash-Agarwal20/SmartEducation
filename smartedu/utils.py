"""Shared helpers: seeding, splitting, metrics, plots, quick EDA, JSON I/O."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import textwrap

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_recall_fscore_support,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from . import config as C  # noqa: E402


def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ splitting
def split_indices(n: int, y=None, stratify: bool = True, seed: int = C.SEED):
    """70/15/15 train/val/test indices. Stratified when y is given."""
    idx = np.arange(n)
    strat = y if (stratify and y is not None) else None
    idx_trval, idx_te = train_test_split(
        idx, test_size=C.TEST_FRAC, random_state=seed, stratify=strat)
    strat2 = np.asarray(y)[idx_trval] if strat is not None else None
    rel_val = C.VAL_FRAC / (1.0 - C.TEST_FRAC)
    idx_tr, idx_val = train_test_split(
        idx_trval, test_size=rel_val, random_state=seed, stratify=strat2)
    return idx_tr, idx_val, idx_te


# -------------------------------------------------------------------- metrics
def reg_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clf_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1 and ROC-AUC (binary or OvR macro)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.arange(len(labels))
    names = [str(c) for c in classes] if labels is None else list(labels)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            n: {"precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
            for n, pi, ri, fi, si in zip(names, p, r, f, s)
        },
    }
    if y_proba is not None:
        try:
            y_proba = np.asarray(y_proba)
            if len(classes) == 2:
                pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                out["roc_auc"] = float(roc_auc_score(y_true, pos))
                out["pr_auc"] = float(average_precision_score(y_true, pos))
            else:
                out["roc_auc"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro", labels=classes))
        except ValueError:
            out["roc_auc"] = None
    return out


def best_threshold(y_true, p_pos, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on validation data (beta=2 favours recall)."""
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p_pos >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        s = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if (prec + rec) else 0.0
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


# ---------------------------------------------------------------------- plots
def plot_confusion(y_true, y_pred, labels, path: Path, title: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(textwrap.fill(str(title), 46))
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(history: dict, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    if history.get("best_epoch") is not None:
        ax.axvline(history["best_epoch"], color="grey", ls="--", lw=1, label="best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(textwrap.fill(str(title), 46))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pred_vs_actual(y_true, y_pred, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(textwrap.fill(str(title), 46))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(df: pd.DataFrame, path: Path, title: str, top: int = 15,
                    xlabel: str = "Importance") -> None:
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(d) + 1.4))
    ax.barh(d["feature"], d["importance"], color="#3b6ea8")
    ax.set_xlabel(xlabel)
    ax.set_title(textwrap.fill(str(title), 46))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def quick_eda(df: pd.DataFrame, name: str, target: str | None = None) -> None:
    """Saves a few standard EDA figures: target distribution, missing values, correlations."""
    fig_dir = C.FIG_DIR
    if target is not None and target in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        if df[target].nunique() <= 12:
            df[target].value_counts().sort_index().plot.bar(ax=ax, color="#3b6ea8")
        else:
            df[target].plot.hist(ax=ax, bins=20, color="#3b6ea8")
        ax.set_title(f"{name}: distribution of {target}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"eda_{name}_target.png", dpi=130)
        plt.close(fig)
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(15)
    if len(miss):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        miss.iloc[::-1].plot.barh(ax=ax, color="#c0504d")
        ax.set_title(f"{name}: share of missing values")
        fig.tight_layout()
        fig.savefig(fig_dir / f"eda_{name}_missing.png", dpi=130)
        plt.close(fig)
    num = df.select_dtypes("number")
    if num.shape[1] >= 3:
        if target is not None and target in num.columns:
            top = num.corr()[target].abs().sort_values(ascending=False).head(15).index
        else:
            top = num.columns[:15]
        corr = num[top].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(top)), top, rotation=75, ha="right", fontsize=7)
        ax.set_yticks(range(len(top)), top, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{name}: correlation (top numeric features)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"eda_{name}_correlation.png", dpi=130)
        plt.close(fig)


# ------------------------------------------------------------------------ I/O
def save_json(obj, path: Path) -> None:
    def _default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    Path(path).write_text(json.dumps(obj, indent=2, default=_default))


def load_json(path: Path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None
