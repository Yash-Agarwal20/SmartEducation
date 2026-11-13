#!/usr/bin/env python
"""Train every module, save models to models/ and metrics/figures to reports/.

Examples
--------
  python train_all.py --source synthetic --quick     # 1-minute pipeline test (fake data!)
  python train_all.py --source real                  # real datasets in data/raw/
  python train_all.py --source real --tune           # + Optuna hyper-parameter search
  python train_all.py --only dropout sentiment       # a subset of modules
  python train_all.py --attendance-source oulad      # real engagement data (OULAD) for module C
  python train_all.py --imbalance smote              # SMOTE instead of class weights (dropout)
"""
import argparse
import time

from smartedu import config as C
from smartedu import attendance, dropout, performance, sentiment
from smartedu.utils import load_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="real", choices=["real", "synthetic"],
                    help="'real' = CSVs in data/raw/, 'synthetic' = generated test data")
    ap.add_argument("--attendance-source", default=None, choices=["synthetic", "oulad"],
                    help="default: simulated attendance (documented rules)")
    ap.add_argument("--only", nargs="+", choices=["performance", "dropout", "attendance", "sentiment"])
    ap.add_argument("--quick", action="store_true", help="few epochs / small ensembles (smoke test)")
    ap.add_argument("--tune", action="store_true", help="Optuna search for the MLPs (performance, dropout)")
    ap.add_argument("--imbalance", default="weights", choices=["weights", "smote"])
    ap.add_argument("--summary-only", action="store_true",
                    help="only rebuild reports/results_summary.md from the saved metrics (e.g. after Colab DistilBERT)")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default=None)
    a = ap.parse_args()

    if a.summary_only:
        write_summary(a.source)
        print("Rebuilt reports/results_summary.md")
        return
    todo = a.only or ["performance", "dropout", "attendance", "sentiment"]
    t0 = time.time()
    print(f"Training {todo} | source={a.source} | quick={a.quick} | tune={a.tune}\n")
    if "performance" in todo:
        performance.run(a.source, a.quick, a.tune)
    if "dropout" in todo:
        dropout.run(a.source, a.quick, a.tune, a.imbalance)
    if "attendance" in todo:
        src = a.attendance_source or "synthetic"
        attendance.run(src, a.quick)
    if "sentiment" in todo:
        sentiment.run(a.source, a.quick, a.text_col, a.label_col)

    write_summary(a.source)
    print(f"\nDone in {time.time() - t0:.0f}s.  Models -> models/   Metrics+figures -> reports/")
    print("Next:  streamlit run app/Home.py")


def write_summary(source):
    """reports/results_summary.md : one table per module, ready to paste into the report."""
    lines = ["# Results summary\n"]
    prov = {"performance": "university student records (real)", "dropout": "university student records (real)",
            "sentiment": "Coursera course reviews (real)"}
    mix = []
    lines.append("**Data provenance**\n")
    for k in ("performance", "dropout", "sentiment"):
        mm = load_json(C.REPORTS_DIR / f"metrics_{k}.json")
        if mm:
            if mm.get("source") == "synthetic":
                mix.append(k)
            lines.append(f"- {k}: {'SYNTHETIC - not a result' if mm.get('source') == 'synthetic' else prov[k]}")
    ma = load_json(C.REPORTS_DIR / "metrics_attendance.json")
    if ma:
        lines.append(f"- attendance: {ma['source']} (documented simulation unless OULAD was used)")
    if mix:
        lines.append(f"\n> **WARNING: {', '.join(mix)} ran on synthetic test data - those numbers are NOT results.** "
                     "Re-run with `--source real`.\n")
    m = load_json(C.REPORTS_DIR / "metrics_performance.json")
    if m:
        for s, d in m["settings"].items():
            lines += [f"\n## A. Subjects cleared in semester 2 - {d['label']}  (test n={d['n_test']}, "
                      f"spread of the target SD={d['std_target']:.2f})\n",
                      "| Model | RMSE (subjects) | MAE | R2 | Within +-1 subject | CV RMSE |", "|---|---|---|---|---|---|"]
            for k, v in d["test"].items():
                cv = d["cv"].get(k, {})
                lines.append(f"| {k} | {v['rmse']:.3f} | {v['mae']:.3f} | {v['r2']:.3f} | {v['within_1_subject']:.2f} | "
                             f"{cv.get('rmse_mean', float('nan')):.2f} +- {cv.get('rmse_std', float('nan')):.2f} |")
    m = load_json(C.REPORTS_DIR / "metrics_dropout.json")
    if m:
        for s, d in m["settings"].items():
            lines += [f"\n## B. Dropout - {d['label']} (test n={d['n_test']}, imbalance={m['imbalance']})\n",
                      "| Model | Accuracy | Macro-F1 | Dropout recall | Dropout precision | ROC-AUC |", "|---|---|---|---|---|---|"]
            for k, v in d["test"].items():
                lines.append(f"| {k} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} | {v['dropout_recall']:.3f} | "
                             f"{v['dropout_precision']:.3f} | {v['roc_auc']:.3f} |")
    m = ma
    if m:
        lines += [f"\n## C. Attendance risk ({m['source']}; label: {m['label_definition']}; test n={m['n_test']})\n",
                  "| Model | Recall (At-Risk) | Precision | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
        for k, v in m["test"].items():
            lines.append(f"| {k} | {v['recall']:.3f} | {v['precision']:.3f} | {v['f1']:.3f} | "
                         f"{v['roc_auc']:.3f} | {v['pr_auc']:.3f} |")
    m = load_json(C.REPORTS_DIR / "metrics_sentiment.json")
    if m:
        lines += [f"\n## D. Sentiment (test n={m['n_test']})\n", "| Model | Accuracy | Macro-F1 |", "|---|---|---|"]
        for k, v in m["test"].items():
            lines.append(f"| {k} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
        b = load_json(C.REPORTS_DIR / "metrics_sentiment_distilbert.json")
        if b:
            for k, v in b["test"].items():
                lines.append(f"| {k} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    (C.REPORTS_DIR / "results_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
