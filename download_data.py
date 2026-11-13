#!/usr/bin/env python
"""Download the REAL datasets into data/raw/ and prepare the feedback sample.

    python download_data.py

What it fetches (all public, from GitHub mirrors of the original datasets):
  * dropout.csv            UCI 'Predict Students' Dropout and Academic Success'
                           (Realinho et al., 2022) - 4,424 real university students.
                           Feeds BOTH the subject forecast (A) and the dropout warning (B).
  * feedback.csv           Coursera course reviews (~107k) -> class-balanced sample of
                           12,000 reviews, rating 1-2 = Negative, 3 = Neutral, 4-5 = Positive

Prefer the official file? Download it from https://archive.ics.uci.edu (dataset id 697) and drop them into data/raw/ under the same names - the loaders accept both the
original ';' separated files and the ',' separated mirrors.

Module C (attendance) has no public dataset; it uses a documented simulation
(smartedu/synthetic.py).  For real engagement data see OULAD (--attendance-source oulad).
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FILES = {
    "dropout.csv": ("https://raw.githubusercontent.com/hamzaezzine/Predict-students-dropout-and-"
                    "academic-success-using-machine-learning-algorithms/main/dataset.csv"),
    "coursera_reviews_full.csv": "https://raw.githubusercontent.com/sharmaroshan/Coursera-Reviews-Analysis/master/reviews.csv",
}
PER_CLASS = 4000
SEED = 42


def fetch(name, url):
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[skip] {name} already present")
        return
    print(f"[get ] {name}")
    urllib.request.urlretrieve(url, dest)


def check():
    d = pd.read_csv(RAW / "dropout.csv", sep=None, engine="python", encoding="utf-8-sig")
    assert d.shape == (4424, 35), f"dropout.csv has shape {d.shape}, expected (4424, 35)"
    print(f"[ok  ] dropout.csv {d.shape}, classes {d['Target'].value_counts().to_dict()}")


def make_feedback():
    out = RAW / "feedback.csv"
    src = pd.read_csv(RAW / "coursera_reviews_full.csv", encoding="utf-8-sig")
    src = src.dropna(subset=["Review", "Label"])
    src["Review"] = src["Review"].astype(str).str.strip()
    src = src[src["Review"].str.len() >= 3]
    src["label"] = pd.cut(src["Label"], bins=[0, 2, 3, 5], labels=["Negative", "Neutral", "Positive"]).astype(str)
    parts = [g.sample(n=min(PER_CLASS, len(g)), random_state=SEED) for _, g in src.groupby("label")]
    df = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = df.rename(columns={"Review": "text", "Label": "stars"})[["text", "label", "stars"]]
    df.to_csv(out, index=False)
    print(f"[ok  ] feedback.csv {df.shape}, classes {df['label'].value_counts().to_dict()} "
          f"(from {len(src):,} reviews; natural split was {src['label'].value_counts(normalize=True).round(3).to_dict()})")


if __name__ == "__main__":
    try:
        for n, u in FILES.items():
            fetch(n, u)
        check()
        make_feedback()
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"\nDownload failed: {e}\nSee the module docstring for the manual download links.")
    print("\nNext:  python train_all.py --source real")
