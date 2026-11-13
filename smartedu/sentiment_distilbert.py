"""Module D (stronger model) - fine-tune DistilBERT for 3-class sentiment.

Needs:  pip install transformers   and internet access to download 'distilbert-base-uncased'
(~260 MB) the first time. On CPU this is slow; for >5k reviews use a free GPU (Google Colab,
Kaggle) or subsample with --max-train. Uses the SAME train/val/test split as the BiLSTM so
the two models are directly comparable.

    python -m smartedu.sentiment_distilbert --source real --epochs 3
    python -m smartedu.sentiment_distilbert --smoke        # offline self-test, tiny random model
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data_loaders import load_feedback
from .nn import _balanced_weights
from .sentiment import LABELS, L2I, Vocab, tokenize
from .utils import clf_metrics, plot_confusion, save_json, set_seed, split_indices


class DistilBertPredictorModel:
    """Inference wrapper around a saved fine-tuned model directory."""

    def __init__(self, path, max_len=128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
        self.max_len = max_len

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=32):
        texts, out = list(texts), []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], truncation=True, padding=True,
                           max_length=self.max_len, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, dim=1).numpy())
        return np.concatenate(out)


def _predict(model, encode, texts, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            ids, mask = encode(texts[i:i + bs])
            out.append(torch.softmax(model(input_ids=ids, attention_mask=mask).logits, dim=1).numpy())
    return np.concatenate(out)


def fine_tune(model, encode, texts_tr, y_tr, texts_va, y_va, epochs=3, batch_size=16, lr=3e-5,
              patience=2, seed=C.SEED, verbose=True):
    """Plain PyTorch fine-tuning loop (class-weighted CE, linear warm-up/decay, early stopping)."""
    set_seed(seed)
    w = torch.as_tensor(_balanced_weights(np.asarray(y_tr), len(LABELS)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(texts_tr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(1, int(0.1 * steps)), max(0.0, (steps - s) / max(1, steps))))
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1.0, None, 0, {"train_loss": [], "val_loss": [], "best_epoch": 0}
    ytr_t = torch.as_tensor(np.asarray(y_tr))
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(texts_tr))
        tot, cnt, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            ids, mask = encode([texts_tr[j] for j in idx])
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        pv = _predict(model, encode, texts_va)
        vloss = float(nn.functional.nll_loss(
            torch.log(torch.as_tensor(pv) + 1e-9), torch.as_tensor(np.asarray(y_va)), weight=w).item())
        vf1 = clf_metrics(np.asarray(y_va), pv.argmax(1))["macro_f1"]
        hist["train_loss"].append(tot / max(cnt, 1))
        hist["val_loss"].append(vloss)
        if verbose:
            print(f"    epoch {ep + 1}/{epochs}  train loss {tot / max(cnt, 1):.4f}  "
                  f"val macroF1 {vf1:.4f}  ({time.time() - t0:.0f}s)")
        if vf1 > best:
            best, bad, hist["best_epoch"] = vf1, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def run(source="real", model_name="distilbert-base-uncased", epochs=3, batch_size=16, lr=3e-5,
        max_len=128, max_train=None, smoke=False, text_col=None, label_col=None, verbose=True):
    set_seed()
    df = load_feedback("synthetic" if smoke else source, n_synth=600 if smoke else 3000,
                       text_col=text_col, label_col=label_col)
    df = df.assign(y=df["label"].map(L2I)).dropna(subset=["y"]).reset_index(drop=True)
    texts, y = df["text"].tolist(), df["y"].astype(int).to_numpy()
    itr, iva, ite = split_indices(len(df), y)          # same split as the BiLSTM
    if max_train and len(itr) > max_train:
        itr = np.random.default_rng(C.SEED).choice(itr, max_train, replace=False)
    sel = lambda idx: [texts[i] for i in idx]  # noqa: E731

    if smoke:   # offline: tiny random DistilBERT + our own vocabulary (tests the loop only)
        from transformers import DistilBertConfig, DistilBertForSequenceClassification
        vocab = Vocab.build([tokenize(t) for t in sel(itr)], min_freq=1)
        cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=1, n_heads=2,
                               max_position_embeddings=max_len, num_labels=len(LABELS))
        model = DistilBertForSequenceClassification(cfg)

        def encode(batch):
            enc = [vocab.encode(tokenize(t), 48) for t in batch]
            ids = torch.as_tensor([e[0] for e in enc])
            return ids, (ids != vocab.PAD).long()
        epochs = min(epochs, 3)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)

        def encode(batch):
            e = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            return e["input_ids"], e["attention_mask"]

    t0 = time.time()
    model, hist = fine_tune(model, encode, sel(itr), y[itr], sel(iva), y[iva], epochs=epochs,
                            batch_size=batch_size, lr=3e-4 if smoke else lr, verbose=verbose)
    pr = _predict(model, encode, sel(ite))
    m = clf_metrics(y[ite], pr.argmax(1), pr, LABELS)
    if verbose:
        print(f"    DistilBERT{' (smoke)' if smoke else ''}   acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}")
    if smoke:
        return {"smoke": True, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    out_dir = C.MODELS_DIR / "sentiment_distilbert"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    plot_confusion(y[ite], pr.argmax(1), LABELS, C.FIG_DIR / "sentiment_confusion_distilbert.png",
                   "Sentiment DistilBERT - test set")
    res = {"module": "sentiment_distilbert", "source": source, "model": model_name,
           "n_train": len(itr), "n_val": len(iva), "n_test": len(ite), "epochs_run": len(hist["train_loss"]),
           "test": {"DistilBERT (fine-tuned)": {
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "roc_auc": m.get("roc_auc"),
               "per_class": m["per_class"], "train_seconds": round(time.time() - t0, 1)}}}
    save_json(res, C.REPORTS_DIR / "metrics_sentiment_distilbert.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "synthetic"])
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=None, help="subsample training rows (CPU)")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--smoke", action="store_true", help="offline self-test with a tiny random model")
    a = ap.parse_args()
    run(a.source, a.model_name, a.epochs, a.batch_size, a.lr, a.max_len, a.max_train, a.smoke,
        a.text_col, a.label_col)
