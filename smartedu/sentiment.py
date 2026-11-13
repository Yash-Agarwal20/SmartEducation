"""Module D - Student feedback sentiment analysis (Negative / Neutral / Positive).

Pipeline: text cleaning -> tokenisation -> vocabulary -> Embedding -> BiLSTM -> pooling -> softmax
Baseline: TF-IDF (1-2 grams) + Logistic Regression.
Stronger model (optional, needs internet + transformers): fine-tuned DistilBERT, see
``sentiment_distilbert.py``. ``load_sentiment_predictor`` picks DistilBERT when it has been
trained, otherwise the BiLSTM.
"""
from __future__ import annotations

import re
import time
from collections import Counter

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config as C
from .data_loaders import load_feedback
from .nn import _balanced_weights, fit_loop
from .utils import clf_metrics, plot_confusion, plot_history, save_json, set_seed, split_indices

LABELS = C.SENTIMENT_LABELS
L2I = {l: i for i, l in enumerate(LABELS)}
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?]")


def clean_text(t: str) -> str:
    t = _HTML.sub(" ", _URL.sub(" ", str(t))).lower()
    return re.sub(r"\s+", " ", t).strip()


def tokenize(t: str) -> list[str]:
    return _TOKEN.findall(clean_text(t))


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=20000):
        cnt = Counter(w for toks in token_lists for w in toks)
        words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(w, self.UNK) for w in tokens[:max_len]] or [self.UNK]
        return ids + [self.PAD] * (max_len - len(ids)), len(ids)

    def __len__(self):
        return len(self.itos)


class TextNet(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=64, n_classes=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(4 * hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(64, n_classes))

    def forward(self, tokens, lengths):
        x = self.drop(self.emb(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=tokens.shape[1])
        mask = (torch.arange(tokens.shape[1])[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (out * mask).sum(1) / lengths[:, None].clamp(min=1)
        mx = out.masked_fill(~mask, -1e9).max(1).values
        return self.out(torch.cat([mean, mx], dim=1))


class TorchTextClassifier:
    """Embedding + BiLSTM text classifier with a ``predict_proba(list[str])`` API."""

    def __init__(self, emb=128, hidden=64, dropout=0.4, lr=2e-3, weight_decay=1e-4, batch_size=64,
                 max_epochs=40, patience=5, max_len=80, seed=C.SEED, verbose=False):
        self.emb, self.hidden, self.dropout = emb, hidden, dropout
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.max_len = max_epochs, patience, max_len
        self.seed, self.verbose = seed, verbose
        self.vocab, self.model, self.history_ = None, None, None

    def __getstate__(self):
        d = self.__dict__.copy()
        if d.get("model") is not None:
            d["model"] = {k: v.cpu() for k, v in d["model"].state_dict().items()}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        if isinstance(self.model, dict):
            sd = self.model
            self.model = TextNet(len(self.vocab), self.emb, self.hidden, len(LABELS), self.dropout)
            self.model.load_state_dict(sd)
            self.model.eval()

    def _encode(self, texts):
        enc = [self.vocab.encode(tokenize(t), self.max_len) for t in texts]
        return (torch.as_tensor([e[0] for e in enc]), torch.as_tensor([e[1] for e in enc]))

    def fit(self, texts, y, val_texts, val_y):
        set_seed(self.seed)
        self.vocab = Vocab.build([tokenize(t) for t in texts])
        self.model = TextNet(len(self.vocab), self.emb, self.hidden, len(LABELS), self.dropout)
        w = torch.as_tensor(_balanced_weights(np.asarray(y).astype(int), len(LABELS)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        tok, ln = self._encode(texts)
        vtok, vln = self._encode(val_texts)
        self.history_ = fit_loop(
            self.model, loss_fn, (tok, ln, torch.as_tensor(np.asarray(y).astype(int))),
            (vtok, vln, torch.as_tensor(np.asarray(val_y).astype(int))),
            batch_size=self.batch_size, lr=self.lr, weight_decay=self.weight_decay,
            max_epochs=self.max_epochs, patience=self.patience, seed=self.seed, verbose=self.verbose)
        return self

    @torch.no_grad()
    def predict_proba(self, texts):
        self.model.eval()
        tok, ln = self._encode(list(texts))
        return torch.softmax(self.model(tok, ln), dim=1).numpy()

    def predict(self, texts):
        return self.predict_proba(texts).argmax(1)


class SentimentPredictor:
    """Uniform wrapper used by the dashboard (BiLSTM or DistilBERT behind the same API)."""

    def __init__(self, model, name):
        self.model, self.name, self.labels = model, name, LABELS

    def predict_proba(self, texts):
        return self.model.predict_proba(list(texts))

    def explain_tokens(self, text: str):
        """Word-occlusion attribution, in reading order: drop one word at a time and measure the fall in the
        predicted class's probability. Returns (label, words, effects). Positive effect = word supports the label."""
        toks = text.split()
        if not toks:
            return None, [], np.array([])
        base = self.predict_proba([text])[0]
        cls = int(base.argmax())
        variants = [" ".join(toks[:i] + toks[i + 1:]) or "." for i in range(len(toks))]
        eff = base[cls] - self.predict_proba(variants)[:, cls]
        return LABELS[cls], toks, eff

    def explain(self, text: str, top: int = 12):
        """Top-``top`` most influential words for the predicted class (largest |effect| first)."""
        label, toks, eff = self.explain_tokens(text)
        if label is None:
            return None, []
        order = np.argsort(-np.abs(eff))[:top]
        return label, [(toks[i], float(eff[i])) for i in order]


def load_sentiment_predictor():
    """DistilBERT if it was trained (and transformers is installed), else the BiLSTM bundle."""
    bert_dir = C.MODELS_DIR / "sentiment_distilbert"
    if bert_dir.exists():
        try:
            from .sentiment_distilbert import DistilBertPredictorModel
            return SentimentPredictor(DistilBertPredictorModel(bert_dir), "DistilBERT (fine-tuned)")
        except Exception as e:  # noqa: BLE001
            print("[sentiment] DistilBERT unavailable, using BiLSTM:", e)
    p = C.MODELS_DIR / "sentiment_bilstm.joblib"
    if not p.exists():
        raise FileNotFoundError("Train first:  python train_all.py")
    return SentimentPredictor(joblib.load(p), "BiLSTM")


def run(source="real", quick=False, text_col=None, label_col=None, verbose=True) -> dict:
    set_seed()
    df = load_feedback(source, text_col=text_col, label_col=label_col)
    df = df.assign(y=df["label"].map(L2I)).dropna(subset=["y"]).reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    texts = df["text"].tolist()
    y = df["y"].to_numpy()
    itr, iva, ite = split_indices(len(df), y)
    if verbose:
        print(f"  [Sentiment] rows {len(df)}  class counts "
              f"{ {l: int((y == i).sum()) for l, i in L2I.items()} }")

    # fig: class distribution
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.bar(LABELS, [(y == i).sum() for i in range(3)], color=["#c0504d", "#9b9b9b", "#4f81bd"])
    ax.set_title("Sentiment: class distribution")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_sentiment_target.png", dpi=130)
    plt.close(fig)

    sel = lambda idx: [texts[i] for i in idx]  # noqa: E731
    table = {}

    # ---- baseline: TF-IDF + logistic regression
    tfidf = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = tfidf.fit_transform(sel(itr))
    lr = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced").fit(Xtr, y[itr])
    pr = lr.predict_proba(tfidf.transform(sel(ite)))
    m = clf_metrics(y[ite], pr.argmax(1), pr, LABELS)
    table["TF-IDF + Logistic Regression"] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                                             "roc_auc": m.get("roc_auc"), "per_class": m["per_class"]}
    if verbose:
        print(f"    TF-IDF + LogReg        acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}")
    # top words per class from the baseline (interpretability for the report)
    vocab_arr = np.array(tfidf.get_feature_names_out())
    top_words = {LABELS[k]: vocab_arr[np.argsort(-lr.coef_[k])[:12]].tolist() for k in range(3)}

    # ---- deep model: Embedding + BiLSTM
    t0 = time.time()
    clf = TorchTextClassifier(max_epochs=15 if quick else 40, patience=3 if quick else 5)
    clf.fit(sel(itr), y[itr], sel(iva), y[iva])
    pr = clf.predict_proba(sel(ite))
    m = clf_metrics(y[ite], pr.argmax(1), pr, LABELS)
    table["Embedding + BiLSTM (deep learning)"] = {
        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "roc_auc": m.get("roc_auc"),
        "per_class": m["per_class"], "train_seconds": round(time.time() - t0, 1),
        "best_epoch": clf.history_["best_epoch"]}
    if verbose:
        print(f"    Embedding + BiLSTM     acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}")
    plot_history(clf.history_, C.FIG_DIR / "sentiment_training.png", "BiLSTM training")
    plot_confusion(y[ite], pr.argmax(1), LABELS, C.FIG_DIR / "sentiment_confusion_bilstm.png",
                   "Sentiment BiLSTM - test set")

    joblib.dump(clf, C.MODELS_DIR / "sentiment_bilstm.joblib")
    out = {"module": "sentiment", "source": source, "n_rows": len(df),
           "n_train": len(itr), "n_val": len(iva), "n_test": len(ite),
           "test": table, "top_words_baseline": top_words}
    save_json(out, C.REPORTS_DIR / "metrics_sentiment.json")
    return out
