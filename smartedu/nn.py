"""PyTorch building blocks shared by all modules.

* ``fit_loop``       - generic mini-batch trainer with AdamW, LR scheduling,
                       gradient clipping and EarlyStopping (restores best weights)
* ``MLP``            - Dense -> BatchNorm -> ReLU -> Dropout blocks
* ``TorchTabular``   - sklearn-style wrapper for tabular classification/regression
* ``SeqRiskNet`` / ``TorchSeqRisk`` - BiLSTM over a weekly sequence + static features
* ``tune_mlp``       - small Optuna search for MLP hyper-parameters
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .utils import clf_metrics, reg_metrics, set_seed

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# ============================================================ generic trainer
def fit_loop(model, loss_fn, train_data, val_data, *, batch_size=64, lr=1e-3,
             weight_decay=1e-4, max_epochs=200, patience=20, seed=C.SEED,
             verbose=False):
    """Train ``model`` on tuples of tensors ``(inputs..., y)``.

    ``model(*inputs)`` returns the network output, ``loss_fn(output, y)`` a scalar.
    Early stopping monitors validation loss; the best weights are restored.
    Returns a history dict with train_loss / val_loss / best_epoch.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(3, patience // 3))
    n = len(train_data[-1])
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    hist = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            batch = [t[idx] for t in train_data]
            opt.zero_grad()
            loss = loss_fn(model(*batch[:-1]), batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(*val_data[:-1]), val_data[-1]).item()
        hist["train_loss"].append(tot / max(cnt, 1))
        hist["val_loss"].append(v)
        sched.step(v)
        if v < best - 1e-5:
            best, best_state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"  epoch {ep:3d} train {hist['train_loss'][-1]:.4f} val {v:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_ep
    model.eval()
    return hist


def _balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return len(y) / (n_classes * counts)


# ===================================================================== MLP
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchTabular:
    """sklearn-like MLP for tabular data. ``task`` is 'classification' or 'regression'."""

    def __init__(self, task="classification", n_classes=2, hidden=(128, 64, 32),
                 dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 max_epochs=200, patience=20, class_weight="balanced",
                 seed=C.SEED, verbose=False):
        self.task, self.n_classes = task, n_classes
        self.hidden, self.dropout = tuple(hidden), dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.class_weight, self.seed, self.verbose = class_weight, seed, verbose
        self.model, self.history_, self.in_dim_ = None, None, None
        self.y_mean_, self.y_std_ = 0.0, 1.0

    # ---- pickling: store weights, not the module object
    def __getstate__(self):
        d = self.__dict__.copy()
        if d.get("model") is not None:
            d["model"] = {k: v.cpu() for k, v in d["model"].state_dict().items()}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        if isinstance(self.model, dict):
            sd = self.model
            self.model = self._build(self.in_dim_)
            self.model.load_state_dict(sd)
            self.model.eval()

    def _build(self, in_dim):
        out = self.n_classes if self.task == "classification" else 1
        return MLP(in_dim, out, self.hidden, self.dropout)

    def fit(self, X, y, X_val, y_val):
        set_seed(self.seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        Xv = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
        self.in_dim_ = X.shape[1]
        self.model = self._build(self.in_dim_)
        if self.task == "classification":
            y_np, yv_np = np.asarray(y).astype(int), np.asarray(y_val).astype(int)
            w = None
            if self.class_weight == "balanced":
                w = torch.as_tensor(_balanced_weights(y_np, self.n_classes), dtype=torch.float32)
            loss_fn = nn.CrossEntropyLoss(weight=w)
            yt, yvt = torch.as_tensor(y_np), torch.as_tensor(yv_np)
        else:
            y_np = np.asarray(y, float)
            self.y_mean_, self.y_std_ = float(y_np.mean()), float(y_np.std() or 1.0)
            yt = torch.as_tensor((y_np - self.y_mean_) / self.y_std_, dtype=torch.float32)
            yvt = torch.as_tensor((np.asarray(y_val, float) - self.y_mean_) / self.y_std_,
                                  dtype=torch.float32)
            mse = nn.MSELoss()
            loss_fn = lambda out, t: mse(out.squeeze(-1), t)  # noqa: E731
        self.history_ = fit_loop(
            self.model, loss_fn, (X, yt), (Xv, yvt), batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, max_epochs=self.max_epochs,
            patience=self.patience, seed=self.seed, verbose=self.verbose)
        return self

    @torch.no_grad()
    def _raw(self, X):
        self.model.eval()
        return self.model(torch.as_tensor(np.asarray(X), dtype=torch.float32))

    def predict_proba(self, X):
        return torch.softmax(self._raw(X), dim=1).numpy()

    def predict(self, X):
        if self.task == "classification":
            return self.predict_proba(X).argmax(1)
        return self._raw(X).squeeze(-1).numpy() * self.y_std_ + self.y_mean_


# ============================================================ sequence model
class SeqRiskNet(nn.Module):
    """BiLSTM over weekly signals, fused with an embedding of static features."""

    def __init__(self, seq_dim, static_dim, hidden=48, layers=1, dropout=0.3, n_out=2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.static = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden * 2 + 32, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_out))

    def forward(self, seq, static):
        out, _ = self.lstm(seq)                       # (B, T, 2H)
        pooled = torch.cat([out.mean(1), out[:, -1, :]], dim=1)   # mean + last step
        return self.head(torch.cat([pooled, self.static(static)], dim=1))


class TorchSeqRisk:
    """Binary classifier on (sequence, static) inputs. Channels are standardised internally."""

    def __init__(self, hidden=48, layers=1, dropout=0.3, lr=2e-3, weight_decay=1e-4,
                 batch_size=64, max_epochs=150, patience=15, class_weight="balanced",
                 seed=C.SEED, verbose=False):
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.class_weight = max_epochs, patience, class_weight
        self.seed, self.verbose = seed, verbose
        self.model, self.history_ = None, None
        self.seq_dim_ = self.static_dim_ = None
        self.mu_ = self.sd_ = None

    def __getstate__(self):
        d = self.__dict__.copy()
        if d.get("model") is not None:
            d["model"] = {k: v.cpu() for k, v in d["model"].state_dict().items()}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        if isinstance(self.model, dict):
            sd = self.model
            self.model = self._build()
            self.model.load_state_dict(sd)
            self.model.eval()

    def _build(self):
        return SeqRiskNet(self.seq_dim_, self.static_dim_, self.hidden, self.layers, self.dropout)

    def _scale(self, seq):
        return torch.as_tensor((np.asarray(seq, float) - self.mu_) / self.sd_, dtype=torch.float32)

    def fit(self, seq, static, y, seq_val, static_val, y_val):
        set_seed(self.seed)
        seq = np.asarray(seq, float)
        self.seq_dim_, self.static_dim_ = seq.shape[2], np.asarray(static).shape[1]
        self.mu_ = seq.mean(axis=(0, 1), keepdims=True)
        self.sd_ = seq.std(axis=(0, 1), keepdims=True) + 1e-6
        self.model = self._build()
        y_np = np.asarray(y).astype(int)
        w = None
        if self.class_weight == "balanced":
            w = torch.as_tensor(_balanced_weights(y_np, 2), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        tr = (self._scale(seq), torch.as_tensor(np.asarray(static), dtype=torch.float32),
              torch.as_tensor(y_np))
        va = (self._scale(seq_val), torch.as_tensor(np.asarray(static_val), dtype=torch.float32),
              torch.as_tensor(np.asarray(y_val).astype(int)))
        self.history_ = fit_loop(
            self.model, loss_fn, tr, va, batch_size=self.batch_size, lr=self.lr,
            weight_decay=self.weight_decay, max_epochs=self.max_epochs,
            patience=self.patience, seed=self.seed, verbose=self.verbose)
        return self

    @torch.no_grad()
    def predict_proba(self, seq, static):
        self.model.eval()
        out = self.model(self._scale(seq), torch.as_tensor(np.asarray(static), dtype=torch.float32))
        return torch.softmax(out, dim=1).numpy()


# ================================================================== tuning
def tune_mlp(Xtr, ytr, Xval, yval, task="classification", n_classes=2, n_trials=20,
             max_epochs=120, seed=C.SEED):
    """Optuna search over MLP hyper-parameters. Returns the best-parameter dict."""
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install optuna to use --tune:  pip install optuna") from e
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        depth = trial.suggest_int("depth", 1, 3)
        params = dict(
            hidden=tuple(max(8, width // (2 ** i)) for i in range(depth)),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        )
        m = TorchTabular(task=task, n_classes=n_classes, max_epochs=max_epochs,
                         patience=12, seed=seed, **params).fit(Xtr, ytr, Xval, yval)
        if task == "classification":
            return -clf_metrics(yval, m.predict(Xval))["macro_f1"]
        return reg_metrics(yval, m.predict(Xval))["rmse"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    return dict(
        hidden=tuple(max(8, bp["width"] // (2 ** i)) for i in range(bp["depth"])),
        dropout=bp["dropout"], lr=bp["lr"], weight_decay=bp["weight_decay"],
        batch_size=bp["batch_size"])
