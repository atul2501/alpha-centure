"""Shared PyTorch trainer for every deep / transformer / SSM model (same inputs, budget, loss, early stopping).

Inputs: the compact SEQ_COLUMNS channel set (or `columns`), standardized with statistics of training rows only.
A sample for decision bar t is the window of the last L bars ending at t (inclusive): only past data.
Training: up to MAX_SAMPLES windows from training rows (strided), AdamW, at most MAX_STEPS steps, validation
loss every EVAL_EVERY steps on validation rows, early stopping, best weights restored.
Loss: Huber on the vol-normalized target (reg) or cross-entropy over SHORT / FLAT / LONG (cls).
Device: Apple MPS when available (falls back to CPU on any MPS error); one torch job at a time.
"""

import copy
import os

import numpy as np
import pandas as pd

from alpha.tournament.features.groups import SEQ_COLUMNS
from alpha.tournament.models.base import BaseTradingModel, Preprocessor, train_rows

MAX_SAMPLES = 40_000
MAX_STEPS = 1500
EVAL_EVERY = 150
PATIENCE = 3
BATCH = 256


def _torch():
    import torch

    torch.set_num_threads(int(os.environ.get("TOURNAMENT_TORCH_THREADS", "4")))
    return torch


def device():
    torch = _torch()
    if os.environ.get("TOURNAMENT_DEVICE"):
        return torch.device(os.environ["TOURNAMENT_DEVICE"])
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TorchModel(BaseTradingModel):
    family = "deep"
    complexity = 3
    input_kind = "sequence"
    target_kinds = ("reg", "cls")
    seq_len = 64
    tabular = False  # True: one row of features, no window (MLP / DNN)

    def build(self, n_in: int, n_out: int, L: int):
        raise NotImplementedError

    # ---- data ----
    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.columns or SEQ_COLUMNS
        return self.pre.transform(X[cols])

    def _windows(self, A: np.ndarray, ends: np.ndarray, L: int) -> np.ndarray:
        if self.tabular:
            return A[ends]
        offs = np.arange(-L + 1, 1)
        ix = ends[:, None] + offs[None, :]
        w = A[np.clip(ix, 0, None)]
        w[ix < 0] = 0.0
        return w  # (n, L, C)

    def fit(self, X, y, train, val):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cols = self.columns or SEQ_COLUMNS
        self.L = int(self.params.get("seq_len", self.seq_len))
        ti = train_rows(train, y)
        self.pre = Preprocessor().fit(X[cols].iloc[ti])
        A = self._matrix(X)
        ti = ti[ti >= (0 if self.tabular else self.L)]
        if len(ti) > MAX_SAMPLES:
            ti = ti[np.linspace(0, len(ti) - 1, MAX_SAMPLES).astype(int)]
        vi = train_rows(val, y)
        if len(vi) > 8000:
            vi = vi[np.linspace(0, len(vi) - 1, 8000).astype(int)]
        cls = self.target.kind == "cls"
        yv = y.to_numpy()
        self.dev = device()
        self.net = self.build(A.shape[1], 3 if cls else 1, self.L)
        try:
            self._train(A, yv, ti, vi, cls)
        except (RuntimeError, NotImplementedError) as e:  # MPS kernel gaps -> CPU
            if self.dev.type != "mps":
                raise
            self.dev = torch.device("cpu")
            self.net = self.build(A.shape[1], 3 if cls else 1, self.L)
            self._train(A, yv, ti, vi, cls)
        return self

    def _train(self, A, yv, ti, vi, cls):
        torch = _torch()
        net = self.net.to(self.dev)
        opt = torch.optim.AdamW(net.parameters(), lr=float(self.params.get("lr", 1e-3)),
                                weight_decay=float(self.params.get("weight_decay", 1e-2)))
        lossf = torch.nn.CrossEntropyLoss() if cls else torch.nn.HuberLoss(delta=1.0)
        rng = np.random.default_rng(self.seed)
        Xv = torch.tensor(self._windows(A, vi, self.L), device=self.dev)
        Yv = torch.tensor(yv[vi].astype(np.float32), device=self.dev)
        Yv = Yv.long() if cls else Yv.float()
        best, best_state, bad = np.inf, None, 0
        steps = int(self.params.get("max_steps", MAX_STEPS))
        for step in range(1, steps + 1):
            net.train()
            b = rng.choice(ti, size=min(BATCH, len(ti)), replace=False)
            xb = torch.tensor(self._windows(A, b, self.L), device=self.dev)
            yb = torch.tensor(yv[b].astype(np.float32), device=self.dev)
            out = net(xb)
            loss = lossf(out, yb.long()) if cls else lossf(out.squeeze(-1), yb.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % EVAL_EVERY == 0 or step == steps:
                net.eval()
                with torch.no_grad():
                    vl = 0.0
                    for i in range(0, len(vi), 2048):
                        o = net(Xv[i:i + 2048])
                        vl += float(lossf(o, Yv[i:i + 2048]) if cls else lossf(o.squeeze(-1), Yv[i:i + 2048])) \
                            * len(o)
                    vl /= max(1, len(vi))
                if vl < best - 1e-5:
                    best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.best_val_loss = best
        self.steps_run = step
        self.net = net.to("cpu")
        self.dev = _torch().device("cpu")

    def _raw(self, X, rows):
        torch = _torch()
        A = self._matrix(X)
        ends = np.flatnonzero(rows)
        out = []
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(ends), 2048):
                xb = torch.tensor(self._windows(A, ends[i:i + 2048], self.L))
                o = self.net(xb)
                out.append(torch.softmax(o, -1).numpy() if self.target.kind == "cls" else o.squeeze(-1).numpy())
        return np.concatenate(out) if out else np.zeros((0,))

    def __getstate__(self):
        d = super().__getstate__()
        d.pop("dev", None)
        return d
