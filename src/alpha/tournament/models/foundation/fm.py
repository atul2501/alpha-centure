"""Time-series foundation models, zero-shot (EXPERIMENTAL). Nothing is fitted on SOL data: each bar's forecast
uses the last CONTEXT closes up to and including bar t, so the forecasts are causal by construction and are
cached per bar (data/research/sol15/fm_<model>.parquet) to be shared by every fold and target.

score_t = 1e4 * (median forecast of close_{t+h} / close_t - 1)   (bps; the shared policy then applies)

Runnable here: Chronos (T5 small), Chronos-Bolt (small), TimesFM 2.5 (200M).
Not runnable in this environment (logged as not_run): Moirai (uni2ts needs torch 2.4 / numpy < 2), Lag-Llama
(gluonts + numpy < 2), MOMENT (momentfm pins transformers 4.33, conflicting with Chronos' transformers 5).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from alpha.tournament.models.base import BaseTradingModel

CONTEXT = 512
HORIZONS = (1, 2, 4, 8, 16)
CACHE = Path("data/research/sol15")
BATCH = 256


class NotRunnable(RuntimeError):
    pass


class _FM(BaseTradingModel):
    family = "foundation"
    complexity = 4
    input_kind = "series"
    target_kinds = ("reg",)
    output_units = "bps"
    experimental = True
    model_id = ""

    def fit(self, X, y, train, val):
        return self

    def _forecast(self, closes: np.ndarray) -> np.ndarray:
        """closes (B, CONTEXT) -> median forecast (B, 16)."""
        raise NotImplementedError

    def _cache_path(self):
        tag = self.context.dataset_id if hasattr(self, "context") else "na"
        return CACHE / f"fm_{self.name}_{tag}.parquet"

    def _raw(self, X, rows):
        close = self.context.raw["sol_close"].to_numpy(float)[: len(X)]
        idx = X.index
        path = self._cache_path()
        cache = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=[f"h{h}" for h in HORIZONS])
        need = idx[rows].difference(cache.index)
        if len(need):
            pos = idx.get_indexer(need)
            new = []
            for i in range(0, len(pos), BATCH):
                p = pos[i:i + BATCH]
                ctx = np.stack([_window(close, j) for j in p])
                fc = self._forecast(ctx)
                new.append(pd.DataFrame({f"h{h}": 1e4 * (fc[:, h - 1] / close[p] - 1) for h in HORIZONS},
                                        index=need[i:i + BATCH]))
            cache = pd.concat([cache, *new]).sort_index()
            cache = cache[~cache.index.duplicated()]
            path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(path)
        return cache.reindex(idx[rows])[f"h{self.target.h}"].to_numpy()


def _window(close: np.ndarray, j: int) -> np.ndarray:
    lo = j - CONTEXT + 1
    w = close[max(0, lo): j + 1]
    if lo < 0:
        w = np.concatenate([np.full(-lo, w[0]), w])
    return np.nan_to_num(w, nan=float(np.nanmean(w)))


_PIPES: dict = {}


class ChronosBolt(_FM):
    name, model_id = "chronos_bolt", "amazon/chronos-bolt-small"

    def _pipe(self):
        if self.model_id not in _PIPES:
            import torch
            from chronos import BaseChronosPipeline

            _PIPES[self.model_id] = BaseChronosPipeline.from_pretrained(self.model_id, device_map="cpu",
                                                                        torch_dtype=torch.float32)
        return _PIPES[self.model_id]

    def _forecast(self, closes):
        import torch

        q, mean = self._pipe().predict_quantiles(torch.tensor(closes, dtype=torch.float32), prediction_length=16,
                                                 quantile_levels=[0.5])
        return q[:, :, 0].numpy()


class Chronos(ChronosBolt):
    name, model_id = "chronos", "amazon/chronos-t5-small"


class TimesFM(_FM):
    name, model_id = "timesfm", "google/timesfm-2.5-200m-pytorch"

    def _forecast(self, closes):
        import timesfm

        if self.model_id not in _PIPES:
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
            m.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=16, normalize_inputs=True,
                                             per_core_batch_size=BATCH))
            _PIPES[self.model_id] = m
        point, _ = _PIPES[self.model_id].forecast(horizon=16, inputs=list(closes))
        return np.asarray(point)[:, :16]


class _Unavailable(_FM):
    reason = ""

    def fit(self, X, y, train, val):
        raise NotRunnable(self.reason)


class Moirai(_Unavailable):
    name = "moirai"
    reason = "uni2ts requires torch 2.4.x and numpy < 2 (project env: torch 2.14, numpy 2.5); needs a separate venv"


class LagLlama(_Unavailable):
    name = "lag_llama"
    reason = "lag-llama requires gluonts 0.14 with numpy < 2 (project env numpy 2.5); needs a separate venv"


class Moment(_Unavailable):
    name = "moment"
    reason = "momentfm pins transformers 4.33, conflicting with chronos-forecasting (transformers 5)"
