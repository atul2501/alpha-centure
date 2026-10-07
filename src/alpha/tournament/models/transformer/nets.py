"""Transformer-family and linear forecasters, compact PyTorch implementations sized for an 8 GB M1.

Faithful to each architecture's core idea, not to its full published configuration:
    VanillaTransformer   encoder over time steps, last-token head
    TemporalTransformer  causal-masked encoder + learned time-of-day embedding channel
    PatchTST             channel-independent patches (patch 8, stride 4) -> shared encoder -> flatten head
    DLinear              trend (moving average) + remainder, one linear map each over time
    NLinear              subtract last value, linear over time
    iTransformer         tokens = variables (each channel's whole window embedded), attention across variables
    TFTLite              variable selection (GRN softmax gates) + LSTM + one masked self-attention layer
    InformerLite         ProbSparse-style top-u query attention + distilling conv/max-pool between layers
    AutoformerLite       series decomposition blocks + FFT auto-correlation (top-k lags) instead of attention
    FEDformerLite        frequency-enhanced block: attention-free mixing of a fixed random set of Fourier modes
    TimesNetLite         FFT top periods -> fold 1D into 2D -> inception-style 2D conv -> aggregate
    CrossformerLite      dimension-segment-wise embedding + two-stage (time, then dimension) attention
All but the first five are marked experimental.
"""

import math

from alpha.tournament.models.deep.torch_base import TorchModel, _torch

D = 32


def _enc(d=D, heads=4, layers=2, ff=64):
    nn = _torch().nn
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True,
                                                            norm_first=True), layers, enable_nested_tensor=False)


class _T(TorchModel):
    family = "transformer"
    complexity = 4


class VanillaTransformer(_T):
    name = "transformer"

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)
                self.enc = _enc()
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                return self.head(self.enc(self.inp(x) + self.pos)[:, -1])

        return Net()


class TemporalTransformer(_T):
    name = "temporal_transformer"

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)
                self.enc = _enc()
                self.head = nn.Linear(D, n_out)
                self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

            def forward(self, x):
                h = self.enc(self.inp(x) + self.pos, mask=self.mask[: x.shape[1], : x.shape[1]])
                return self.head(h[:, -1])

        return Net()


class PatchTST(_T):
    name = "patchtst"

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        P, S = 8, 4
        n_p = (L - P) // S + 1

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Linear(P, D)
                self.pos = nn.Parameter(torch.randn(1, n_p, D) * 0.02)
                self.enc = _enc()
                self.head = nn.Sequential(nn.Flatten(1), nn.Linear(n_in * n_p * D, n_out))

            def forward(self, x):                      # x (B, L, C)
                B, L_, C = x.shape
                p = x.transpose(1, 2).unfold(2, P, S)  # (B, C, n_p, P)
                z = self.emb(p).reshape(B * C, n_p, D) + self.pos
                z = self.enc(z).reshape(B, C, n_p, D)
                return self.head(z)

        return Net()


class DLinear(_T):
    name = "dlinear"
    complexity = 2  # linear in its parameters

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        k = min(25, L - (1 - L % 2))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.avg = nn.AvgPool1d(k, 1, padding=k // 2, count_include_pad=False)
                self.lt = nn.Linear(L, 1)
                self.ls = nn.Linear(L, 1)
                self.head = nn.Linear(n_in, n_out)

            def forward(self, x):
                xt = x.transpose(1, 2)
                trend = self.avg(xt)[:, :, : xt.shape[2]]
                z = (self.lt(trend) + self.ls(xt - trend)).squeeze(-1)
                return self.head(z)

        return Net()


class NLinear(_T):
    name = "nlinear"
    complexity = 2

    def build(self, n_in, n_out, L):
        nn = _torch().nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(L, 1)
                self.head = nn.Linear(2 * n_in, n_out)

            def forward(self, x):
                last = x[:, -1:, :]
                z = self.l((x - last).transpose(1, 2)).squeeze(-1)
                return self.head(_torch().cat([z, last.squeeze(1)], -1))

        return Net()


class ITransformer(_T):
    name = "itransformer"

    def build(self, n_in, n_out, L):
        nn = _torch().nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Linear(L, D)
                self.enc = _enc()
                self.head = nn.Sequential(nn.Flatten(1), nn.Linear(n_in * D, n_out))

            def forward(self, x):
                return self.head(self.enc(self.emb(x.transpose(1, 2))))

        return Net()


class TFTLite(_T):
    name, experimental = "tft", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn

        class GRN(nn.Module):
            def __init__(self, din, dout):
                super().__init__()
                self.a = nn.Linear(din, dout)
                self.b = nn.Linear(dout, dout)
                self.g = nn.Linear(dout, 2 * dout)
                self.skip = nn.Linear(din, dout)
                self.n = nn.LayerNorm(dout)

            def forward(self, x):
                h = self.b(nn.functional.elu(self.a(x)))
                return self.n(self.skip(x) + nn.functional.glu(self.g(h)))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.var_emb = nn.Linear(1, D)
                self.sel = GRN(n_in, n_in)
                self.lstm = nn.LSTM(D, D, batch_first=True)
                self.att = nn.MultiheadAttention(D, 4, batch_first=True)
                self.post = GRN(D, D)
                self.head = nn.Linear(D, n_out)
                self.register_buffer("mask", torch.triu(torch.ones(L, L, dtype=torch.bool), 1))

            def forward(self, x):                                   # (B, L, C)
                w = torch.softmax(self.sel(x), -1).unsqueeze(-1)    # variable selection weights
                z = (self.var_emb(x.unsqueeze(-1)) * w).sum(2)      # (B, L, D)
                h, _ = self.lstm(z)
                a, _ = self.att(h, h, h, attn_mask=self.mask[: x.shape[1], : x.shape[1]])
                return self.head(self.post(a + h)[:, -1])

        return Net()


class InformerLite(_T):
    name, experimental = "informer", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn

        class ProbAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(D, 3 * D)
                self.o = nn.Linear(D, D)

            def forward(self, x):
                B, T, _ = x.shape
                q, k, v = self.qkv(x).chunk(3, -1)
                scores = q @ k.transpose(1, 2) / math.sqrt(D)
                u = max(1, int(5 * math.log(T + 1)))
                sparsity = scores.max(-1).values - scores.mean(-1)          # query "activeness"
                top = sparsity.topk(min(u, T), dim=-1).indices
                out = v.mean(1, keepdim=True).expand(B, T, D).clone()       # lazy queries -> mean of V
                sel = torch.gather(scores, 1, top.unsqueeze(-1).expand(-1, -1, T))
                upd = torch.softmax(sel, -1) @ v
                out.scatter_(1, top.unsqueeze(-1).expand(-1, -1, D), upd)
                return self.o(out)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.att = ProbAttention()
                self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
                self.ff = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, D))
                self.distil = nn.Sequential(nn.Conv1d(D, D, 3, padding=1), nn.ELU(), nn.MaxPool1d(3, 2, 1))

            def forward(self, x):
                x = self.n1(x + self.att(x))
                x = self.n2(x + self.ff(x))
                return self.distil(x.transpose(1, 2)).transpose(1, 2)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)
                self.layers = nn.Sequential(Layer(), Layer())
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                return self.head(self.layers(self.inp(x) + self.pos)[:, -1])

        return Net()


def _decomp(torch, x, k=13):
    xt = x.transpose(1, 2)
    pad = torch.nn.functional.pad(xt, (k - 1, 0), mode="replicate")  # causal moving average
    trend = torch.nn.functional.avg_pool1d(pad, k, 1).transpose(1, 2)
    return x - trend, trend


class AutoformerLite(_T):
    name, experimental = "autoformer", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn

        class AutoCorr(nn.Module):
            def __init__(self):
                super().__init__()
                self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))

            def forward(self, x):
                B, T, _ = x.shape
                q, k, v = self.q(x), self.k(x), self.v(x)
                corr = torch.fft.irfft(torch.fft.rfft(q, dim=1) * torch.conj(torch.fft.rfft(k, dim=1)), n=T, dim=1)
                c = corr.mean(-1)                                     # (B, T) correlation per lag
                topk = max(1, int(math.log(T)))
                w, lags = c.topk(topk, dim=-1)
                w = torch.softmax(w, -1)
                out = torch.zeros_like(v)
                for i in range(topk):
                    rolled = torch.stack([torch.roll(v[b], int(lags[b, i]), 0) for b in range(B)])
                    out = out + rolled * w[:, i].view(B, 1, 1)
                return self.o(out)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.ac = AutoCorr()
                self.ff = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, D))
                self.trend = nn.Linear(D, D)
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                h = self.inp(x)
                s, t1 = _decomp(torch, h + self.ac(h))
                s, t2 = _decomp(torch, s + self.ff(s))
                return self.head(s[:, -1] + self.trend(t1 + t2)[:, -1])

        return Net()


class FEDformerLite(_T):
    name, experimental = "fedformer", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        n_freq = L // 2 + 1
        modes = sorted(torch.randperm(n_freq, generator=torch.Generator().manual_seed(self.seed))[: min(16, n_freq)]
                       .tolist())

        class FEB(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("modes", torch.tensor(modes))
                self.wr = nn.Parameter(torch.randn(len(modes), D, D) / D)
                self.wi = nn.Parameter(torch.randn(len(modes), D, D) / D)

            def forward(self, x):
                f = torch.fft.rfft(x, dim=1)                         # (B, F, D)
                sel = f[:, self.modes]                                # (B, M, D)
                w = torch.complex(self.wr, self.wi)
                mixed = torch.einsum("bmd,mde->bme", sel, w)
                out = torch.zeros_like(f)
                out[:, self.modes] = mixed
                return torch.fft.irfft(out, n=x.shape[1], dim=1)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.feb = FEB()
                self.ff = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, D))
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                h = self.inp(x)
                s, t1 = _decomp(torch, h + self.feb(h))
                s, t2 = _decomp(torch, s + self.ff(s))
                return self.head((s + t1 + t2)[:, -1])

        return Net()


class TimesNetLite(_T):
    name, experimental = "timesnet", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.ModuleList([nn.Conv2d(D, D, k, padding=k // 2) for k in (1, 3, 5)])

                self.register_buffer("periods", torch.tensor([2, 4]))

            def forward(self, x):
                B, T, _ = x.shape
                amp_s = torch.fft.rfft(x, dim=1).abs().mean(-1)        # per-sample amplitude (B, F)
                if self.training:  # periods are learned from training batches only, then frozen: at inference a
                    a = amp_s.mean(0).detach().clone()                 # row's output never depends on other rows
                    a[0] = 0
                    self.periods = a.topk(2).indices.clamp(min=1)
                top = self.periods
                outs, ws = [], []
                for f in top.tolist():
                    p = max(2, T // f)
                    n = math.ceil(T / p) * p
                    z = nn.functional.pad(x, (0, 0, 0, n - T)).reshape(B, n // p, p, D).permute(0, 3, 1, 2)
                    z = sum(c(z) for c in self.conv) / 3
                    outs.append(nn.functional.gelu(z).permute(0, 2, 3, 1).reshape(B, n, D)[:, :T])
                    ws.append(amp_s[:, f])
                w = torch.softmax(torch.stack(ws, -1), -1)                # (B, 2)
                return x + sum(o * w[:, i].view(B, 1, 1) for i, o in enumerate(outs))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.blocks = nn.Sequential(Block(), Block())
                self.n = nn.LayerNorm(D)
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                return self.head(self.n(self.blocks(self.inp(x)))[:, -1])

        return Net()


class CrossformerLite(_T):
    name, experimental = "crossformer", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        seg = 8
        n_seg = L // seg

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Linear(seg, D)
                self.pos = nn.Parameter(torch.randn(1, n_in, n_seg, D) * 0.02)
                self.time_att = _enc(layers=1)
                self.dim_att = _enc(layers=1)
                self.head = nn.Sequential(nn.Flatten(1), nn.Linear(n_in * D, n_out))

            def forward(self, x):                                     # (B, L, C)
                B = x.shape[0]
                z = x[:, -n_seg * seg:].transpose(1, 2).reshape(B, n_in, n_seg, seg)
                z = self.emb(z) + self.pos                            # (B, C, S, D)
                z = self.time_att(z.reshape(B * n_in, n_seg, D)).reshape(B, n_in, n_seg, D)
                z = z[:, :, -1]                                       # last segment per variable
                return self.head(self.dim_att(z))

        return Net()
