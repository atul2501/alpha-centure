"""State-space sequence models in pure PyTorch (mamba-ssm needs CUDA; this machine has an M1).

    S4D             diagonal state space layer (S4D-Lin init), kernel computed explicitly, FFT convolution
    S5              MIMO diagonal SSM with a shared state across channels, discretized (ZOH), sequential scan
    Mamba           selective SSM block: input-dependent (Delta, B, C), depthwise causal conv, gated output,
                    sequential selective scan (L <= 192 so a Python-level scan is affordable)
    Mamba2          SSD-style head: scalar A per head and input-dependent dt, B, C shared within a head
    MambaAttention  Mamba block followed by one self-attention layer
    MambaCNN        causal conv stack followed by a Mamba block
    MambaTransformer Mamba block interleaved with a transformer encoder layer
"""

import math

from alpha.tournament.models.deep.torch_base import TorchModel, _torch
from alpha.tournament.models.transformer.nets import D, _enc


class _SSM(TorchModel):
    family = "ssm"
    complexity = 4


def _mamba_block(d=D, n_state=16, expand=2, conv=4):
    torch = _torch()
    nn = torch.nn

    class Mamba(nn.Module):
        def __init__(self):
            super().__init__()
            di = expand * d
            self.inp = nn.Linear(d, 2 * di)
            self.conv = nn.Conv1d(di, di, conv, groups=di, padding=conv - 1)
            self.x_proj = nn.Linear(di, n_state * 2 + 1)
            self.dt_proj = nn.Linear(1, di)
            A = torch.arange(1, n_state + 1).float().repeat(di, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.Dskip = nn.Parameter(torch.ones(di))
            self.out = nn.Linear(di, d)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):                                  # (B, L, d)
            B_, L, _ = x.shape
            xz = self.inp(self.norm(x))
            u, z = xz.chunk(2, -1)
            u = nn.functional.silu(self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
            p = self.x_proj(u)
            dt = nn.functional.softplus(self.dt_proj(p[..., :1]))  # (B, L, di)
            Bm, Cm = p[..., 1:1 + n_state], p[..., 1 + n_state:]
            A = -torch.exp(self.A_log)                             # (di, n)
            h = torch.zeros(B_, u.shape[-1], n_state, device=x.device)
            ys = []
            for t in range(L):                                      # selective scan
                dA = torch.exp(dt[:, t, :, None] * A)
                h = dA * h + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
                ys.append((h * Cm[:, t, None, :]).sum(-1))
            y = torch.stack(ys, 1) + u * self.Dskip
            return x + self.out(y * nn.functional.silu(z))

    return Mamba()


class Mamba(_SSM):
    name = "mamba"

    def build(self, n_in, n_out, L):
        nn = _torch().nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.blocks = nn.Sequential(_mamba_block(), _mamba_block())
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                return self.head(self.blocks(self.inp(x))[:, -1])

        return Net()


class Mamba2(_SSM):
    name, experimental = "mamba2", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        H, P, N = 4, 16, 16  # heads, head dim, state

        class SSD(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(D, H * P + 2 * N + H)
                self.A_log = nn.Parameter(torch.zeros(H))
                self.out = nn.Linear(H * P, D)
                self.norm = nn.LayerNorm(D)

            def forward(self, x):
                B_, L_, _ = x.shape
                p = self.inp(self.norm(x))
                u = p[..., : H * P].reshape(B_, L_, H, P)
                Bm, Cm = p[..., H * P:H * P + N], p[..., H * P + N:H * P + 2 * N]
                dt = nn.functional.softplus(p[..., -H:])          # (B, L, H)
                a = -torch.exp(self.A_log)                         # scalar decay per head
                h = torch.zeros(B_, H, P, N, device=x.device)
                ys = []
                for t in range(L_):
                    dA = torch.exp(dt[:, t] * a)[:, :, None, None]
                    h = dA * h + (dt[:, t, :, None, None] * u[:, t, :, :, None] * Bm[:, t, None, None, :])
                    ys.append((h * Cm[:, t, None, None, :]).sum(-1).reshape(B_, H * P))
                return x + self.out(torch.stack(ys, 1))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.blocks = nn.Sequential(SSD(), SSD())
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                return self.head(self.blocks(self.inp(x))[:, -1])

        return Net()


class S4D(_SSM):
    name = "s4"

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        N = 32

        class S4DLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.log_dt = nn.Parameter(torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
                self.A_re = nn.Parameter(torch.full((D, N // 2), -0.5))
                self.A_im = nn.Parameter(math.pi * torch.arange(N // 2).float().repeat(D, 1))
                self.C = nn.Parameter(torch.randn(D, N // 2, 2) * 0.5)
                self.Dskip = nn.Parameter(torch.randn(D))
                self.out = nn.Linear(D, D)
                self.norm = nn.LayerNorm(D)

            def forward(self, x):                                       # (B, L, D)
                L_ = x.shape[1]
                dt = torch.exp(self.log_dt)[:, None]
                A = torch.complex(-torch.exp(self.A_re), self.A_im)     # (D, N/2)
                C = torch.view_as_complex(self.C) * (torch.exp(dt * A) - 1) / A
                k = 2 * torch.einsum("dn,dnl->dl", C, torch.exp(dt[..., None] * A[..., None]
                                                                * torch.arange(L_, device=x.device))).real
                u = self.norm(x).transpose(1, 2)
                y = torch.fft.irfft(torch.fft.rfft(u, n=2 * L_) * torch.fft.rfft(k, n=2 * L_), n=2 * L_)[..., :L_]
                y = y + u * self.Dskip[:, None]
                return x + self.out(nn.functional.gelu(y.transpose(1, 2)))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.blocks = nn.Sequential(S4DLayer(), S4DLayer())
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                return self.head(self.blocks(self.inp(x))[:, -1])

        return Net()


class S5(_SSM):
    name, experimental = "s5", True

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        P = 32

        class S5Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.Lre = nn.Parameter(torch.full((P,), -0.5))
                self.Lim = nn.Parameter(math.pi * torch.arange(P).float())
                self.B = nn.Parameter(torch.randn(P, D, 2) / math.sqrt(2 * D))
                self.C = nn.Parameter(torch.randn(D, P, 2) / math.sqrt(P))
                self.log_dt = nn.Parameter(torch.full((P,), math.log(0.01)))
                self.Dskip = nn.Parameter(torch.ones(D))
                self.norm = nn.LayerNorm(D)

            def forward(self, x):
                Lam = torch.complex(-torch.exp(self.Lre), self.Lim)
                dt = torch.exp(self.log_dt)
                Lb = torch.exp(Lam * dt)
                Bb = ((Lb - 1) / Lam)[:, None] * torch.view_as_complex(self.B)  # ZOH
                C = torch.view_as_complex(self.C)
                u = self.norm(x)
                h = torch.zeros(x.shape[0], P, dtype=torch.complex64, device=x.device)
                ys = []
                for t in range(x.shape[1]):
                    h = Lb * h + u[:, t].to(torch.complex64) @ Bb.T
                    ys.append((h @ C.T).real)
                return x + nn.functional.gelu(torch.stack(ys, 1) + u * self.Dskip)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(n_in, D)
                self.blocks = nn.Sequential(S5Layer(), S5Layer())
                self.head = nn.Linear(D, n_out)

            def forward(self, x):
                return self.head(self.blocks(self.inp(x))[:, -1])

        return Net()


def _hybrid(kind):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.inp = nn.Linear(n_in, D)
            if kind == "attention":
                self.body = nn.ModuleList([_mamba_block(), nn.MultiheadAttention(D, 4, batch_first=True)])
            elif kind == "cnn":
                self.conv = nn.Sequential(nn.Conv1d(D, D, 5, padding=4), nn.GELU())
                self.body = nn.ModuleList([_mamba_block()])
            else:
                self.body = nn.ModuleList([_mamba_block(), _enc(layers=1), _mamba_block()])
            self.head = nn.Linear(D, n_out)

        def forward(self, x):
            h = self.inp(x)
            if kind == "attention":
                h = self.body[0](h)
                a, _ = self.body[1](h, h, h)
                h = h + a
            elif kind == "cnn":
                h = h + self.conv(h.transpose(1, 2))[:, :, : h.shape[1]].transpose(1, 2)
                h = self.body[0](h)
            else:
                for b in self.body:
                    h = b(h)
            return self.head(h[:, -1])

    return Net


class MambaAttention(_SSM):
    name, experimental = "mamba_attention", True

    def build(self, n_in, n_out, L):
        return _hybrid("attention")(n_in, n_out)


class MambaCNN(_SSM):
    name, experimental = "mamba_cnn", True

    def build(self, n_in, n_out, L):
        return _hybrid("cnn")(n_in, n_out)


class MambaTransformer(_SSM):
    name, experimental = "mamba_transformer", True

    def build(self, n_in, n_out, L):
        return _hybrid("transformer")(n_in, n_out)
