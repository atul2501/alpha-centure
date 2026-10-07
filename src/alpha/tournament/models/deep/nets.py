"""Deep models (small widths for an 8 GB M1): MLP, DNN, 1D-CNN, TCN, RNN, LSTM, GRU, BiLSTM, BiGRU.
The bidirectional RNNs read the past window in both directions; the window still ends at t (no future data)."""

from alpha.tournament.models.deep.torch_base import TorchModel, _torch


def _nn():
    return _torch().nn


class MLP(TorchModel):
    name, tabular = "mlp", True
    description = "2-layer MLP on the bar's feature row"

    def build(self, n_in, n_out, L):
        nn = _nn()
        h = int(self.params.get("hidden", 64))
        return nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Dropout(0.2), nn.Linear(h, h), nn.GELU(),
                             nn.Linear(h, n_out))


class DNN(TorchModel):
    name, tabular = "dnn", True
    description = "4-layer residual-free deep MLP with layer norm"

    def build(self, n_in, n_out, L):
        nn = _nn()
        h = int(self.params.get("hidden", 128))
        layers, d = [], n_in
        for _ in range(4):
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2)]
            d = h
        return nn.Sequential(*layers, nn.Linear(h, n_out))


def _rnn_net(kind: str, n_in: int, n_out: int, hidden: int, bidirectional: bool):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
            self.rnn = cell(n_in, hidden, batch_first=True, bidirectional=bidirectional)
            self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_out)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1])

    return Net()


class _RNN(TorchModel):
    kind, bidir = "lstm", False

    def build(self, n_in, n_out, L):
        return _rnn_net(self.kind, n_in, n_out, int(self.params.get("hidden", 32)), self.bidir)


class RNN(_RNN):
    name, kind = "rnn", "rnn"


class LSTM(_RNN):
    name, kind = "lstm", "lstm"


class GRU(_RNN):
    name, kind = "gru", "gru"


class BiLSTM(_RNN):
    name, kind, bidir = "bilstm", "lstm", True


class BiGRU(_RNN):
    name, kind, bidir = "bigru", "gru", True


class CNN1D(TorchModel):
    name = "cnn1d"

    def build(self, n_in, n_out, L):
        nn = _nn()
        h = int(self.params.get("hidden", 32))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Sequential(nn.Conv1d(n_in, h, 5, padding=2), nn.GELU(), nn.Conv1d(h, h, 5, padding=2),
                                       nn.GELU(), nn.AdaptiveAvgPool1d(1))
                self.head = nn.Linear(h, n_out)

            def forward(self, x):
                return self.head(self.c(x.transpose(1, 2)).squeeze(-1))

        return Net()


class TCN(TorchModel):
    name = "tcn"
    description = "temporal convolutional network: causal dilated residual blocks (dilations 1, 2, 4, 8, 16)"

    def build(self, n_in, n_out, L):
        torch = _torch()
        nn = torch.nn
        h = int(self.params.get("hidden", 32))

        class Block(nn.Module):
            def __init__(self, cin, d):
                super().__init__()
                self.pad = 2 * d
                self.c1 = nn.Conv1d(cin, h, 3, dilation=d)
                self.c2 = nn.Conv1d(h, h, 3, dilation=d)
                self.skip = nn.Conv1d(cin, h, 1) if cin != h else nn.Identity()

            def forward(self, x):
                y = torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0))))
                y = torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0))))
                return torch.relu(y + self.skip(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.Sequential(*[Block(n_in if i == 0 else h, 2**i) for i in range(5)])
                self.head = nn.Linear(h, n_out)

            def forward(self, x):
                return self.head(self.blocks(x.transpose(1, 2))[:, :, -1])

        return Net()
