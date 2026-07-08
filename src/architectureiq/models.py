import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyTransformer(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, d_model: int = 32, nhead: int = 4):
        super().__init__()
        self.in_dim = in_dim
        # 每个特征位置作为一个 token:标量 -> d_model
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(in_dim, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim) -> (batch, in_dim, 1)
        tokens = self.proj(x.unsqueeze(-1)) + self.pos
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class _GRU(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)  # (batch, in_dim, 1)
        _, h = self.gru(seq)
        return self.head(h.squeeze(0))


class _CNN1d(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, channels: int = 16):
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.act(self.conv(x.unsqueeze(1)))  # (batch, channels, in_dim)
        pooled = feat.mean(dim=2)
        return self.head(pooled)


def build_model(arch: str, in_dim: int, n_classes: int) -> nn.Module:
    if arch == "mlp":
        return _MLP(in_dim, n_classes)
    if arch == "tiny_transformer":
        return _TinyTransformer(in_dim, n_classes)
    if arch == "gru":
        return _GRU(in_dim, n_classes)
    if arch == "cnn1d":
        return _CNN1d(in_dim, n_classes)
    raise ValueError(f"unknown arch: {arch}")
