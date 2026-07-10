"""极简 GPT-2 风格因果 LM。V0 两个决策轴不动架构,尺寸全由 ScalePreset 决定。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ScalePreset


class Block(nn.Module):
    def __init__(self, p: ScalePreset) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(p.n_embd)
        self.attn = nn.MultiheadAttention(p.n_embd, p.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(p.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(p.n_embd, 4 * p.n_embd), nn.GELU(), nn.Linear(4 * p.n_embd, p.n_embd)
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h),
                         attn_mask=mask, need_weights=False)
        h = h + a
        return h + self.mlp(self.ln2(h))


class GPT(nn.Module):
    def __init__(self, preset: ScalePreset) -> None:
        super().__init__()
        self.preset = preset
        self.tok = nn.Embedding(preset.vocab_size, preset.n_embd)
        self.pos = nn.Embedding(preset.seq_len, preset.n_embd)
        self.blocks = nn.ModuleList(Block(preset) for _ in range(preset.n_layer))
        self.ln_f = nn.LayerNorm(preset.n_embd)
        self.head = nn.Linear(preset.n_embd, preset.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_f(h))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
