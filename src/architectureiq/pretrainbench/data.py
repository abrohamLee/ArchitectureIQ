"""混合域 token shard 加载:按 data_mix 权重逐序列确定性采样域。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .config import DOMAINS, RunConfig


def make_test_shards(dir: Path, vocab: int, tokens_per_domain: int) -> None:
    """合成 shard:每个域占用不重叠的 token 区间,便于测试断言配比。"""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    for i, dom in enumerate(DOMAINS):
        rng = np.random.default_rng(1000 + i)
        lo = i * 100
        for split, n in (("train", tokens_per_domain), ("val", tokens_per_domain // 10)):
            toks = rng.integers(lo, lo + 100, size=n, dtype=np.int64)
            np.save(dir / f"{dom}_{split}.npy", toks)


class MixLoader:
    def __init__(self, shard_dir: Path, cfg: RunConfig, split: str) -> None:
        self.cfg = cfg
        self.split = split
        self.shards = {d: np.load(Path(shard_dir) / f"{d}_{split}.npy") for d in DOMAINS}
        self.rng = np.random.default_rng(cfg.seed * 7919 + (0 if split == "train" else 1))

    def _sample_seq(self) -> np.ndarray:
        L = self.cfg.preset.seq_len
        doms = list(DOMAINS)
        w = np.array([self.cfg.data_mix[d] for d in doms])
        dom = doms[int(self.rng.choice(len(doms), p=w / w.sum()))]
        shard = self.shards[dom]
        start = int(self.rng.integers(0, len(shard) - L - 1))
        return shard[start : start + L + 1]

    def batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        B = self.cfg.preset.batch_size
        while True:
            seqs = np.stack([self._sample_seq() for _ in range(B)])
            t = torch.from_numpy(seqs)
            yield t[:, :-1].contiguous(), t[:, 1:].contiguous()
