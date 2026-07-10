"""PretrainBench run 配置:决策轴 + 尺度预设 + 内容哈希。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScalePreset:
    n_layer: int
    n_head: int
    n_embd: int
    seq_len: int
    batch_size: int
    total_steps: int
    vocab_size: int = 512  # test 档用小词表;130m 档由真实 tokenizer 决定,V0 先统一


SCALES: dict[str, ScalePreset] = {
    # ~1M 参数,CPU 秒级,专供测试与本地 TDD
    "test": ScalePreset(n_layer=2, n_head=2, n_embd=64, seq_len=64,
                        batch_size=8, total_steps=30),
    # ~130M 参数(GPT-2 medium 减配),chinchilla-optimal 附近
    "130m": ScalePreset(n_layer=12, n_head=12, n_embd=768, seq_len=1024,
                        batch_size=256, total_steps=10000, vocab_size=50304),
}

DECAYS = ("cosine", "linear", "constant")
DOMAINS = ("web", "code", "math")


@dataclass
class RunConfig:
    scale: str
    peak_lr: float
    warmup_frac: float          # 决策轴 1:warmup 步数 = warmup_frac * total_steps
    decay: str
    data_mix: dict[str, float]  # 决策轴 2:域配比,键必须是 DOMAINS
    seed: int

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise ValueError(f"unknown scale {self.scale}")
        if self.decay not in DECAYS:
            raise ValueError(f"unknown decay {self.decay}")
        if set(self.data_mix) != set(DOMAINS):
            raise ValueError(f"data_mix keys must be {DOMAINS}")
        if abs(sum(self.data_mix.values()) - 1.0) > 1e-6:
            raise ValueError("data_mix must sum to 1")
        if not (0.0 <= self.warmup_frac < 1.0):
            raise ValueError("warmup_frac must be in [0,1)")

    @property
    def preset(self) -> ScalePreset:
        return SCALES[self.scale]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(**d)


def config_hash(cfg: RunConfig) -> str:
    """内容哈希,标识'同一个实验点'。seed 除外(同点多 seed 归并)。"""
    d = cfg.to_dict()
    d.pop("seed")
    blob = json.dumps(d, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
