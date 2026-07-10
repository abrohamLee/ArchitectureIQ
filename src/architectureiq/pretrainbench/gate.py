"""题目显著性闸门(spec §7):seed 方差盖不住选项差异的配置对不能成题。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from statistics import mean


@dataclass
class GateResult:
    passed: bool
    gap: float
    win_rate: float
    better: str  # "a" | "b"


def pair_gate(a: list[float], b: list[float],
              min_gap: float = 0.05, min_win_rate: float = 0.9) -> GateResult:
    ma, mb = mean(a), mean(b)
    better = "a" if ma < mb else "b"
    lo, hi = (a, b) if better == "a" else (b, a)
    wins = sum(x < y for x, y in product(lo, hi))
    win_rate = wins / (len(lo) * len(hi))
    gap = abs(ma - mb) / max(abs(ma), abs(mb), 1e-9)
    return GateResult(passed=(gap >= min_gap and win_rate >= min_win_rate),
                      gap=gap, win_rate=win_rate, better=better)
