"""④ 预算锦标赛 · Tier 2(风洞)—— 收纳 A1 跨尺度实验决策。

机制与锦标赛同构:花预算买"更贵更有信息的观测",选出最优。区别 = 观测轴是**尺度**:
你在**小尺度**跑便宜代理实验,真答案在**大尺度**评分,而**每局的小→大迁移可能反转**
(小尺度赢家在大尺度未必赢)。逼 agent 别信便宜代理、去更大尺度验证。litmus:小→大
映射每局隐藏且可反转 → 背 scaling law 没用,必须跑实验。
"""
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WindTunnelConfig:
    n_candidates: int
    scale_costs: list[int]   # 各尺度的成本(小尺度便宜、大尺度贵)
    budget: int
    human_budget: int
    regret_threshold: float
    seed: int


@dataclass
class RunResult:
    candidate: str
    scale: int
    value: float
    cost: int
    budget_remaining: int
    over_budget: bool


@dataclass
class CommitResult:
    score: float
    correct: bool
    regret: float
    chosen: str
    agent_steps: int


def default_windtunnel_config(seed: int = 0) -> WindTunnelConfig:
    return WindTunnelConfig(
        n_candidates=6, scale_costs=[1, 2, 4, 8, 16], budget=60,
        human_budget=40, regret_threshold=0.05, seed=seed,
    )


class WindTunnel:
    def __init__(self, config: WindTunnelConfig, budget_spent: int = 0):
        self.config = config
        self.budget_spent = budget_spent
        self._build()

    def _build(self):
        # 差异随尺度**涌现**:小尺度大家挤在 baseline 附近、噪声大(弱信息);大尺度才拉开真差距。
        # 逼 agent 用便宜小/中尺度粗筛、再花贵的大尺度确认 —— 信小尺度会被噪声骗。
        c = self.config
        r = random.Random(c.seed)
        M, K = c.n_candidates, len(c.scale_costs)
        ids = [chr(65 + i) for i in range(M)]
        baseline = 0.5
        large = {i: 0.4 + r.random() * 0.5 for i in ids}            # 大尺度真值(拉开)
        self.curves = {}
        for i in ids:
            row = []
            for k in range(K):
                w = k / (K - 1)                                     # 尺度权重 0→1(差异涌现)
                noise = r.gauss(0, 0.10 * (1 - w))                  # 小尺度噪声(压缩),大尺度趋 0
                v = large[i] * w + baseline * (1 - w) + noise
                row.append(max(0.05, min(0.99, v)))
            self.curves[i] = row
        self.ids = ids
        self.large = large
        self.best = max(large, key=large.get)
        self._K = K

    def value_at(self, candidate: str, scale: int) -> float:
        return self.curves[candidate][scale]

    def run(self, candidate: str, scale: int) -> RunResult:
        c = self.config
        cost = c.scale_costs[scale]
        remaining = c.budget - self.budget_spent
        if cost > remaining:
            return RunResult(candidate, scale, 0.0, 0, max(remaining, 0), over_budget=True)
        self.budget_spent += cost
        return RunResult(candidate, scale, self.value_at(candidate, scale), cost,
                         c.budget - self.budget_spent, over_budget=False)

    def commit(self, candidate: str) -> CommitResult:
        from architectureiq.scoring import rhae_ml_score
        best_large = max(self.large.values())
        regret = best_large - self.large[candidate]
        correct = regret <= self.config.regret_threshold
        score = rhae_ml_score(correct, self.budget_spent, self.config.human_budget)
        return CommitResult(score, correct, regret, candidate, self.budget_spent)
