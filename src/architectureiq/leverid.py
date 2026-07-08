"""⑤ 黑盒鉴定 · Tier 2(真实杠杆)—— 收纳 A3 探针设计识别。

环境藏一个**真实杠杆的答案**(默认:优化器 ∈ {adam, sgd, rmsprop}),agent 设计探测
数据集,拿池中各答案的参考 Δ-签名 + 黑盒的 mystery 签名,识别它。与玩具黑盒同构、
共用 discriminator/scoring;区别只是签名从"架构"换成"杠杆"(levers.lever_signature)。
"""
from dataclasses import dataclass

from architectureiq.datasets import DatasetSpec
from architectureiq.discriminator import discriminate_signatures
from architectureiq.levers import lever_signature, lever_values
from architectureiq.scoring import rhae_ml_score


@dataclass(frozen=True)
class LeverIDConfig:
    family: str
    hidden_value: str
    ref_seeds: list[int]
    mystery_seed: int
    steps: int
    budget_steps: int
    human_steps: int
    data_seed: int


@dataclass
class ProbeResult:
    references: dict[str, list[float]]
    mystery: list[float]
    cost: int
    budget_remaining: int
    over_budget: bool


@dataclass
class GuessResult:
    score: float
    correct: bool
    chosen: str
    hidden: str
    agent_steps: int


def default_leverid_config(family: str = "optimizer", hidden_value: str = "sgd") -> LeverIDConfig:
    ref_seeds = [1, 2]
    steps = 80
    pool = lever_values(family)
    cost = steps * (len(pool) * len(ref_seeds) + 1) * 2
    return LeverIDConfig(
        family=family, hidden_value=hidden_value, ref_seeds=ref_seeds, mystery_seed=7,
        steps=steps, budget_steps=4000, human_steps=cost, data_seed=0,
    )


class LeverIDEpisode:
    def __init__(self, config: LeverIDConfig, budget_spent: int = 0):
        self.config = config
        self.budget_spent = budget_spent
        self.pool = lever_values(config.family)

    def probe_cost(self) -> int:
        c = self.config
        return c.steps * (len(self.pool) * len(c.ref_seeds) + 1) * 2

    def _sig(self, value: str, spec: DatasetSpec, seed: int) -> list[float]:
        return lever_signature(self.config.family, value, spec, seed=seed,
                               steps=self.config.steps, data_seed=self.config.data_seed)

    def probe(self, spec: DatasetSpec) -> ProbeResult:
        c = self.config
        cost = self.probe_cost()
        remaining = c.budget_steps - self.budget_spent
        if cost > remaining:
            return ProbeResult({}, [], 0, max(remaining, 0), over_budget=True)
        references: dict[str, list[float]] = {}
        for v in self.pool:
            sigs = [self._sig(v, spec, s) for s in c.ref_seeds]
            references[v] = [sum(z) / len(z) for z in zip(*sigs)]
        mystery = self._sig(c.hidden_value, spec, c.mystery_seed)
        self.budget_spent += cost
        return ProbeResult(references, mystery, cost,
                           c.budget_steps - self.budget_spent, over_budget=False)

    def guess(self, value: str) -> GuessResult:
        c = self.config
        correct = value == c.hidden_value
        score = rhae_ml_score(correct, self.budget_spent, c.human_steps)
        return GuessResult(score, correct, value, c.hidden_value, self.budget_spent)
