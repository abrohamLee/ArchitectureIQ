import math
from dataclasses import dataclass

from architectureiq.datasets import DatasetSpec, generate
from architectureiq.fingerprint import structure_advantage
from architectureiq.scoring import rhae_ml_score


@dataclass(frozen=True)
class BlackboxConfig:
    pool: list[str]
    hidden_arch: str
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


def nearest(mystery: list[float], references: dict[str, list[float]]) -> str:
    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return min(references, key=lambda a: dist(mystery, references[a]))


def default_blackbox_config(hidden_arch: str = "mlp") -> BlackboxConfig:
    pool = ["mlp", "tiny_transformer", "gru"]
    ref_seeds = [1, 2]
    steps = 80
    cost = steps * (len(pool) * len(ref_seeds) + 1) * 2
    return BlackboxConfig(
        pool=pool, hidden_arch=hidden_arch, ref_seeds=ref_seeds, mystery_seed=7,
        steps=steps, budget_steps=4000, human_steps=cost, data_seed=0,
    )


class BlackboxEpisode:
    def __init__(self, config: BlackboxConfig, budget_spent: int = 0):
        self.config = config
        self.budget_spent = budget_spent

    def probe_cost(self) -> int:
        c = self.config
        return c.steps * (len(c.pool) * len(c.ref_seeds) + 1) * 2

    def _sig(self, arch: str, spec: DatasetSpec, seed: int) -> list[float]:
        X, y, in_dim, n_classes = generate(spec, seed=self.config.data_seed)
        return structure_advantage(arch, X, y, in_dim, n_classes,
                                   steps=self.config.steps, seed=seed)

    def probe(self, spec: DatasetSpec) -> ProbeResult:
        c = self.config
        cost = self.probe_cost()
        remaining = c.budget_steps - self.budget_spent
        if cost > remaining:
            return ProbeResult({}, [], 0, max(remaining, 0), over_budget=True)
        references: dict[str, list[float]] = {}
        for arch in c.pool:
            sigs = [self._sig(arch, spec, s) for s in c.ref_seeds]
            references[arch] = [sum(v) / len(v) for v in zip(*sigs)]
        mystery = self._sig(c.hidden_arch, spec, c.mystery_seed)
        self.budget_spent += cost
        return ProbeResult(references, mystery, cost,
                           c.budget_steps - self.budget_spent, over_budget=False)

    def guess(self, arch: str) -> GuessResult:
        c = self.config
        correct = arch == c.hidden_arch
        score = rhae_ml_score(correct, self.budget_spent, c.human_steps)
        return GuessResult(score, correct, arch, c.hidden_arch, self.budget_spent)
