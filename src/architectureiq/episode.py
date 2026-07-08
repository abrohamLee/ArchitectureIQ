from dataclasses import dataclass

from architectureiq.datasets import DatasetSpec
from architectureiq.fingerprint import score_fingerprint
from architectureiq.scoring import rhae_ml_score


@dataclass(frozen=True)
class EpisodeConfig:
    archs: list[str]
    budget_steps: int
    probe_steps: int
    ref_seeds: list[int]
    query_seeds: list[int]
    commit_seeds: list[int]
    correct_margin: float
    human_steps: int
    data_seed: int


@dataclass
class ProbeResult:
    accuracy: float
    margin: float
    cost: int
    budget_remaining: int
    over_budget: bool


@dataclass
class CommitResult:
    score: float
    correct: bool
    margin: float
    agent_steps: int
    efficiency_capped: bool


def default_config() -> EpisodeConfig:
    archs = ["mlp", "tiny_transformer", "gru", "cnn1d"]
    probe_steps = 100
    ref_seeds = [0, 1, 2]
    query_seeds = [3, 4, 5]
    cost = probe_steps * len(archs) * (len(ref_seeds) + len(query_seeds)) * 2
    return EpisodeConfig(
        archs=archs,
        budget_steps=8000,
        probe_steps=probe_steps,
        ref_seeds=ref_seeds,
        query_seeds=query_seeds,
        commit_seeds=[10, 11, 12, 13, 14, 15],
        correct_margin=0.22,
        human_steps=cost * 2,
        data_seed=0,
    )


class Environment:
    def __init__(self, config: EpisodeConfig, budget_spent: int = 0, committed: bool = False):
        self.config = config
        self.budget_spent = budget_spent
        self.committed = committed

    def probe_cost(self) -> int:
        c = self.config
        return c.probe_steps * len(c.archs) * (len(c.ref_seeds) + len(c.query_seeds)) * 2

    def probe(self, spec: DatasetSpec) -> ProbeResult:
        c = self.config
        cost = self.probe_cost()
        remaining = c.budget_steps - self.budget_spent
        if cost > remaining:
            return ProbeResult(0.0, 0.0, 0, max(remaining, 0), over_budget=True)
        fp = score_fingerprint(
            spec, c.archs, ref_seeds=c.ref_seeds, query_seeds=c.query_seeds,
            steps=c.probe_steps, data_seed=c.data_seed,
        )
        self.budget_spent += cost
        return ProbeResult(
            accuracy=fp.accuracy, margin=fp.margin, cost=cost,
            budget_remaining=c.budget_steps - self.budget_spent, over_budget=False,
        )

    def commit(self, spec: DatasetSpec) -> CommitResult:
        c = self.config
        half = len(c.commit_seeds) // 2
        ref, query = c.commit_seeds[:half], c.commit_seeds[half:]
        fp = score_fingerprint(
            spec, c.archs, ref_seeds=ref, query_seeds=query,
            steps=c.probe_steps, data_seed=c.data_seed,
        )
        correct = fp.margin >= c.correct_margin
        agent_steps = self.budget_spent + self.probe_cost()
        score = rhae_ml_score(correct, agent_steps, c.human_steps)
        self.committed = True
        return CommitResult(
            score=score, correct=correct, margin=fp.margin,
            agent_steps=agent_steps, efficiency_capped=(c.human_steps >= agent_steps),
        )
