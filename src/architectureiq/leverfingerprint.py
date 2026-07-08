"""① 架构指纹 · Tier 2(真实杠杆)—— 设计数据集分开一组真实杠杆的答案。

玩具指纹分开 4 个架构;这里分开一个**真实杠杆家族**的答案(优化器 {adam,sgd,rmsprop}
或激活 {relu,tanh,gelu})。agent 设计数据集,让这些杠杆的结构优势 Δ-签名最大程度可区分。
复用判别器 + 结构优势;签名从"架构"换成"杠杆"(levers.lever_signature)。
"""
from dataclasses import dataclass

from architectureiq.datasets import DatasetSpec
from architectureiq.discriminator import discriminate_signatures
from architectureiq.fingerprint import FingerprintScore
from architectureiq.levers import lever_signature, lever_values
from architectureiq.scoring import rhae_ml_score


def score_lever_fingerprint(family: str, spec: DatasetSpec, ref_seeds: list[int],
                            query_seeds: list[int], steps: int, data_seed: int = 0) -> FingerprintScore:
    vals = lever_values(family)

    def sigs(seeds):
        return {v: [lever_signature(family, v, spec, seed=s, steps=steps, data_seed=data_seed)
                    for s in seeds] for v in vals}

    reference = sigs(ref_seeds)
    query = sigs(query_seeds)
    queries = [(v, sig) for v, sl in query.items() for sig in sl]
    acc, margin = discriminate_signatures(reference, queries)
    return FingerprintScore(accuracy=acc, margin=margin)


@dataclass(frozen=True)
class LeverFPConfig:
    family: str
    budget_steps: int
    probe_steps: int
    ref_seeds: list[int]
    query_seeds: list[int]
    commit_seeds: list[int]
    correct_margin: float
    human_steps: int


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


def default_leverfp_config(family: str = "optimizer") -> LeverFPConfig:
    probe_steps = 80
    ref_seeds, query_seeds = [0, 1, 2], [3, 4, 5]
    n = len(lever_values(family))
    cost = probe_steps * n * (len(ref_seeds) + len(query_seeds)) * 2
    return LeverFPConfig(
        family=family, budget_steps=cost * 3, probe_steps=probe_steps,
        ref_seeds=ref_seeds, query_seeds=query_seeds, commit_seeds=[10, 11, 12, 13, 14, 15],
        correct_margin=0.22, human_steps=cost * 2,
    )


class LeverFingerprintEpisode:
    def __init__(self, config: LeverFPConfig, budget_spent: int = 0):
        self.config = config
        self.budget_spent = budget_spent

    def probe_cost(self) -> int:
        c = self.config
        return c.probe_steps * len(lever_values(c.family)) * (len(c.ref_seeds) + len(c.query_seeds)) * 2

    def probe(self, spec: DatasetSpec) -> ProbeResult:
        c = self.config
        cost = self.probe_cost()
        remaining = c.budget_steps - self.budget_spent
        if cost > remaining:
            return ProbeResult(0.0, 0.0, 0, max(remaining, 0), over_budget=True)
        fp = score_lever_fingerprint(c.family, spec, c.ref_seeds, c.query_seeds, c.probe_steps)
        self.budget_spent += cost
        return ProbeResult(fp.accuracy, fp.margin, cost,
                           c.budget_steps - self.budget_spent, over_budget=False)

    def commit(self, spec: DatasetSpec) -> CommitResult:
        c = self.config
        half = len(c.commit_seeds) // 2
        ref, query = c.commit_seeds[:half], c.commit_seeds[half:]
        fp = score_lever_fingerprint(c.family, spec, ref, query, c.probe_steps)
        correct = fp.margin >= c.correct_margin
        agent_steps = self.budget_spent + self.probe_cost()
        score = rhae_ml_score(correct, agent_steps, c.human_steps)
        return CommitResult(score, correct, fp.margin, agent_steps)
