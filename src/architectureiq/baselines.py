import random
from dataclasses import dataclass

from architectureiq.curvebank import CurveBank
from architectureiq.datasets import DatasetSpec
from architectureiq.episode import CommitResult, EpisodeConfig, Environment
from architectureiq.tournament import TournamentConfig

_FAMILIES = ["modular_addition", "parity", "random"]


def random_spec(rng_seed: int) -> DatasetSpec:
    r = random.Random(rng_seed)
    return DatasetSpec(
        family=r.choice(_FAMILIES),
        n_samples=r.choice([150, 300, 500]),
        modulus=r.choice([5, 7, 11]),
        n_bits=r.choice([6, 8, 10]),
        label_noise=r.choice([0.0, 0.0, 0.2]),
    )


def run_random_agent(config: EpisodeConfig, n_probes: int, rng_seed: int) -> CommitResult:
    env = Environment(config)
    best_spec = random_spec(rng_seed)
    best_margin = -1.0
    for i in range(n_probes):
        spec = random_spec(rng_seed * 1000 + i)
        res = env.probe(spec)
        if res.over_budget:
            break
        if res.margin > best_margin:
            best_margin, best_spec = res.margin, spec
    return env.commit(best_spec)


@dataclass
class SHResult:
    chosen_id: str
    steps_spent: int
    regret: float


def run_successive_halving(config: TournamentConfig, eta: int = 2) -> SHResult:
    """Successive Halving 非智能地板:语义盲,只看 acc 排序、逐轮砍半。"""
    bank = CurveBank(config.spec, max_steps=config.max_steps,
                     eval_every=config.eval_every, seed=config.seed)
    survivors = list(config.candidates)
    trained: dict[str, int] = {c.id: 0 for c in survivors}
    steps_spent = 0

    rung_budget = config.eval_every
    while len(survivors) > 1:
        rung_budget = min(rung_budget, config.max_steps)
        for c in survivors:
            steps_spent += max(rung_budget - trained[c.id], 0)
            trained[c.id] = max(trained[c.id], rung_budget)
        # 语义盲:只按当前 acc 排序,保留前 1/eta
        survivors.sort(key=lambda c: bank.acc_at(c, rung_budget), reverse=True)
        keep = max(1, len(survivors) // eta)
        survivors = survivors[:keep]
        rung_budget *= eta
        if rung_budget > config.max_steps and len(survivors) > 1:
            # 已到最大预算仍多于 1,按最终 acc 取最优收尾
            survivors = [max(survivors, key=lambda c: bank.final_acc(c))]

    chosen = survivors[0]
    best_final = max(bank.final_acc(c) for c in config.candidates)
    regret = best_final - bank.final_acc(chosen)
    return SHResult(chosen.id, steps_spent, regret)
