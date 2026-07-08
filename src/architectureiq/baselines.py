import random

from architectureiq.datasets import DatasetSpec
from architectureiq.episode import CommitResult, EpisodeConfig, Environment

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
