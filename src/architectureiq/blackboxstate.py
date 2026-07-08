import json
import os

from architectureiq.blackbox import BlackboxConfig, BlackboxEpisode

_STATE = "bb_state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_blackbox(run_dir: str, config: BlackboxConfig) -> BlackboxEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = BlackboxEpisode(config)
    save_blackbox(run_dir, ep)
    return ep


def save_blackbox(run_dir: str, ep: BlackboxEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    c = ep.config
    payload = {
        "config": {
            "pool": c.pool, "hidden_arch": c.hidden_arch, "ref_seeds": c.ref_seeds,
            "mystery_seed": c.mystery_seed, "steps": c.steps,
            "budget_steps": c.budget_steps, "human_steps": c.human_steps,
            "data_seed": c.data_seed,
        },
        "budget_spent": ep.budget_spent,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_blackbox(run_dir: str) -> BlackboxEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    config = BlackboxConfig(**p["config"])
    return BlackboxEpisode(config, budget_spent=p["budget_spent"])
