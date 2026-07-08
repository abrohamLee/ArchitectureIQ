import json
import os
from dataclasses import asdict

from architectureiq.episode import EpisodeConfig, Environment

_STATE = "state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_state(run_dir: str, config: EpisodeConfig) -> Environment:
    os.makedirs(run_dir, exist_ok=True)
    env = Environment(config)
    save_state(run_dir, env)
    return env


def save_state(run_dir: str, env: Environment) -> None:
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "config": asdict(env.config),
        "budget_spent": env.budget_spent,
        "committed": env.committed,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_state(run_dir: str) -> Environment:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        payload = json.load(f)
    config = EpisodeConfig(**payload["config"])
    return Environment(config, budget_spent=payload["budget_spent"], committed=payload["committed"])
