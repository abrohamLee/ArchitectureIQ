import json
import os

from architectureiq.curvebank import Candidate
from architectureiq.datasets import DatasetSpec
from architectureiq.tournament import Tournament, TournamentConfig

_STATE = "tour_state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_tournament(run_dir: str, config: TournamentConfig) -> Tournament:
    os.makedirs(run_dir, exist_ok=True)
    t = Tournament(config)
    save_tournament(run_dir, t)
    return t


def save_tournament(run_dir: str, t: Tournament) -> None:
    os.makedirs(run_dir, exist_ok=True)
    c = t.config
    payload = {
        "config": {
            "spec": c.spec.__dict__,
            "candidates": [cand.__dict__ for cand in c.candidates],
            "budget_steps": c.budget_steps,
            "max_steps": c.max_steps,
            "eval_every": c.eval_every,
            "human_steps": c.human_steps,
            "regret_threshold": c.regret_threshold,
            "seed": c.seed,
        },
        "trained": t.snapshot()["trained"],
        "budget_spent": t.budget_spent,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_tournament(run_dir: str) -> Tournament:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    cfg_d = p["config"]
    config = TournamentConfig(
        spec=DatasetSpec(**cfg_d["spec"]),
        candidates=[Candidate(**c) for c in cfg_d["candidates"]],
        budget_steps=cfg_d["budget_steps"],
        max_steps=cfg_d["max_steps"],
        eval_every=cfg_d["eval_every"],
        human_steps=cfg_d["human_steps"],
        regret_threshold=cfg_d["regret_threshold"],
        seed=cfg_d["seed"],
    )
    return Tournament(config, trained=p["trained"], budget_spent=p["budget_spent"])
