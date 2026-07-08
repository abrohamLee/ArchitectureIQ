import json
import os

from architectureiq.datasets import DatasetSpec
from architectureiq.forecast import ForecastConfig, ForecastEpisode

_STATE = "fc_state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_forecast(run_dir: str, config: ForecastConfig) -> ForecastEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = ForecastEpisode(config)
    save_forecast(run_dir, ep)
    return ep


def save_forecast(run_dir: str, ep: ForecastEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    c = ep.config
    payload = {
        "config": {
            "arch": c.arch, "spec": c.spec.__dict__, "lr": c.lr,
            "max_steps": c.max_steps, "eval_every": c.eval_every, "seed": c.seed,
        },
        "cursor": ep.cursor, "skill_sum": ep.skill_sum, "rounds": ep.rounds,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_forecast(run_dir: str) -> ForecastEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    cd = p["config"]
    config = ForecastConfig(
        arch=cd["arch"], spec=DatasetSpec(**cd["spec"]), lr=cd["lr"],
        max_steps=cd["max_steps"], eval_every=cd["eval_every"], seed=cd["seed"],
    )
    return ForecastEpisode(config, cursor=p["cursor"], skill_sum=p["skill_sum"], rounds=p["rounds"])
