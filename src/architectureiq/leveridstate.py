import json
import os

from architectureiq.leverid import LeverIDConfig, LeverIDEpisode

_STATE = "bbl_state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_leverid(run_dir: str, config: LeverIDConfig) -> LeverIDEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = LeverIDEpisode(config)
    save_leverid(run_dir, ep)
    return ep


def save_leverid(run_dir: str, ep: LeverIDEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    payload = {"config": ep.config.__dict__, "budget_spent": ep.budget_spent}
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_leverid(run_dir: str) -> LeverIDEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    return LeverIDEpisode(LeverIDConfig(**p["config"]), budget_spent=p["budget_spent"])
