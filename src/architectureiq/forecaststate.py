import json
import os

from architectureiq.datasets import DatasetSpec
from architectureiq.forecast import ForecastConfig, ForecastEpisode
from architectureiq.realcurvebank import get_curve

_STATE = "fc_state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_forecast(run_dir: str, config: ForecastConfig) -> ForecastEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = ForecastEpisode(config)
    save_forecast(run_dir, ep)
    return ep


def init_real_forecast(run_dir: str, curve_id: str) -> ForecastEpisode:
    """real tier:用一条真实曲线(Pythia 等)开局。"""
    os.makedirs(run_dir, exist_ok=True)
    ep = ForecastEpisode(curve=get_curve(curve_id), curve_id=curve_id)
    save_forecast(run_dir, ep)
    return ep


def save_forecast(run_dir: str, ep: ForecastEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    rolling = {"cursor": ep.cursor, "skill_sum": ep.skill_sum, "rounds": ep.rounds}
    if ep.curve_id is not None:  # real tier:只存曲线 id,load 时重建
        payload = {"real": True, "curve_id": ep.curve_id, **rolling}
    else:
        c = ep.config
        payload = {
            "real": False,
            "config": {
                "arch": c.arch, "spec": c.spec.__dict__, "lr": c.lr,
                "max_steps": c.max_steps, "eval_every": c.eval_every, "seed": c.seed,
            },
            **rolling,
        }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_forecast(run_dir: str) -> ForecastEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    roll = {"cursor": p["cursor"], "skill_sum": p["skill_sum"], "rounds": p["rounds"]}
    if p.get("real"):
        return ForecastEpisode(curve=get_curve(p["curve_id"]), curve_id=p["curve_id"], **roll)
    cd = p["config"]
    config = ForecastConfig(
        arch=cd["arch"], spec=DatasetSpec(**cd["spec"]), lr=cd["lr"],
        max_steps=cd["max_steps"], eval_every=cd["eval_every"], seed=cd["seed"],
    )
    return ForecastEpisode(config, **roll)
