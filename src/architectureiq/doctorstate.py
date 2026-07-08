import json
import os

from architectureiq.datasets import DatasetSpec
from architectureiq.doctor import DoctorConfig, DoctorEpisode

_STATE = "dr_state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_doctor(run_dir: str, config: DoctorConfig) -> DoctorEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = DoctorEpisode(config)
    save_doctor(run_dir, ep)
    return ep


def save_doctor(run_dir: str, ep: DoctorEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    c = ep.config
    payload = {
        "config": {
            "spec": c.spec.__dict__, "arch": c.arch, "sick_lr": c.sick_lr,
            "grid": c.grid, "max_steps": c.max_steps, "eval_every": c.eval_every,
            "budget_steps": c.budget_steps, "cure_acc": c.cure_acc,
            "human_steps": c.human_steps, "seed": c.seed,
        },
        "budget_spent": ep.budget_spent,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_doctor(run_dir: str) -> DoctorEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    cd = p["config"]
    config = DoctorConfig(
        spec=DatasetSpec(**cd["spec"]), arch=cd["arch"], sick_lr=cd["sick_lr"],
        grid=cd["grid"], max_steps=cd["max_steps"], eval_every=cd["eval_every"],
        budget_steps=cd["budget_steps"], cure_acc=cd["cure_acc"],
        human_steps=cd["human_steps"], seed=cd["seed"],
    )
    return DoctorEpisode(config, budget_spent=p["budget_spent"])
