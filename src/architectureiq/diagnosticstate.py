import json
import os

from architectureiq.datasets import DatasetSpec
from architectureiq.diagnostic import DiagnosticConfig, DiagnosticEpisode


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, "dx_state.json")


def init_diagnostic(run_dir: str, config: DiagnosticConfig) -> DiagnosticEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = DiagnosticEpisode(config)
    save_diagnostic(run_dir, ep)
    return ep


def save_diagnostic(run_dir: str, ep: DiagnosticEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    c = ep.config
    payload = {
        "config": {
            "pathology": c.pathology, "spec": c.spec.__dict__, "budget": c.budget,
            "human_budget": c.human_budget, "steps": c.steps, "seed": c.seed,
        },
        "budget_spent": ep.budget_spent,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_diagnostic(run_dir: str) -> DiagnosticEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    cd = p["config"]
    config = DiagnosticConfig(
        pathology=cd["pathology"], spec=DatasetSpec(**cd["spec"]), budget=cd["budget"],
        human_budget=cd["human_budget"], steps=cd["steps"], seed=cd["seed"],
    )
    return DiagnosticEpisode(config, budget_spent=p["budget_spent"])
