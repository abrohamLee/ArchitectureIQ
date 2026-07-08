"""批量收割 Marin 公开 WandB 的真实 train/loss 曲线,汇成 marin_curves.jsonl。

Marin(Stanford CRFM / marin-community)是开放实验室,大量 project 的每个 run
(含失败/病态)全公开。相比 Pythia 的 eval-acc 曲线,这里是 train/loss 动力学模态,
且天然多样(不同优化器/批量/LR)。用于 forecast/doctor real tier。

⚠️ 需 wandb 认证:先 `! wandb login` 或 `! export WANDB_API_KEY=<key>`。

用法:
    uv run python scripts/fetch_marin_curves.py                       # 收割默认 sweep projects
    uv run python scripts/fetch_marin_curves.py --projects optimizer-scaling BatchSize
    uv run python scripts/fetch_marin_curves.py --run marin-community/marin/tootsie-8b-soft-raccoon --metric train/loss
"""
import argparse
import json
import os

ENTITY = "marin-community"
DEFAULT_PROJECTS = ["optimizer-scaling", "BatchSize", "Lr_datasize", "Switch-Optimizer"]
METRIC = "train/loss"


def _api():
    import wandb
    return wandb.Api()


def _pull(run, metric: str, samples: int) -> tuple[list[int], list[float]]:
    hist = run.history(keys=[metric, "_step"], samples=samples, pandas=False)
    pts = [(int(row["_step"]), float(row[metric])) for row in hist
           if row.get(metric) is not None and row.get("_step") is not None]
    pts.sort(key=lambda x: x[0])
    return [s for s, _ in pts], [v for _, v in pts]


def harvest(projects: list[str], max_runs: int, samples: int, out: str) -> None:
    api = _api()
    total = 0
    with open(out, "w") as f:
        for proj in projects:
            n = 0
            for run in api.runs(f"{ENTITY}/{proj}"):
                if METRIC not in run.summary:
                    continue
                try:
                    steps, values = _pull(run, METRIC, samples)
                except Exception:  # noqa: BLE001
                    continue
                if len(steps) < 20:
                    continue
                rec = {
                    "id": f"marin|{proj}|{run.id}|train-loss",
                    "model": run.id, "shot": "marin", "task": proj, "metric": "train/loss",
                    "steps": steps, "values": values,
                    "source": f"Marin WandB {ENTITY}/{proj}/{run.id}",
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
                total += 1
                print(f"  {proj:20s} {run.id[:30]:30s} {len(steps)} pts")
                if n >= max_runs:
                    break
            print(f"[{proj}] {n} 曲线")
    print(f"共 {total} 条 -> {out}")


def fetch_one(run_path: str, metric: str, samples: int, out_dir: str) -> None:
    """拉单条 run(如 raccoon 病例),append 到 marin_pathologies.jsonl。"""
    api = _api()
    run = api.run(run_path)
    steps, values = _pull(run, metric, samples)
    proj = run_path.split("/")[1]
    rec = {
        "id": f"marin|{proj}|{run.id}|{metric.replace('/', '-')}",
        "model": run.id, "shot": "marin", "task": proj, "metric": metric,
        "steps": steps, "values": values, "source": f"Marin WandB {run_path}",
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "marin_pathologies.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(steps)} pts ({rec['id']}) -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="*", default=DEFAULT_PROJECTS)
    ap.add_argument("--max-runs", type=int, default=30)
    ap.add_argument("--samples", type=int, default=250)
    ap.add_argument("--run", default=None, help="拉单条 run: entity/project/run_id")
    ap.add_argument("--metric", default=METRIC)
    ap.add_argument("--out", default="data/real_curves/marin_curves.jsonl")
    args = ap.parse_args()
    if args.run:
        fetch_one(args.run, args.metric, args.samples, os.path.dirname(args.out))
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    harvest(args.projects, args.max_runs, args.samples, args.out)


if __name__ == "__main__":
    main()
