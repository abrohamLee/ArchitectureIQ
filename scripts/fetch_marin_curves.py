"""从 Marin 的公开 WandB 拉真实训练曲线(含 loss 尖峰/发散等病例),存成 curve JSON。

Marin(Stanford CRFM)是"开放实验室",每个实验(含失败/病态 run)的 WandB 全公开:
entity/project = stanford-mercury/marin。8B 复盘记录的真实病例(lm_head 爆炸→z-loss、
换 batch/LR 的 loss spike)是 ③ 训练医生 real tier 的真实病历。

⚠️ 需 wandb 认证(公开项目也要 key):先在本会话运行
    ! wandb login          # 按提示粘贴 wandb.ai/authorize 的 key
或  ! export WANDB_API_KEY=<key>

用法(认证后):
    # 先探索:列出名字含关键词的 run 及其可用 metric
    uv run python scripts/fetch_marin_curves.py --list --name-contains raccoon
    # 再拉:把某 run 的 loss-vs-step 存成曲线
    uv run python scripts/fetch_marin_curves.py --run <entity/project/run_id> --metric train/loss

输出:data/real_curves/marin_<run>.json = {source, steps, values, meta}
"""
import argparse
import json
import os

ENTITY_PROJECT = "stanford-mercury/marin"


def _api():
    import wandb
    return wandb.Api()


def list_runs(name_contains: str | None, limit: int = 20) -> None:
    api = _api()
    print(f"扫描 {ENTITY_PROJECT} 的 run ...")
    shown = 0
    for run in api.runs(ENTITY_PROJECT):
        if name_contains and name_contains.lower() not in run.name.lower():
            continue
        # 探索可用的 loss/metric key(取 summary 里数值型)
        metric_keys = [k for k, v in run.summary.items()
                       if isinstance(v, (int, float)) and ("loss" in k.lower() or "bpb" in k.lower())]
        print(f"  {run.id}  state={run.state:8s}  name={run.name}")
        print(f"      loss-ish metrics: {metric_keys[:8]}")
        shown += 1
        if shown >= limit:
            break
    print(f"共列出 {shown} 个 run。用 --run {ENTITY_PROJECT}/<id> --metric <key> 拉取。")


def fetch_run(run_path: str, metric: str, step_key: str = "_step") -> dict:
    api = _api()
    run = api.run(run_path)
    hist = run.history(keys=[metric, step_key], pandas=False)
    steps, values = [], []
    for row in hist:
        if metric in row and row[metric] is not None:
            steps.append(int(row.get(step_key, len(steps))))
            values.append(float(row[metric]))
    return {
        "source": f"Marin WandB {run_path}",
        "steps": steps,
        "values": values,
        "meta": {"run": run_path, "name": run.name, "metric": metric, "state": run.state},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出 run 及可用 metric")
    ap.add_argument("--name-contains", default=None)
    ap.add_argument("--run", default=None, help="entity/project/run_id")
    ap.add_argument("--metric", default="train/loss")
    ap.add_argument("--out-dir", default="data/real_curves")
    args = ap.parse_args()

    if args.list:
        list_runs(args.name_contains)
        return
    if not args.run:
        ap.error("给 --run <entity/project/run_id> 或用 --list 先探索")

    curve = fetch_run(args.run, args.metric)
    os.makedirs(args.out_dir, exist_ok=True)
    safe = args.run.replace("/", "_") + "_" + args.metric.replace("/", "-")
    out = os.path.join(args.out_dir, f"marin_{safe}.json")
    with open(out, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"wrote {len(curve['steps'])} points -> {out}")


if __name__ == "__main__":
    main()
