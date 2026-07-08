"""下载 Pythia 跨检查点的 eval 曲线,存成 RealCurveBank 用的 curve JSON。

数据源:EleutherAI/pythia 仓库 evals/pythia-v1/<model>/<shot>/<sz>_step<STEP>.json,
每个 checkpoint 一个 lm-eval-harness 结果文件;跨 step 提取 results[task][metric]
即一条真实 metric-vs-step 曲线(真 LLM,公开、可复现)。

用法:
    uv run python scripts/fetch_pythia_curves.py --model pythia-1.4b --sz 1.4b \\
        --shot zero-shot --task piqa --metric acc

输出:data/real_curves/pythia-1.4b_piqa_zeroshot.json = {source, steps, values, meta}
"""
import argparse
import json
import os
import re
import urllib.request

API = "https://api.github.com/repos/EleutherAI/pythia/contents/evals/pythia-v1/{model}/{shot}"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "architectureiq"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch_curve(model: str, sz: str, shot: str, task: str, metric: str) -> dict:
    listing = json.loads(_get(API.format(model=model, shot=shot)))
    step_files = []
    for entry in listing:
        m = re.search(r"step(\d+)\.json$", entry["name"])
        if m:
            step_files.append((int(m.group(1)), entry["download_url"]))
    step_files.sort()

    steps, values = [], []
    for step, url in step_files:
        try:
            data = json.loads(_get(url))
            val = data["results"][task][metric]
        except Exception as e:  # noqa: BLE001
            print(f"  skip step{step}: {e}")
            continue
        steps.append(step)
        values.append(val)
        print(f"  step{step}: {task}.{metric} = {val:.4f}")

    return {
        "source": f"EleutherAI/pythia evals/pythia-v1/{model}/{shot}",
        "steps": steps,
        "values": values,
        "meta": {"model": model, "shot": shot, "task": task, "metric": metric},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--sz", default="1.4b")
    ap.add_argument("--shot", default="zero-shot")
    ap.add_argument("--task", default="piqa")
    ap.add_argument("--metric", default="acc")
    ap.add_argument("--out-dir", default="data/real_curves")
    args = ap.parse_args()

    curve = fetch_curve(args.model, args.sz, args.shot, args.task, args.metric)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.model}_{args.task}_{args.shot.replace('-', '')}.json")
    with open(out, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"wrote {len(curve['steps'])} points -> {out}")


if __name__ == "__main__":
    main()
