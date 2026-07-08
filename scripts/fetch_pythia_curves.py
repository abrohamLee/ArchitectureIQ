"""批量下载 Pythia 跨检查点的 eval 曲线,汇成一个 jsonl 供 RealCurveBank 使用。

数据源:EleutherAI/pythia 仓库 evals/pythia-v1/<model>/<shot>/<sz>_step<STEP>.json。
每个 checkpoint 一个 lm-eval-harness 结果文件(含 ~106 任务);一次下载榨出多任务曲线。
矩阵 = 21 模型变体 × {zero,five}-shot × 一批干净任务 → 数百条真实 metric-vs-step 曲线。

策略:git-tree API 一次列全部文件(避免 rate-limit),按 (model, shot) 分组,
并发下载 raw JSON,抽取 curated 任务,按 step 排序成曲线,逐条 append 到 jsonl。

用法:
    uv run python scripts/fetch_pythia_curves.py                  # 全矩阵
    uv run python scripts/fetch_pythia_curves.py --models pythia-1.4b pythia-410m
"""
import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TREE = "https://api.github.com/repos/EleutherAI/pythia/git/trees/main?recursive=1"
RAW = "https://raw.githubusercontent.com/EleutherAI/pythia/main/{path}"
PATH_RE = re.compile(r"evals/pythia-v1/([^/]+)/([^/]+)/[^/]+_step(\d+)\.json$")

# 有真实学习信号的干净任务(避开近瞎猜的 MMLU 子项与 bias 类 crows_pairs)
CURATED_TASKS = [
    "lambada_openai", "piqa", "sciq", "winogrande",
    "arc_easy", "arc_challenge", "logiqa", "wsc",
    "openbookqa", "lambada_standard",
]
METRIC = "acc"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "architectureiq"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read()


def list_eval_files(models: list[str] | None) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """回 {(model, shot): [(step, path), ...]},来自一次 git-tree 调用。"""
    tree = json.loads(_get(TREE))["tree"]
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for node in tree:
        m = PATH_RE.match(node.get("path", ""))
        if not m:
            continue
        model, shot, step = m.group(1), m.group(2), int(m.group(3))
        if models and model not in models:
            continue
        groups.setdefault((model, shot), []).append((step, node["path"]))
    for k in groups:
        groups[k].sort()
    return groups


def _fetch_step(step_path: tuple[int, str]) -> tuple[int, dict]:
    step, path = step_path
    try:
        return step, json.loads(_get(RAW.format(path=path)))["results"]
    except Exception:  # noqa: BLE001
        return step, {}


def build_curves(model: str, shot: str, files: list[tuple[int, str]]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=8) as ex:
        step_results = list(ex.map(_fetch_step, files))
    step_results.sort(key=lambda x: x[0])
    records = []
    for task in CURATED_TASKS:
        steps, values = [], []
        for step, results in step_results:
            if task in results and METRIC in results[task]:
                steps.append(step)
                values.append(results[task][METRIC])
        if len(steps) >= 10:  # 至少 10 点才算一条曲线
            records.append({
                "id": f"pythia|{model}|{shot}|{task}",
                "model": model, "shot": shot, "task": task, "metric": METRIC,
                "steps": steps, "values": values,
                "source": f"EleutherAI/pythia evals/pythia-v1/{model}/{shot}",
            })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="限定模型;缺省=全部")
    ap.add_argument("--out", default="data/real_curves/pythia_curves.jsonl")
    args = ap.parse_args()

    groups = list_eval_files(args.models)
    print(f"{len(groups)} (model,shot) 组, {sum(len(v) for v in groups.values())} 个 step 文件")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    total = 0
    with open(args.out, "w") as f:
        for (model, shot), files in sorted(groups.items()):
            recs = build_curves(model, shot, files)
            for r in recs:
                f.write(json.dumps(r) + "\n")
            total += len(recs)
            print(f"  {model:26s} {shot:10s} -> {len(recs)} 曲线 ({len(files)} 检查点)")
    print(f"共 {total} 条曲线 -> {args.out}")


if __name__ == "__main__":
    main()
