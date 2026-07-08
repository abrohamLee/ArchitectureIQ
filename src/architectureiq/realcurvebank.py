"""加载 vendored 的真实 LLM learning curve(Pythia/Marin),对齐 LearningCurve 接口。

数据在 data/real_curves/pythia_curves.jsonl(由 scripts/fetch_pythia_curves.py 批量产出),
每行一条曲线记录:{id, model, shot, task, metric, steps, values, source}。
real tier 用它替换玩具 CurveBank —— 只存真跑 ground truth,绝不 surrogate 预测。
id 形如 "pythia|pythia-1.4b|zero-shot|piqa"。
"""
import json
import os

from architectureiq.trainer import LearningCurve

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         "data", "real_curves")
_BANK = os.path.join(_DATA_DIR, "pythia_curves.jsonl")


def _records(path: str = _BANK) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bank(path: str = _BANK) -> dict[str, dict]:
    """回 {id: record}。"""
    return {r["id"]: r for r in _records(path)}


def list_curve_ids(path: str = _BANK) -> list[str]:
    return sorted(load_bank(path).keys())


def _to_curve(rec: dict) -> LearningCurve:
    # acc=真实 metric 值,loss 留空(real tier 只预测该 metric)
    return LearningCurve(steps=list(rec["steps"]), loss=[], acc=list(rec["values"]))


def get_curve(curve_id: str, path: str = _BANK) -> LearningCurve:
    bank = load_bank(path)
    if curve_id not in bank:
        raise KeyError(curve_id)
    return _to_curve(bank[curve_id])


def filter_ids(model: str | None = None, shot: str | None = None,
               task: str | None = None, path: str = _BANK) -> list[str]:
    """按维度筛选曲线 id,便于按尺寸/shot/任务组织 tier。"""
    out = []
    for rec in _records(path):
        if model and rec["model"] != model:
            continue
        if shot and rec["shot"] != shot:
            continue
        if task and rec["task"] != task:
            continue
        out.append(rec["id"])
    return sorted(out)
