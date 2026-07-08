"""加载 vendored 的真实 LLM learning curve(Pythia/Marin),对齐 LearningCurve 接口。

curve JSON 由 scripts/fetch_pythia_curves.py 产出:{source, steps, values, meta}。
real tier 用它替换玩具 CurveBank —— 只存真跑 ground truth,绝不 surrogate 预测。
"""
import json
import os

from architectureiq.trainer import LearningCurve

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         "data", "real_curves")


def curve_path(name: str) -> str:
    """按名(不含 .json)解析到 data/real_curves 下的路径。"""
    return os.path.join(_DATA_DIR, f"{name}.json")


def load_real_curve(name_or_path: str) -> LearningCurve:
    """读真实曲线为 LearningCurve:acc=真实 metric 值,loss 留空(该 tier 只预测 metric)。"""
    path = name_or_path if name_or_path.endswith(".json") else curve_path(name_or_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        d = json.load(f)
    return LearningCurve(steps=list(d["steps"]), loss=[], acc=list(d["values"]))


def list_real_curves() -> list[str]:
    if not os.path.isdir(_DATA_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(_DATA_DIR) if f.endswith(".json"))
