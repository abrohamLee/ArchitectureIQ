"""杠杆库(lever library)—— 真实、有名字的研究杠杆的答案集 + 签名。

一个杠杆 = (真实研究组件, 已知答案集, 每答案的行为)。v1 的真实 tier 用它把玩具架构
换成真实杠杆(优化器 / 激活 / init …)。这里先落"优化器"家族(MLS 旗舰杠杆、Marin
adam↔muon 现象),在**玩具尺度**上跑真实优化器——杠杆是真的,尺度是玩具的,故便宜可
retrain(生成式任务的前提)。

签名 = 结构优势 Δ-签名(复用 fingerprint.structure_advantage),固定基座 mlp、只换优化器。
"""
from architectureiq.datasets import DatasetSpec, generate
from architectureiq.fingerprint import structure_advantage

# 优化器杠杆:真实答案集 + 各自合适的 lr(已验证在玩具上可分)
OPTIMIZERS = ["adam", "sgd", "rmsprop"]
_OPT_LR = {"adam": 1e-2, "sgd": 0.3, "rmsprop": 1e-2}

LEVER_FAMILIES = {"optimizer": OPTIMIZERS}


def lever_values(family: str) -> list[str]:
    return list(LEVER_FAMILIES[family])


def lever_signature(family: str, value: str, spec: DatasetSpec, seed: int,
                    steps: int = 80, eval_every: int = 20, data_seed: int = 0) -> list[float]:
    """在 spec 数据集上,应用杠杆 (family,value) 训练玩具基座,返回结构优势 Δ-签名。"""
    X, y, in_dim, n_classes = generate(spec, seed=data_seed)
    if family == "optimizer":
        return structure_advantage("mlp", X, y, in_dim, n_classes, steps=steps, seed=seed,
                                   eval_every=eval_every, optimizer_name=value, lr=_OPT_LR[value])
    raise ValueError(f"unknown lever family: {family}")
