"""杠杆库(lever library)—— 真实、有名字的研究杠杆的答案集 + 签名。

一个杠杆 = (真实研究组件, 已知答案集, 每答案的行为)。v1 的真实 tier 用它把玩具架构
换成真实杠杆(优化器 / 激活 / init …)。这里先落"优化器"家族(MLS 旗舰杠杆、Marin
adam↔muon 现象),在**玩具尺度**上跑真实优化器——杠杆是真的,尺度是玩具的,故便宜可
retrain(生成式任务的前提)。

签名 = 结构优势 Δ-签名(复用 fingerprint.structure_advantage),固定基座 mlp、只换优化器。
"""
import torch
from torch import nn

from architectureiq.datasets import DatasetSpec, generate
from architectureiq.determinism import set_determinism
from architectureiq.fingerprint import structure_advantage

# 真实杠杆的答案集(玩具尺度跑真实杠杆:杠杆真、尺度玩具、可 retrain)
OPTIMIZERS = ["adam", "sgd", "rmsprop"]
_OPT_LR = {"adam": 1e-2, "sgd": 0.3, "rmsprop": 1e-2}
ACTIVATIONS = ["relu", "tanh", "gelu"]
_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}

LEVER_FAMILIES = {"optimizer": OPTIMIZERS, "activation": ACTIVATIONS}


def lever_values(family: str) -> list[str]:
    return list(LEVER_FAMILIES[family])


def _act_mlp(activation: str, in_dim: int, n_classes: int, hidden: int = 64) -> nn.Module:
    A = _ACT[activation]
    return nn.Sequential(nn.Linear(in_dim, hidden), A(), nn.Linear(hidden, hidden), A(),
                         nn.Linear(hidden, n_classes))


def _act_signature(value, X, y, in_dim, n_classes, steps, seed, eval_every) -> list[float]:
    """激活杠杆的结构优势 Δ-签名(自含:MLP 换激活 + Adam,减掉标签打乱基线)。"""
    g = torch.Generator().manual_seed(seed * 100 + 7)
    y_shuf = y[torch.randperm(len(y), generator=g)]

    def curve(labels):
        set_determinism(seed)
        m = _act_mlp(value, in_dim, n_classes)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        out = []
        for s in range(steps + 1):
            if s % eval_every == 0:
                with torch.no_grad():
                    out.append(nn.functional.cross_entropy(m(X), labels).item())
            if s < steps:
                opt.zero_grad()
                nn.functional.cross_entropy(m(X), labels).backward()
                opt.step()
        return out

    real, ctrl = curve(y), curve(y_shuf)
    return [c - r for c, r in zip(ctrl, real)]


def lever_signature(family: str, value: str, spec: DatasetSpec, seed: int,
                    steps: int = 80, eval_every: int = 20, data_seed: int = 0) -> list[float]:
    """在 spec 数据集上,应用杠杆 (family,value) 训练玩具基座,返回结构优势 Δ-签名。"""
    X, y, in_dim, n_classes = generate(spec, seed=data_seed)
    if family == "optimizer":
        return structure_advantage("mlp", X, y, in_dim, n_classes, steps=steps, seed=seed,
                                   eval_every=eval_every, optimizer_name=value, lr=_OPT_LR[value])
    if family == "activation":
        return _act_signature(value, X, y, in_dim, n_classes, steps, seed, eval_every)
    raise ValueError(f"unknown lever family: {family}")
