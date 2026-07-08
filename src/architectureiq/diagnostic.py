"""③ 训练医生 · Tier 2(诊断台)—— 收纳 A2 交互式因果诊断。

多个病因(lr_too_low / dead_relu / vanishing_grad)**故意做成 raw loss 曲线难分**
(都卡在高位平台),但在**不同 observable** 上留下清晰签名。agent 必须**查对诊断信号**
才能定因——背"loss-spike=LR太高"没用。已用数据验证(见 spec §3 ③ 的 ⚠️ 验证)。

observable 定价不同,逼 agent 查最有信息量的那个。非智能地板=不查只猜多数病因。
"""
from dataclasses import dataclass

import torch
from torch import nn

from architectureiq.datasets import DatasetSpec, generate
from architectureiq.determinism import set_determinism
from architectureiq.scoring import rhae_ml_score

PATHOLOGIES = ["lr_too_low", "dead_relu", "vanishing_grad"]
QUERY_COSTS = {"grad_norm": 1, "weight_norm": 1, "dead_fraction": 1, "per_layer_grad": 2}


@dataclass(frozen=True)
class DiagnosticConfig:
    pathology: str
    spec: DatasetSpec
    budget: int
    human_budget: int
    steps: int
    seed: int


@dataclass
class QueryResult:
    observable: str
    value: object   # scalar 或 list
    cost: int
    budget_remaining: int
    over_budget: bool


@dataclass
class DiagnoseResult:
    score: float
    correct: bool
    chosen: str
    truth: str
    agent_steps: int


def default_diagnostic_config(pathology: str = "dead_relu") -> DiagnosticConfig:
    return DiagnosticConfig(
        pathology=pathology,
        spec=DatasetSpec(family="modular_addition", n_samples=300, modulus=7),
        budget=10, human_budget=3, steps=150, seed=0,
    )


def _build(pathology: str, in_dim: int, n_classes: int) -> nn.Sequential:
    act = nn.Tanh if pathology == "vanishing_grad" else nn.ReLU
    depth = 6 if pathology == "vanishing_grad" else 4
    layers, d = [], in_dim
    for i in range(depth):
        lin = nn.Linear(d, 64)
        if pathology == "vanishing_grad":
            with torch.no_grad():
                lin.weight.mul_(0.3)                  # 小权重 + tanh → 梯度消失
        if pathology == "dead_relu" and i == 0:
            with torch.no_grad():
                lin.bias.fill_(-3.0)                  # 大负 bias → ReLU 大面积死
        layers += [lin, act()]
        d = 64
    layers.append(nn.Linear(d, n_classes))
    return nn.Sequential(*layers)


def _run(pathology: str, spec: DatasetSpec, steps: int, seed: int) -> dict:
    """训练病态 run,返回 loss 曲线 + 各 observable 读数(确定性)。"""
    X, y, in_dim, n_classes = generate(spec, seed=0)
    lr = 1e-4 if pathology == "lr_too_low" else 1e-2
    set_determinism(seed)
    model = _build(pathology, in_dim, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for s in range(steps + 1):
        if s % 30 == 0:
            with torch.no_grad():
                curve.append(round(nn.functional.cross_entropy(model(X), y).item(), 4))
        if s < steps:
            opt.zero_grad()
            nn.functional.cross_entropy(model(X), y).backward()
            opt.step()
    # observable:一次 forward(抓激活)+ backward(抓逐层梯度)
    acts = []
    hooks = [m.register_forward_hook(lambda mod, i, o: acts.append(o.detach()))
             for m in model if isinstance(m, (nn.ReLU, nn.Tanh))]
    model.zero_grad()
    nn.functional.cross_entropy(model(X), y).backward()
    for h in hooks:
        h.remove()
    dead_fraction = sum((a.abs() < 1e-3).float().mean().item() for a in acts) / len(acts)
    per_layer = [round(p.grad.norm().item(), 5) for n, p in model.named_parameters()
                 if "weight" in n and p.grad is not None]
    grad_norm = round(sum(g ** 2 for g in per_layer) ** 0.5, 5)
    weight_norm = round(sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5, 3)
    return {
        "loss_curve": curve,
        "grad_norm": grad_norm,
        "weight_norm": weight_norm,
        "dead_fraction": round(dead_fraction, 4),
        "per_layer_grad": per_layer,
    }


class DiagnosticEpisode:
    def __init__(self, config: DiagnosticConfig, budget_spent: int = 0):
        self.config = config
        self.budget_spent = budget_spent
        self._data = _run(config.pathology, config.spec, config.steps, config.seed)

    def sick_curve(self) -> list[float]:
        return self._data["loss_curve"]

    def query(self, observable: str) -> QueryResult:
        cost = QUERY_COSTS[observable]
        remaining = self.config.budget - self.budget_spent
        if cost > remaining:
            return QueryResult(observable, None, 0, max(remaining, 0), over_budget=True)
        self.budget_spent += cost
        return QueryResult(observable, self._data[observable], cost,
                           self.config.budget - self.budget_spent, over_budget=False)

    def answer(self, cause: str) -> DiagnoseResult:
        correct = cause == self.config.pathology
        score = rhae_ml_score(correct, self.budget_spent, self.config.human_budget)
        return DiagnoseResult(score, correct, cause, self.config.pathology, self.budget_spent)
