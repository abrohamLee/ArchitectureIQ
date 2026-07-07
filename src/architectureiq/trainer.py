from dataclasses import dataclass

import torch
from torch import nn

from architectureiq.determinism import set_determinism
from architectureiq.models import build_model


@dataclass
class LearningCurve:
    steps: list[int]
    loss: list[float]
    acc: list[float]


def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    model.train()
    return loss, acc


def train_curve(
    arch: str,
    X: torch.Tensor,
    y: torch.Tensor,
    in_dim: int,
    n_classes: int,
    steps: int,
    seed: int,
    eval_every: int = 10,
    lr: float = 1e-2,
) -> LearningCurve:
    set_determinism(seed)
    model = build_model(arch, in_dim, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rec_steps: list[int] = []
    rec_loss: list[float] = []
    rec_acc: list[float] = []

    def record(step: int) -> None:
        loss, acc = _evaluate(model, X, y)
        rec_steps.append(step)
        rec_loss.append(loss)
        rec_acc.append(acc)

    record(0)
    for step in range(1, steps + 1):
        opt.zero_grad()
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            record(step)
    return LearningCurve(rec_steps, rec_loss, rec_acc)
