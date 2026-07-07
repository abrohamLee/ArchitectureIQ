from dataclasses import dataclass

import torch

from architectureiq.datasets import DatasetSpec, generate
from architectureiq.discriminator import discriminate_signatures
from architectureiq.trainer import train_curve


@dataclass(frozen=True)
class FingerprintScore:
    accuracy: float
    margin: float


def structure_advantage(
    arch: str,
    X: torch.Tensor,
    y: torch.Tensor,
    in_dim: int,
    n_classes: int,
    steps: int,
    seed: int,
    eval_every: int = 20,
) -> list[float]:
    """架构 A 的「结构优势」签名 Δ(k) = loss(标签打乱, k步) − loss(真实数据, k步)。

    对照 = 同一输入、标签随机打乱(randomization test),抵消该架构自身的记忆
    速度(nuisance)。Δ 大 ⟺ 数据结构匹配 A 的 inductive bias。generic/无结构
    数据 → Δ ≈ 0(真实与打乱无区别),使所有架构签名塌到零 → 不可分。
    """
    g = torch.Generator().manual_seed(seed * 100 + 7)
    y_shuffled = y[torch.randperm(len(y), generator=g)]

    real = train_curve(arch, X, y, in_dim, n_classes, steps=steps, seed=seed, eval_every=eval_every)
    ctrl = train_curve(arch, X, y_shuffled, in_dim, n_classes, steps=steps, seed=seed, eval_every=eval_every)
    return [c - r for c, r in zip(ctrl.loss, real.loss)]


def score_fingerprint(
    spec: DatasetSpec,
    archs: list[str],
    ref_seeds: list[int],
    query_seeds: list[int],
    steps: int,
    data_seed: int = 0,
) -> FingerprintScore:
    X, y, in_dim, n_classes = generate(spec, seed=data_seed)

    def signatures_for(seeds: list[int]) -> dict[str, list[list[float]]]:
        return {
            arch: [
                structure_advantage(arch, X, y, in_dim, n_classes, steps=steps, seed=s)
                for s in seeds
            ]
            for arch in archs
        }

    reference = signatures_for(ref_seeds)
    query_sigs = signatures_for(query_seeds)
    queries = [
        (arch, sig)
        for arch, sig_list in query_sigs.items()
        for sig in sig_list
    ]

    acc, margin = discriminate_signatures(reference, queries)
    return FingerprintScore(accuracy=acc, margin=margin)
