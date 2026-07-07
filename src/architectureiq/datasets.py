from dataclasses import dataclass

import torch

from architectureiq.determinism import set_determinism


@dataclass(frozen=True)
class DatasetSpec:
    family: str
    n_samples: int
    modulus: int = 7
    n_bits: int = 8
    label_noise: float = 0.0


def _apply_noise(y: torch.Tensor, n_classes: int, noise: float, g: torch.Generator) -> torch.Tensor:
    if noise <= 0.0:
        return y
    mask = torch.rand(y.shape, generator=g) < noise
    rand_labels = torch.randint(0, n_classes, y.shape, generator=g)
    return torch.where(mask, rand_labels, y)


def generate(spec: DatasetSpec, seed: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    set_determinism(seed)
    g = torch.Generator().manual_seed(seed)

    if spec.family in ("modular_addition", "random"):
        m = spec.modulus
        a = torch.randint(0, m, (spec.n_samples,), generator=g)
        b = torch.randint(0, m, (spec.n_samples,), generator=g)
        X = torch.zeros(spec.n_samples, 2 * m, dtype=torch.float32)
        X[torch.arange(spec.n_samples), a] = 1.0
        X[torch.arange(spec.n_samples), m + b] = 1.0
        if spec.family == "modular_addition":
            y = (a + b) % m
        else:
            y = torch.randint(0, m, (spec.n_samples,), generator=g)
        n_classes = m
        in_dim = 2 * m
    elif spec.family == "parity":
        X = (torch.rand(spec.n_samples, spec.n_bits, generator=g) < 0.5).float()
        y = (X.sum(dim=1).long() % 2)
        n_classes = 2
        in_dim = spec.n_bits
    else:
        raise ValueError(f"unknown family: {spec.family}")

    y = _apply_noise(y.long(), n_classes, spec.label_noise, g)
    return X, y.long(), in_dim, n_classes
