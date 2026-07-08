from dataclasses import dataclass

from architectureiq.datasets import DatasetSpec, generate
from architectureiq.trainer import LearningCurve, train_curve


@dataclass(frozen=True)
class Candidate:
    id: str
    arch: str
    lr: float


class CurveBank:
    """每个 candidate 真训一次到 max_steps 并缓存整条曲线;查询 = 前缀 lookup。

    缓存只存真跑 ground truth(spec §3.2),不做任何 surrogate 预测。
    """

    def __init__(self, spec: DatasetSpec, max_steps: int, eval_every: int = 20,
                 seed: int = 0, data_seed: int = 0):
        self.spec = spec
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.seed = seed
        self._X, self._y, self._in_dim, self._n_classes = generate(spec, seed=data_seed)
        self._cache: dict[str, LearningCurve] = {}

    def curve(self, cand: Candidate) -> LearningCurve:
        if cand.id not in self._cache:
            self._cache[cand.id] = train_curve(
                cand.arch, self._X, self._y, self._in_dim, self._n_classes,
                steps=self.max_steps, seed=self.seed, eval_every=self.eval_every, lr=cand.lr,
            )
        return self._cache[cand.id]

    def checkpoints(self) -> list[int]:
        return list(range(0, self.max_steps + 1, self.eval_every))

    def acc_at(self, cand: Candidate, steps: int) -> float:
        curve = self.curve(cand)
        # 向下取最近的 checkpoint <= steps
        idx = 0
        for i, s in enumerate(curve.steps):
            if s <= steps:
                idx = i
            else:
                break
        return curve.acc[idx]

    def final_acc(self, cand: Candidate) -> float:
        return self.acc_at(cand, self.max_steps)
