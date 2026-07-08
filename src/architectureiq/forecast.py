from dataclasses import dataclass

from architectureiq.curvebank import Candidate, CurveBank
from architectureiq.datasets import DatasetSpec


def linear_extrapolate(steps: list[int], values: list[float], target: int) -> float:
    if len(values) < 2:
        return values[-1]
    x0, x1 = steps[-2], steps[-1]
    y0, y1 = values[-2], values[-1]
    if x1 == x0:
        return y1
    slope = (y1 - y0) / (x1 - x0)
    return y1 + slope * (target - x1)


def skill_score(agent_err: float, baseline_err: float) -> float:
    if baseline_err == 0.0:
        return 1.0 if agent_err == 0.0 else 0.0
    return 1.0 - agent_err / baseline_err


@dataclass(frozen=True)
class ForecastConfig:
    arch: str
    spec: DatasetSpec
    lr: float
    max_steps: int
    eval_every: int
    seed: int


@dataclass
class RoundResult:
    target_step: int
    predicted: float
    truth: float
    agent_err: float
    baseline_err: float
    skill: float
    done: bool


def default_forecast_config() -> ForecastConfig:
    # tiny_transformer/lr=3e-3 在 modular_addition 上是延迟 grokking 曲线:
    # 长平台(acc≈0.16)后突然起跳到 1.0 再饱和 —— 动力学丰富,线性外推会被
    # 平台骗、在饱和处过冲。mlp 因 grok 太快(2 步到 1.0)退化,故不用作默认。
    return ForecastConfig(
        arch="tiny_transformer",
        spec=DatasetSpec(family="modular_addition", n_samples=300, modulus=7),
        lr=3e-3, max_steps=300, eval_every=20, seed=0,
    )


class ForecastEpisode:
    def __init__(self, config: ForecastConfig, cursor: int = 1,
                 skill_sum: float = 0.0, rounds: int = 0):
        self.config = config
        self.cursor = cursor
        self.skill_sum = skill_sum
        self.rounds = rounds
        bank = CurveBank(config.spec, max_steps=config.max_steps,
                         eval_every=config.eval_every, seed=config.seed)
        cand = Candidate(id=f"{config.arch}_{config.lr:g}", arch=config.arch, lr=config.lr)
        self._curve = bank.curve(cand)  # 隐藏曲线(真跑一次)

    def _last_index(self) -> int:
        return len(self._curve.steps) - 1

    def observed(self) -> dict:
        return {
            "steps": self._curve.steps[: self.cursor + 1],
            "values": self._curve.acc[: self.cursor + 1],
        }

    def next_step(self) -> int:
        return self._curve.steps[self.cursor + 1]

    def is_done(self) -> bool:
        return self.cursor >= self._last_index()

    def predict(self, value: float) -> RoundResult:
        target_idx = self.cursor + 1
        target_step = self._curve.steps[target_idx]
        truth = self._curve.acc[target_idx]
        obs = self.observed()
        baseline = linear_extrapolate(obs["steps"], obs["values"], target_step)
        agent_err = abs(value - truth)
        baseline_err = abs(baseline - truth)
        skill = skill_score(agent_err, baseline_err)
        self.skill_sum += skill
        self.rounds += 1
        self.cursor = target_idx
        return RoundResult(target_step, value, truth, agent_err, baseline_err, skill,
                           done=self.is_done())

    def score(self) -> float:
        return self.skill_sum / max(self.rounds, 1)
