"""③ 训练医生 · real tier:从真实 loss 曲线**早期检测发散**。

玩具版是"开 LR 处方";真数据只有发生过的 run,无法反事实重训,故 real tier 改为
**诊断**:给真实训练 loss 曲线,尽早判断它是否会发散(病态)。越早判对分越高(RHAE 效率)。

数据:Marin 扫描(optimizer-scaling/BatchSize/Lr_datasize/Switch-Optimizer)的 train/loss
曲线,天然含大量炸掉的 run(loss 从最低点冲高)。自动标注:final − min(loss) > 阈值 = 病态。
"""
from dataclasses import dataclass

from architectureiq.realcurvebank import filter_ids, get_curve
from architectureiq.scoring import rhae_ml_score

_REVEAL_COST = 10  # 每次多揭一段的成本(单位无关曲线长度)


def is_pathological(values: list[float], threshold: float = 0.5) -> bool:
    """final loss 明显高于全程最低点 = 发散/病态。"""
    return (values[-1] - min(values)) > threshold


def real_doctor_ids() -> list[str]:
    """可用真实 loss 曲线池(Marin train/loss)。"""
    return [i for i in filter_ids(metric="train/loss") if i.startswith("marin|")]


@dataclass(frozen=True)
class RealDoctorConfig:
    curve_id: str
    init_frac: float = 0.3        # 开局默认揭示前 30%
    reveal_frac: float = 0.1      # 每次 reveal 再揭 10%
    human_budget: int = 20        # 专家 ~2 次 reveal 就能判
    threshold: float = 0.5


@dataclass
class RealDoctorResult:
    score: float
    correct: bool
    guess: str
    truth: str
    budget_spent: int
    frac_revealed: float


class RealDoctorEpisode:
    def __init__(self, config: RealDoctorConfig, cursor: int | None = None, budget_spent: int = 0):
        self.config = config
        self._curve = get_curve(config.curve_id)
        self._n = len(self._curve.acc)  # acc 字段存 loss 值
        self.cursor = cursor if cursor is not None else max(2, int(config.init_frac * self._n))
        self.budget_spent = budget_spent

    def _chunk(self) -> int:
        return max(1, int(self.config.reveal_frac * self._n))

    def observe(self) -> dict:
        return {
            "steps": self._curve.steps[: self.cursor],
            "loss": self._curve.acc[: self.cursor],
            "frac_revealed": round(self.cursor / self._n, 3),
            "budget_spent": self.budget_spent,
        }

    def reveal(self) -> dict:
        self.cursor = min(self.cursor + self._chunk(), self._n)
        self.budget_spent += _REVEAL_COST
        return self.observe()

    def _truth(self) -> str:
        return "pathological" if is_pathological(self._curve.acc, self.config.threshold) else "healthy"

    def diagnose(self, label: str) -> RealDoctorResult:
        truth = self._truth()
        correct = label == truth
        score = rhae_ml_score(correct, self.budget_spent, self.config.human_budget)
        return RealDoctorResult(score, correct, label, truth, self.budget_spent,
                                round(self.cursor / self._n, 3))
