from dataclasses import dataclass

from architectureiq.curvebank import Candidate, CurveBank
from architectureiq.datasets import DatasetSpec
from architectureiq.scoring import rhae_ml_score

_SICK_LR = {"too_high": 1.0, "too_low": 1e-4}


@dataclass(frozen=True)
class DoctorConfig:
    spec: DatasetSpec
    arch: str
    sick_lr: float
    grid: list[float]
    max_steps: int
    eval_every: int
    budget_steps: int
    cure_acc: float
    human_steps: int
    seed: int


@dataclass
class TreatResult:
    lr: float
    final_acc: float
    cured: bool
    cost: int
    budget_remaining: int
    over_budget: bool


@dataclass
class DiagnoseResult:
    score: float
    correct: bool
    chosen_lr: float
    final_acc: float
    agent_steps: int


def doctor_config(pathology: str = "too_high") -> DoctorConfig:
    return DoctorConfig(
        spec=DatasetSpec(family="modular_addition", n_samples=300, modulus=7),
        arch="mlp",
        sick_lr=_SICK_LR[pathology],
        grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        max_steps=120,
        eval_every=20,
        budget_steps=600,
        cure_acc=0.9,
        human_steps=120,
        seed=0,
    )


class DoctorEpisode:
    def __init__(self, config: DoctorConfig, budget_spent: int = 0):
        self.config = config
        self.budget_spent = budget_spent
        self.bank = CurveBank(config.spec, max_steps=config.max_steps,
                              eval_every=config.eval_every, seed=config.seed)

    def _cand(self, lr: float) -> Candidate:
        return Candidate(id=f"{self.config.arch}_{lr:g}", arch=self.config.arch, lr=lr)

    def sick_curve(self) -> dict:
        c = self.bank.curve(self._cand(self.config.sick_lr))
        return {"steps": list(c.steps), "loss": list(c.loss), "acc": list(c.acc)}

    def treat_cost(self) -> int:
        return self.config.max_steps

    def treat(self, lr: float) -> TreatResult:
        cost = self.treat_cost()
        remaining = self.config.budget_steps - self.budget_spent
        if cost > remaining:
            return TreatResult(lr, 0.0, False, 0, max(remaining, 0), over_budget=True)
        final_acc = self.bank.final_acc(self._cand(lr))
        self.budget_spent += cost
        return TreatResult(lr, final_acc, final_acc >= self.config.cure_acc, cost,
                           self.config.budget_steps - self.budget_spent, over_budget=False)

    def commit(self, lr: float) -> DiagnoseResult:
        final_acc = self.bank.final_acc(self._cand(lr))
        correct = final_acc >= self.config.cure_acc
        score = rhae_ml_score(correct, self.budget_spent, self.config.human_steps)
        return DiagnoseResult(score, correct, lr, final_acc, self.budget_spent)
