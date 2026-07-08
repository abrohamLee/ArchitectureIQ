from dataclasses import dataclass

from architectureiq.curvebank import Candidate, CurveBank
from architectureiq.datasets import DatasetSpec
from architectureiq.scoring import rhae_ml_score


@dataclass(frozen=True)
class TournamentConfig:
    spec: DatasetSpec
    candidates: list[Candidate]
    budget_steps: int
    max_steps: int
    eval_every: int
    human_steps: int
    regret_threshold: float
    seed: int


@dataclass
class AdvanceResult:
    candidate_id: str
    trained_steps: int
    acc: float
    cost: int
    budget_remaining: int
    over_budget: bool


@dataclass
class AnswerResult:
    score: float
    correct: bool
    regret: float
    chosen_id: str
    agent_steps: int


def default_tournament_config() -> TournamentConfig:
    cands = []
    for arch in ["mlp", "tiny_transformer", "gru", "cnn1d"]:
        for lr in [1e-2, 3e-3]:
            cands.append(Candidate(id=f"{arch}_{lr:g}", arch=arch, lr=lr))
    return TournamentConfig(
        spec=DatasetSpec(family="modular_addition", n_samples=300, modulus=7),
        candidates=cands,
        budget_steps=1600,
        max_steps=200,
        eval_every=20,
        human_steps=400,
        regret_threshold=0.1,
        seed=0,
    )


def real_tournament_config(task: str = "piqa", shot: str = "zero-shot"):
    """真 Pythia 尺寸锦标赛:候选 = 该 task/shot 下所有模型变体的真 acc 曲线。

    回 (TournamentConfig, RealBank)。测的 prior 与玩具版不同:真 LLM 早期曲线不
    预测最终排名(全在瞎猜线抖),赢在懂 scale laws(大=好);语义盲 SH 会被早期
    噪声骗掉大模型。
    """
    from architectureiq.realcurvebank import RealBank, filter_ids

    ids = filter_ids(task=task, shot=shot, metric="acc")
    cands = [Candidate(id=cid, arch=cid.split("|")[1], lr=0.0) for cid in ids]
    bank = RealBank(ids)
    max_steps = bank.checkpoints()[-1]
    cfg = TournamentConfig(
        spec=DatasetSpec(family="modular_addition", n_samples=1, modulus=7),  # 占位,bank 提供数据
        candidates=cands,
        budget_steps=max_steps * len(cands),
        max_steps=max_steps,
        eval_every=1000,
        human_steps=max_steps,  # 专家 ~ 一次完整 run 的参照
        regret_threshold=0.02,
        seed=0,
    )
    return cfg, bank


class Tournament:
    def __init__(self, config: TournamentConfig, bank: CurveBank | None = None,
                 trained: dict[str, int] | None = None, budget_spent: int = 0):
        self.config = config
        self.bank = bank or CurveBank(
            config.spec, max_steps=config.max_steps,
            eval_every=config.eval_every, seed=config.seed,
        )
        self._by_id = {c.id: c for c in config.candidates}
        self._trained: dict[str, int] = (
            dict(trained) if trained is not None else {c.id: 0 for c in config.candidates}
        )
        self.budget_spent = budget_spent

    def snapshot(self) -> dict:
        return {"trained": dict(self._trained), "budget_spent": self.budget_spent}

    def advance(self, candidate_id: str, extra_steps: int) -> AdvanceResult:
        cand = self._by_id[candidate_id]
        cur = self._trained[candidate_id]
        target = min(cur + max(extra_steps, 0), self.config.max_steps)
        cost = target - cur
        remaining = self.config.budget_steps - self.budget_spent
        if cost > remaining:
            return AdvanceResult(candidate_id, cur, self.bank.acc_at(cand, cur), 0,
                                 max(remaining, 0), over_budget=True)
        self._trained[candidate_id] = target
        self.budget_spent += cost
        return AdvanceResult(
            candidate_id, target, self.bank.acc_at(cand, target), cost,
            self.config.budget_steps - self.budget_spent, over_budget=False,
        )

    def best_candidate_id(self) -> str:
        return max(self.config.candidates, key=lambda c: self.bank.final_acc(c)).id

    def answer(self, candidate_id: str) -> AnswerResult:
        chosen = self._by_id[candidate_id]
        best_final = max(self.bank.final_acc(c) for c in self.config.candidates)
        regret = best_final - self.bank.final_acc(chosen)
        correct = regret <= self.config.regret_threshold
        score = rhae_ml_score(correct, self.budget_spent, self.config.human_steps)
        return AnswerResult(score, correct, regret, candidate_id, self.budget_spent)
