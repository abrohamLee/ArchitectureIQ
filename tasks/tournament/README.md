# ArchitectureIQ · 预算分配锦标赛

## 目标
一组 candidate(不同 arch × 学习率)在同一数据集上竞争。你有一个 train-step **预算**,
要用最少的预算**挑出最终会训得最好的那个候选**。凭 architecture prior 提前判断哪种架构
适合这个数据集、哪条早期曲线值得押注,就能少花预算命中——打败只会对候选机械地逐轮砍半的
语义盲基线(Successive Halving)。

## 你的动作(通过 shell)
- `python -m architectureiq tour-init --run-dir <DIR>` —— 开局,打印候选、预算、规则。
- `python -m architectureiq tour-observe --run-dir <DIR>` —— 查预算/各候选已训步数。
- `python -m architectureiq tour-advance --run-dir <DIR> --candidate <ID> --steps <N>`
  —— 花预算把某候选多训 N 步,返回它训到当前步数的 acc。
- `python -m architectureiq tour-answer --run-dir <DIR> --candidate <ID>` —— 提交你选的
  最优候选,按 regret 与效率评分。

## 评分(RHAE-ML)
`score = 1[regret ≤ regret_threshold] × min(1, human_steps / 你花的 steps)²`
- regret = 真最优候选的最终 acc − 你选中候选的最终 acc。选错(regret 过大)= 0 分。
- 注意:answer 按候选的**真实最终潜力**评分,你不必把它训到底——只需正确**识别**它。
- 花的预算(steps)越少分越高,平方惩罚。

## 预算
初始 `budget_steps`;每次 tour-advance 花「新增步数」。省着探。
