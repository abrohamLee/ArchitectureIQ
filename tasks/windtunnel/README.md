# ArchitectureIQ · 预算锦标赛 (Tier 2 · 风洞)

## 目标
N 个候选 config 竞争,你要选出**在大尺度上最终最优**的那个。但你不能直接看大尺度——
只能花预算在各**尺度**跑便宜的代理实验,看某个候选在某尺度的表现。用最少预算押对。

**关键(跨尺度陷阱)**:候选间的差异**随尺度涌现**——小尺度上大家挤在一起、基本是噪声
(argmax 是随机的、骗人);尺度越大差距越拉开、越接近真值。**信小尺度会翻车**;聪明的
做法是用便宜的小/中尺度粗筛,再花贵的大尺度确认。这正是研究者用小实验决定大 run 的日常。

## 你的动作(通过 shell)
- `python -m architectureiq wt-init --run-dir <DIR> [--seed S]` —— 开局,打印候选、各尺度成本、预算。
- `python -m architectureiq wt-observe --run-dir <DIR>` —— 查候选/尺度/预算/已花。
- `python -m architectureiq wt-run --run-dir <DIR> --candidate <C> --scale <k>` —— 花预算在尺度 k
  跑候选 C 的代理实验(小尺度便宜、大尺度贵),返回它在该尺度的值。
- `python -m architectureiq wt-commit --run-dir <DIR> --candidate <C>` —— 押注大尺度最优候选,评分。

## 评分(RHAE-ML)
`score = 1[大尺度 regret ≤ 阈值] × min(1, human_budget / 你花的预算)²`
- regret = 真最优候选的大尺度值 − 你选中的。押错 = 0。
- 非智能地板:只读最小尺度、选 argmax(被涌现噪声骗)。

## litmus
每局的小→大迁移隐藏在数据里,小尺度纯噪声——**背 scaling law 没用**,必须**跑实验**
爬到足够大的尺度、才能可靠看出真赢家。
