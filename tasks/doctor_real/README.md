# ArchitectureIQ · 训练医生 (real tier)

## 目标
给你一条**真实**训练 loss 曲线(来自 Marin 的优化器/批量/LR 扫描实验),逐段揭示。判断
这个 run 是否会**发散**(病态:loss 从最低点冲高、训练崩掉)还是健康(loss 平滑下降)。
**越早判对分越高。**

玩具版是"开 LR 处方";真数据只有发生过的 run、无法反事实重训,所以 real tier 是**诊断**:
从 loss 曲线的形状读出它的命运。有的 run 一开始就卡在高位(早期可判),有的先降后炸
(晚发散、前段看不出)——尽早、尽准。

## 你的动作(通过 shell)
- `python -m architectureiq dr-real-observe --run-dir <DIR>` —— 查已揭示的 loss 前缀、已揭比例、预算。
- `python -m architectureiq dr-real-reveal --run-dir <DIR>` —— 多揭一段曲线(花预算)。
- `python -m architectureiq dr-real-diagnose --run-dir <DIR> --label healthy|pathological` —— 下诊断,评分。

(出题者用 `dr-real-init --run-dir <DIR> --curve <id>` 设好病例;你只看前缀,不知道答案。)

## 评分(RHAE-ML)
`score = 1[诊断正确] × min(1, human_budget / 你揭示花的预算)²`
- 判错 = 0。揭得越少分越高(平方惩罚)。凭前缀直接判对 = 满分。
- 非智能地板:总猜 "healthy"(多数类)——在真发散的 run 上必错。真本事是**尽早抓出发散**。
