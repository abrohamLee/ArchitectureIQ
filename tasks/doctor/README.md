# ArchitectureIQ · 训练医生

## 目标
给你一条**病态**训练曲线。读它的动力学诊断病因,并开出一个能治愈的学习率——越少试药越好。
- loss 曾冲高过起点 → LR **太高**(更新过冲),应调**低**。
- loss 卡在高位几乎不降(平台)→ LR **太低**,应调**高**。
好医生看一眼曲线就开对药,不必挨个试;只会网格搜遍所有 LR 的是庸医(地板)。

## 你的动作(通过 shell)
- `python -m architectureiq dr-init --run-dir <DIR> [--pathology too_high|too_low]` —— 开局,
  打印病态曲线、可选 LR 网格、预算。
- `python -m architectureiq dr-observe --run-dir <DIR>` —— 复查病态曲线与预算。
- `python -m architectureiq dr-treat --run-dir <DIR> --lr <X>` —— 花预算试一个 LR,返回它能否治愈。
- `python -m architectureiq dr-commit --run-dir <DIR> --lr <X>` —— 开出你的处方 LR,评分。

## 评分(RHAE-ML)
`score = 1[治愈: final_acc ≥ cure_acc] × min(1, human_steps / 你花的 steps)²`
- 没治好 = 0 分;试药越少分越高(平方惩罚)。凭诊断零试药开对药 = 满分。
