# ArchitectureIQ · 训练医生 (Tier 2 · 诊断台)

## 目标
给你一条**病态**训练曲线。多个候选病因(`lr_too_low` / `dead_relu` / `vanishing_grad`)
**在 loss 曲线上长得几乎一样**——都卡在高位平台,光看曲线诊断不出。你要花预算**查诊断
信号**(observable):不同病因在**不同** observable 上留下清晰签名。查对信号后定因。

这是玩具医生("读 loss 形状开药")的 hard 版:病因故意做成 raw 曲线不可分,逼你**主动
查对诊断信号**——背"loss-spike=LR太高"没用。

## 可查的 observable(各不同价)
- `grad_norm`(总梯度范数,1)、`weight_norm`(总权重范数,1)
- `dead_fraction`(激活死亡率,1)—— dead ReLU 会飙高
- `per_layer_grad`(逐层梯度范数,2)—— 梯度消失时早层远小于晚层

## 你的动作(通过 shell)
- `python -m architectureiq dx-init --run-dir <DIR> [--pathology <病因>]` —— 开局(出题者设病因),打印病态曲线、候选病因、可查 observable。
- `python -m architectureiq dx-observe --run-dir <DIR>` —— 复查曲线与预算。
- `python -m architectureiq dx-query --run-dir <DIR> --observable <名>` —— 花预算查一个诊断信号。
- `python -m architectureiq dx-answer --run-dir <DIR> --cause <病因>` —— 下诊断,评分。

## 评分(RHAE-ML)
`score = 1[诊断正确] × min(1, human_budget / 你花的预算)²`
- 诊断错 = 0;查得越少分越高(平方惩罚)。查对最有信息量的 observable = 高效。
- 非智能地板:不查任何 observable、只猜多数病因(3 个里蒙对 1 个)。

## litmus
raw loss 曲线故意不可分,**必须查对 observable 才能定因**——测的是"会不会查对诊断信号"
(交互式因果诊断),不是"背病因知识"。没有单一 observable 能分开三者:`dead_fraction`
认出死 ReLU,`per_layer_grad` 认出梯度消失,排除法定位 LR 太低。
