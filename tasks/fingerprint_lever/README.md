# ArchitectureIQ · 架构指纹 (Tier 2 · 真实杠杆)

## 目标
设计一个数据集,让一个**真实杠杆家族**的答案(默认:优化器 `{adam, sgd, rmsprop}`;
也支持激活 `{relu, tanh, gelu}`)在其上训练产生的「结构优势」签名彼此**最大程度可区分**
——理想情况只看训练行为就能唯一反推是哪个杠杆。

这是玩具指纹(分开 4 个架构)的真实杠杆版:杠杆是真的(优化器/激活是真实研究组件、
MLS 旗舰题材),尺度是玩具的(便宜、可 retrain)。关键:**默认 / random 数据会让不同
杠杆的签名塌成一团、分不开;必须设计好数据,让这些杠杆的动力学差异显形。**

## 你的动作(通过 shell)
- `python -m architectureiq fpl-init --run-dir <DIR> [--lever optimizer|activation]` —— 开局,打印杠杆家族、候选答案、预算、可区分阈值。
- `python -m architectureiq fpl-observe --run-dir <DIR>` —— 查杠杆家族与预算。
- `python -m architectureiq fpl-probe --run-dir <DIR> --family <F> [--n-samples N --modulus M ...]`
  —— 花预算试探一个数据集设计,返回可区分度(accuracy, margin)。
- `python -m architectureiq fpl-commit --run-dir <DIR> --family <F> [...]` —— 最终提交,用 held-out seed 评分。

`--family` 是**数据集**的族(`modular_addition` / `parity` / `random`),你设计的对象;
`--lever` 是要**分开的杠杆**家族,出题者设定。

## 评分(RHAE-ML)
`score = 1[可区分: margin ≥ correct_margin] × min(1, human_steps / 花的 steps)²`
- 签名分不开(margin 不过阈值)= 0 分。**难 ≠ 可区分,交无结构数据得零。**
- 非智能地板:随机数据集(签名塌陷 margin≈0.1,不过阈值)。

## litmus
必须**设计**能 engage 优化器/激活特定动力学差异的数据(结构数据 margin≈0.4;random≈0.1)。
测的是"能不能构造出让真实杠杆机制显形的刺激",不是"背知识"。
