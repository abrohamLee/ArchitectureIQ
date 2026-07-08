# ArchitectureIQ · 黑盒鉴定 (Tier 2 · 真实杠杆)

## 目标
环境藏了一个**真实研究杠杆的答案**——默认是**优化器**(∈ `{adam, sgd, rmsprop}`),不告诉你。
你要**设计探测数据集**把它逼出原形:环境返回池中每个候选优化器在你数据上的「结构优势」
参考签名,以及黑盒自己的 mystery 签名。比对 mystery 最像哪个,猜出黑盒用了哪个优化器。

这是玩具黑盒(认架构)的真实杠杆版——杠杆是真的(优化器是真实研究组件、MLS 旗舰题材),
尺度是玩具的(便宜、可 retrain)。关键:**默认 / random 探测会让签名塌成一团、认不出;
必须设计好探测,让不同优化器的收敛动力学差异显形。** 懂优化器机制才能设计好探测。

## 你的动作(通过 shell)
- `python -m architectureiq bbl-init --run-dir <DIR> [--family optimizer --hidden <值>]` —— 开局(出题者设答案),打印候选池、预算、规则。
- `python -m architectureiq bbl-observe --run-dir <DIR>` —— 查候选池与预算。
- `python -m architectureiq bbl-probe --run-dir <DIR> --family <F> [--n-samples N --modulus M]`
  —— 花预算探测,返回各候选参考签名 + 黑盒 mystery 签名。
- `python -m architectureiq bbl-guess --run-dir <DIR> --value <优化器>` —— 猜黑盒用了哪个,评分。

## 评分(RHAE-ML)
`score = 1[猜对] × min(1, human_steps / 你花的 steps)²`
- 猜错 = 0;探测越少分越高(平方惩罚)。一次好探测就认出 = 满分。
- 非智能地板:固定用 random 数据集探测(签名塌陷 ≈ 瞎猜)。

## litmus
被动看给定曲线**认不出**(默认探针签名塌一起);必须**主动设计**探针让优化器动力学差异
显形——这测的是"能不能设计出能问出答案的实验",而不是"背知识"。
