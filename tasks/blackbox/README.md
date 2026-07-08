# ArchitectureIQ · 黑盒架构鉴定

## 目标
环境藏了一个架构(是 `mlp` / `tiny_transformer` / `gru` 之一,不告诉你)。你要**设计探测数据集**
把它逼出原形:环境会返回池中每个候选架构在你数据上的「结构优势」参考签名,以及黑盒自己的
mystery 签名。比对 mystery 最像哪个参考,猜出黑盒是谁。用最少探测次数。

关键:**好的探测数据集(有结构、能 engage 各架构不同的 inductive bias)会让签名清晰可分,
一眼看出 mystery 是谁;random 数据集会让所有签名塌成一团、难以区分。** 懂架构机制才能设计好探测。

## 你的动作(通过 shell)
- `python -m architectureiq bb-init --run-dir <DIR>` —— 开局,打印候选池、预算、规则。
- `python -m architectureiq bb-observe --run-dir <DIR>` —— 查候选池与预算。
- `python -m architectureiq bb-probe --run-dir <DIR> --family <F> [--n-samples N --modulus M]`
  —— 花预算探测,返回各候选参考签名 + 黑盒 mystery 签名。
- `python -m architectureiq bb-guess --run-dir <DIR> --arch <ARCH>` —— 猜黑盒是哪个架构,评分。

## 评分(RHAE-ML)
`score = 1[猜对] × min(1, human_steps / 你花的 steps)²`
- 猜错 = 0;探测越少分越高(平方惩罚)。一次好探测就认出 = 满分。
