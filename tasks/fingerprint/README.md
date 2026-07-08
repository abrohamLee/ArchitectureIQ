# ArchitectureIQ · 架构指纹

## 目标
设计一个数据集,使 4 个架构(`mlp` / `tiny_transformer` / `gru` / `cnn1d`)在其上训练
产生的「结构优势」签名彼此**可区分**——只看训练行为就能反推是哪个架构。你必须懂每个
架构 inductive bias 的因果机制,才能构造出让机制显形的刺激。

## 你的动作(通过 shell)
- `python -m architectureiq init --run-dir <DIR>` —— 开局,打印预算与规则。
- `python -m architectureiq observe --run-dir <DIR>` —— 查当前预算/已花/规则。
- `python -m architectureiq probe --run-dir <DIR> --family <F> [--n-samples N --modulus M --n-bits B --label-noise L]`
  —— 花预算试探一个数据集,返回可区分度(accuracy, margin)与剩余预算。
- `python -m architectureiq commit --run-dir <DIR> --family <F> [...]` —— 最终提交,用
  held-out seed 评分。

## DSL(数据集族)
- `modular_addition`:输入 (a,b) one-hot,标签 (a+b) mod m。参数 `modulus`。
- `parity`:输入 n_bits 位,标签位和奇偶。参数 `n_bits`。
- `random`:同 modular_addition 输入,但标签随机(无结构)。

## 评分(RHAE-ML)
`score = 1[margin ≥ correct_margin] × min(1, human_steps / 你花的 steps)²`
- 先得让签名**可区分**(margin 过阈值),否则 0 分——**难 ≠ 可区分,交无结构数据得零**。
- 再看**效率**:花的预算(train-steps)越少分越高,平方惩罚。凭 prior 少探针直接命中最高分。

## 预算
初始 `budget_steps`,每次 probe 花 `probe_cost`(见 observe)。commit 不需要额外探针也可以。
