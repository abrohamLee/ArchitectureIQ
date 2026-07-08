# ArchitectureIQ · 架构指纹 · 真实杠杆(设计数据集分开优化器)

你要**设计一个数据集**,让优化器家族 {adam, sgd, rmsprop} 在其上训练产生的「结构优势」签名彼此**可区分**(margin ≥ 0.22)。默认/random 数据会让签名塌成一团。用 CLI 在 `/workspace/game` 上试探:
- `python -m architectureiq fpl-observe --run-dir /workspace/game`
- `python -m architectureiq fpl-probe --run-dir /workspace/game --family <F> [--n-samples N --modulus M]`
然后把你**最终选定的数据集设计**写进 `/workspace/submission.json`。

## 提交格式
把最终答案写进 `/workspace/submission.json`,例如:

```json
{"family": "modular_addition", "n_samples": 300, "modulus": 7}
```

评分 = 正确性(0/1),由隐藏答案独立重算(见 architectureiq.harbor)。
