# ArchitectureIQ · 黑盒鉴定 · 真实杠杆(认出隐藏优化器)

环境藏了一个真实优化器(∈ {adam, sgd, rmsprop})。**设计探测数据集**让它的动力学差异显形,认出它。用 CLI 在 `/workspace/game` 上:
- `python -m architectureiq bbl-observe --run-dir /workspace/game`
- `python -m architectureiq bbl-probe --run-dir /workspace/game --family <F> [...]`
把你的猜测写进 `/workspace/submission.json`。

## 提交格式
把最终答案写进 `/workspace/submission.json`,例如:

```json
{"guess": "sgd"}
```

评分 = 正确性(0/1),由隐藏答案独立重算(见 architectureiq.harbor)。
