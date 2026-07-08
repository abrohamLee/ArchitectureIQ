# ArchitectureIQ · 训练医生 · 诊断台(查 observable 定病因)

给你一条**病态**训练曲线,病因 ∈ {lr_too_low, dead_relu, vanishing_grad},它们在 loss 曲线上几乎一样。花预算**查 observable**(不同病因在不同 observable 上留签名)定因。用 CLI 在 `/workspace/game` 上:
- `python -m architectureiq dx-observe --run-dir /workspace/game`
- `python -m architectureiq dx-query --run-dir /workspace/game --observable <grad_norm|weight_norm|dead_fraction|per_layer_grad>`
把诊断写进 `/workspace/submission.json`。

## 提交格式
把最终答案写进 `/workspace/submission.json`,例如:

```json
{"cause": "vanishing_grad"}
```

评分 = 正确性(0/1),由隐藏答案独立重算(见 architectureiq.harbor)。
