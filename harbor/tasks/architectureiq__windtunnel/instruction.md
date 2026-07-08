# ArchitectureIQ · 预算锦标赛 · 风洞(跨尺度选最优)

N 个候选竞争,你要选出**大尺度**上最优的那个。差异随尺度涌现——小尺度纯噪声骗人。花预算在各尺度跑代理实验,别信小尺度。用 CLI 在 `/workspace/game` 上:
- `python -m architectureiq wt-observe --run-dir /workspace/game`
- `python -m architectureiq wt-run --run-dir /workspace/game --candidate <C> --scale <k>`
把你押注的候选写进 `/workspace/submission.json`。

## 提交格式
把最终答案写进 `/workspace/submission.json`,例如:

```json
{"candidate": "A"}
```

评分 = 正确性(0/1),由隐藏答案独立重算(见 architectureiq.harbor)。
