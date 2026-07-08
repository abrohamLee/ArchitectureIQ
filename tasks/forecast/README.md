# ArchitectureIQ · 滚动曲线预测

## 目标
一条隐藏的训练曲线会逐 checkpoint 揭示它的 acc。每一步,你要预测**下一个 checkpoint 的
acc**。懂训练动力学(曲线会饱和、会有平台、grokking 会在长平台后突然起跳)的模型,预测
得比只会线性外推的基线更准。

## 你的动作(通过 shell)
- `python -m architectureiq fc-init --run-dir <DIR>` —— 开局,打印已观测段与待预测步。
- `python -m architectureiq fc-observe --run-dir <DIR>` —— 查已观测曲线、下一待预测步、当前得分。
- `python -m architectureiq fc-predict --run-dir <DIR> --value <V>` —— 提交你对下一 checkpoint
  acc 的预测(0~1),环境揭真并给出本轮技巧分,自动前进一步。

## 评分
每轮 `skill = 1 − |你的误差| / |线性外推基线的误差|`;>0 表示你比线性外推准。最终分 = 各轮平均。
- acc 有界 [0,1] 且会饱和 —— 线性外推常在后期过冲。
- 平台期后可能突然 grokking 起跳 —— 线性外推会被平台骗;能预判起跳的模型占优。
- 反复 fc-predict 直到 done。
