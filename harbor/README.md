# ArchitectureIQ on Harbor

把 ArchitectureIQ 的交互式任务打包成 [Harbor](https://github.com/harbor-framework/harbor)
数据集。任何 Harbor agent(`claude-code`、`codex`、`terminus-2`…)一条命令即可评测。

**纯 CPU**——不需要 GPU / nvidia toolkit(所有任务在玩具尺度上跑真实杠杆)。

## 快速开始

前置:Docker + Harbor 已安装。

```bash
# 1. 构建基础镜像(build context = 仓库根)
docker build -t architectureiq-harbor:base -f harbor/Dockerfile.base .

# 2. 跑一个真 agent
harbor run -c harbor/run.yaml -a claude-code -m anthropic/claude-opus-4-8

# 只跑某个任务
harbor run -c harbor/run.yaml -p harbor/tasks/architectureiq__fingerprint-lever
```

## 目录里有什么

```
harbor/
├── README.md
├── run.yaml              参考 Harbor 配置(CPU docker 环境 + claude-code)
├── Dockerfile.base       基础镜像:pip 装 architectureiq(CPU torch)
├── generate_tasks.py     任务树生成器(改任务后重跑它)
└── tasks/
    ├── dataset.toml      数据集清单
    └── architectureiq__<id>/
        ├── task.toml             预算(cpus=2, memory=4G, gpus=0, 无外网)
        ├── instruction.md        给 agent 的提示 + submission.json 格式
        ├── environment/Dockerfile  FROM base + 预开局(隐藏实例)
        └── tests/                评分时才挂载(对 agent 隐藏)
            ├── test.sh           重置 PATH → 跑 verify.py → 写 reward.txt
            ├── verify.py         读 submission + hidden → 独立评分
            └── meta/hidden.json  隐藏答案(仅 verifier 可见)
```

## 已打包的任务(4)

| 任务 | 能力 | agent 提交 |
|---|---|---|
| `fingerprint-lever` | 设计数据集分开真实优化器 | 数据集设计 `{family,n_samples,modulus}` |
| `blackbox-lever` | 认出隐藏优化器 | `{guess: "sgd"}` |
| `diagnostic` | 查 observable 定病因 | `{cause: "vanishing_grad"}` |
| `windtunnel` | 跨尺度选最优 | `{candidate: "A"}` |

加任务:在 `generate_tasks.py` 的 `TASKS` 加一条 + 在 `architectureiq.harbor.score_submission`
加一个分支,重跑生成器。

## 评分契约

Harbor 在评分时挂载 `/tests/`(对 agent 隐藏),`test.sh` 跑 `verify.py`:读
agent 写的 `/workspace/submission.json` + 隐藏答案 `meta/hidden.json`,调用
`architectureiq.harbor.score_submission` **独立重算** reward ∈ [0,1],写进
`/logs/verifier/reward.txt`。**verifier 不信 agent 自报的分数,用隐藏答案独立验证。**

v1 用**正确性**(0/1)—— verifier 可独立验证、不可作弊。效率加权(RHAE 的 min(1,human/agent)²)
需要可信的预算追踪,列为后续。

## 已验证 / 未验证(诚实标注)

- ✅ **独立评分逻辑**:`tests/test_harbor.py` 本地单测全过;setup→submission→评分的
  全链路(除 Docker/Harbor 编排)本地模拟跑通(猜对 1.0 / 猜错 0.0)。
- ⚠️ **未在真 Harbor + Docker 上端到端跑过**(开发机没有 Docker daemon / Harbor)。
  Dockerfile / task.toml / run.yaml 按 MLS-Bench 的 Harbor 适配模板写,首次真跑可能要微调。
- ⚠️ **识别类任务的答案藏在 workspace**:`blackbox-lever` / `diagnostic` / `windtunnel`
  预开局时隐藏答案写进了 `/workspace/game` 的 state(agent 有 shell、理论上可 `cat`)。
  当前假设**诚实 agent**(claude-code 等会玩游戏而非 grep 答案)。硬化方案(把隐藏实例
  移到 agent 够不到的 task-server)是**已知后续**。`fingerprint-lever` 无隐藏答案、天然无泄漏。
