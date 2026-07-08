#!/usr/bin/env python3
"""生成 ArchitectureIQ 的 Harbor 数据集(任务目录树)。

用法:  python harbor/generate_tasks.py
重新生成 harbor/tasks/ 下的每个任务目录 + dataset.toml。

每个任务目录是自包含的 Harbor 任务:
  architectureiq__<id>/
    task.toml            预算(cpus/memory,纯 CPU、gpus=0)
    instruction.md       给 agent 的提示(用 CLI 玩、把答案写进 submission.json)
    environment/Dockerfile   FROM 基础镜像 + 预开局(隐藏实例)
    tests/test.sh        评分时才挂载(对 agent 隐藏),调用独立评分写 reward.txt
    tests/verify.py      薄封装:读 submission + hidden → architectureiq.harbor.score_submission
    tests/meta/hidden.json   隐藏答案(仅 verifier 可见)
"""
import json
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(HERE, "tasks")

# 共享:评分时运行的 verifier(所有任务一致,task_id 从 hidden.json 读)
VERIFY_PY = '''\
import json
import os

from architectureiq.harbor import score_submission

with open("/tests/meta/hidden.json") as f:
    hidden = json.load(f)
task_id = hidden["task_id"]
try:
    with open("/workspace/submission.json") as f:
        submission = json.load(f)
except Exception:
    submission = {}
reward = float(score_submission(task_id, hidden, submission))
os.makedirs("/logs/verifier", exist_ok=True)
with open("/logs/verifier/reward.txt", "w") as f:
    f.write(str(reward))
print("reward", reward)
'''

TEST_SH = '''\
#!/bin/bash
# Harbor verifier —— 评分时才挂载在 /tests/,对 agent 隐藏。
# 重置 PATH,防 agent 在 /workspace 留 python shim 遮蔽系统解释器。
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
set -uo pipefail
mkdir -p /logs/verifier
PY=$(command -v python3 || command -v python)
"${PY}" /tests/verify.py > /logs/verifier/verify.log 2>&1
if [ ! -s /logs/verifier/reward.txt ]; then
    echo "0.0" > /logs/verifier/reward.txt
fi
exit 0
'''

TASK_TOML = '''\
schema_version = "1.1"

[task]
name = "architectureiq/{id}"
description = "ArchitectureIQ: {title}"
authors = [{{ name = "ArchitectureIQ", email = "lpeihang18@gmail.com" }}]
keywords = ["architecture-intuition", "cpu-only", "{keyword}"]

[metadata]
difficulty = "{difficulty}"
category = "architecture-intuition"
difficulty_explanation = "交互式实验设计任务。见 instruction.md。"

[agent]
timeout_sec = 1800

[verifier]
timeout_sec = 600

[environment]
os = "linux"
build_timeout_sec = 1800
cpus = 2
memory_mb = 4096
storage_mb = 8192
gpus = 0
allow_internet = false
'''

DOCKERFILE = '''\
# FROM 预构建的基础镜像(见 harbor/README.md 的 docker build 步骤)。
FROM architectureiq-harbor:base
RUN mkdir -p /workspace && \\
    {setup}
WORKDIR /workspace
'''

TASKS = [
    {
        "id": "fingerprint-lever",
        "task_id": "fingerprint_lever",
        "title": "架构指纹 · 真实杠杆(设计数据集分开优化器)",
        "keyword": "stimulus-design",
        "difficulty": "medium",
        "setup": "python -m architectureiq fpl-init --run-dir /workspace/game --lever optimizer",
        "hidden": {
            "task_id": "fingerprint_lever", "lever_family": "optimizer",
            "correct_margin": 0.22, "ref_seeds": [10, 11, 12],
            "query_seeds": [13, 14, 15], "probe_steps": 80,
        },
        "submission_example": {"family": "modular_addition", "n_samples": 300, "modulus": 7},
        "instruction": (
            "你要**设计一个数据集**,让优化器家族 {adam, sgd, rmsprop} 在其上训练产生的"
            "「结构优势」签名彼此**可区分**(margin ≥ 0.22)。默认/random 数据会让签名塌成"
            "一团。用 CLI 在 `/workspace/game` 上试探:\n"
            "- `python -m architectureiq fpl-observe --run-dir /workspace/game`\n"
            "- `python -m architectureiq fpl-probe --run-dir /workspace/game --family <F> "
            "[--n-samples N --modulus M]`\n"
            "然后把你**最终选定的数据集设计**写进 `/workspace/submission.json`。"
        ),
    },
    {
        "id": "blackbox-lever",
        "task_id": "blackbox_lever",
        "title": "黑盒鉴定 · 真实杠杆(认出隐藏优化器)",
        "keyword": "identification",
        "difficulty": "medium",
        "setup": "python -m architectureiq bbl-init --run-dir /workspace/game --family optimizer --hidden sgd",
        "hidden": {"task_id": "blackbox_lever", "lever_family": "optimizer", "hidden_value": "sgd"},
        "submission_example": {"guess": "sgd"},
        "instruction": (
            "环境藏了一个真实优化器(∈ {adam, sgd, rmsprop})。**设计探测数据集**让它的"
            "动力学差异显形,认出它。用 CLI 在 `/workspace/game` 上:\n"
            "- `python -m architectureiq bbl-observe --run-dir /workspace/game`\n"
            "- `python -m architectureiq bbl-probe --run-dir /workspace/game --family <F> [...]`\n"
            "把你的猜测写进 `/workspace/submission.json`。"
        ),
    },
    {
        "id": "diagnostic",
        "task_id": "diagnostic",
        "title": "训练医生 · 诊断台(查 observable 定病因)",
        "keyword": "diagnosis",
        "difficulty": "medium",
        "setup": "python -m architectureiq dx-init --run-dir /workspace/game --pathology vanishing_grad",
        "hidden": {"task_id": "diagnostic", "pathology": "vanishing_grad"},
        "submission_example": {"cause": "vanishing_grad"},
        "instruction": (
            "给你一条**病态**训练曲线,病因 ∈ {lr_too_low, dead_relu, vanishing_grad},"
            "它们在 loss 曲线上几乎一样。花预算**查 observable**(不同病因在不同 observable "
            "上留签名)定因。用 CLI 在 `/workspace/game` 上:\n"
            "- `python -m architectureiq dx-observe --run-dir /workspace/game`\n"
            "- `python -m architectureiq dx-query --run-dir /workspace/game --observable "
            "<grad_norm|weight_norm|dead_fraction|per_layer_grad>`\n"
            "把诊断写进 `/workspace/submission.json`。"
        ),
    },
    {
        "id": "windtunnel",
        "task_id": "windtunnel",
        "title": "预算锦标赛 · 风洞(跨尺度选最优)",
        "keyword": "budgeted-selection",
        "difficulty": "hard",
        "setup": "python -m architectureiq wt-init --run-dir /workspace/game --seed 3",
        "hidden": {"task_id": "windtunnel", "seed": 3, "regret_threshold": 0.05},
        "submission_example": {"candidate": "A"},
        "instruction": (
            "N 个候选竞争,你要选出**大尺度**上最优的那个。差异随尺度涌现——小尺度纯噪声"
            "骗人。花预算在各尺度跑代理实验,别信小尺度。用 CLI 在 `/workspace/game` 上:\n"
            "- `python -m architectureiq wt-observe --run-dir /workspace/game`\n"
            "- `python -m architectureiq wt-run --run-dir /workspace/game --candidate <C> "
            "--scale <k>`\n"
            "把你押注的候选写进 `/workspace/submission.json`。"
        ),
    },
]


def _write(path, content, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def main():
    names = []
    for t in TASKS:
        d = os.path.join(TASKS_DIR, f"architectureiq__{t['id']}")
        _write(os.path.join(d, "task.toml"),
               TASK_TOML.format(id=t["id"], title=t["title"], keyword=t["keyword"],
                                difficulty=t["difficulty"]))
        instr = (
            f"# ArchitectureIQ · {t['title']}\n\n{t['instruction']}\n\n"
            f"## 提交格式\n把最终答案写进 `/workspace/submission.json`,例如:\n\n"
            f"```json\n{json.dumps(t['submission_example'], ensure_ascii=False)}\n```\n\n"
            f"评分 = 正确性(0/1),由隐藏答案独立重算(见 architectureiq.harbor)。\n"
        )
        _write(os.path.join(d, "instruction.md"), instr)
        _write(os.path.join(d, "environment", "Dockerfile"),
               DOCKERFILE.format(setup=t["setup"]))
        _write(os.path.join(d, "tests", "test.sh"), TEST_SH, executable=True)
        _write(os.path.join(d, "tests", "verify.py"), VERIFY_PY)
        _write(os.path.join(d, "tests", "meta", "hidden.json"),
               json.dumps(t["hidden"], ensure_ascii=False, indent=2) + "\n")
        names.append(f"architectureiq/{t['id']}")

    manifest = ['# ArchitectureIQ Harbor 数据集清单',
                '# 重新生成: python harbor/generate_tasks.py', '',
                '[dataset]',
                'name = "architectureiq/architectureiq"',
                'description = "ArchitectureIQ —— 测 LLM 架构直觉的交互式 CPU-only benchmark。"',
                'authors = [{ name = "ArchitectureIQ", email = "lpeihang18@gmail.com" }]',
                'keywords = ["architecture-intuition", "cpu-only", "interactive"]', '']
    for n in names:
        manifest += ['[[tasks]]', f'name = "{n}"', '']
    _write(os.path.join(TASKS_DIR, "dataset.toml"), "\n".join(manifest))
    print(f"生成 {len(TASKS)} 个 Harbor 任务 → {TASKS_DIR}")


if __name__ == "__main__":
    main()
