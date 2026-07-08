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
