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
