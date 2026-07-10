"""runlib/ 查询层:按配置归并 seed、取曲线、取终点。"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class Registry:
    def __init__(self, runlib_dir: Path) -> None:
        self.dir = Path(runlib_dir)

    def runs(self) -> list[dict]:
        out = []
        for meta_path in sorted(self.dir.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text())
            config = json.loads((meta_path.parent / "config.json").read_text())
            out.append({"run_id": meta["run_id"], "config": config, "meta": meta})
        return out

    def groups(self) -> dict[str, list[dict]]:
        g = defaultdict(list)
        for r in self.runs():
            g[r["meta"]["config_hash"]].append(r)
        return dict(g)

    def curve(self, run_id: str, metric: str) -> list[tuple[int, float]]:
        run_dir = self.dir / run_id
        for fname in ("log.jsonl", "eval.jsonl"):
            rows = [json.loads(l) for l in (run_dir / fname).read_text().splitlines()]
            if rows and metric in rows[0]:
                return [(r["step"], float(r[metric])) for r in rows]
        raise KeyError(f"metric {metric} not found in {run_id}")

    def final(self, config_hash: str) -> list[float]:
        return [r["meta"]["final_val_ce"] for r in self.groups()[config_hash]]
