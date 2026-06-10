"""Rescore eval results from raw_outputs.json without re-running inference.

Re-extracts answers from raw_generation using the (fixed) THINK_CLOSE_ONLY
strategy, then recomputes metrics via the original benchmark.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml
from src.tasks import BenchmarkFactory

THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_CLOSE_ONLY_RE = re.compile(r"^(.*?)</think>\s*(.*)$", re.DOTALL)


def re_extract(raw):
    raw = raw or ""
    m = THINK_TAG_RE.search(raw)
    if m:
        return m.group(1).strip(), THINK_TAG_RE.sub("", raw).strip()
    if "</think>" in raw:
        m = THINK_CLOSE_ONLY_RE.match(raw)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="e.g. results/lingoly/qwen3-4b-thinking-base_reasoning/default")
    ap.add_argument("--task", required=True, help="lingoly or mulr")
    ap.add_argument("--subset", default=None, help="subset name (for mulr: en/de/etc)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    raw = json.loads((rdir / "raw_outputs.json").read_text())
    print(f"Loaded {len(raw)} raw outputs from {rdir}")

    fixed = 0
    for o in raw:
        new_reason, new_gen = re_extract(o.get("raw_generation"))
        if new_gen != o.get("generation"):
            fixed += 1
        o["reasoning"] = new_reason
        o["generation"] = new_gen
    print(f"Re-extracted: {fixed}/{len(raw)} entries had updated generation")

    # Build benchmark for scoring
    tasks_cfg = yaml.safe_load((REPO_ROOT / "config/tasks.yaml").read_text())
    task_cfg = tasks_cfg["tasks"].get(args.task)
    if task_cfg is None:
        raise SystemExit(f"task {args.task} not found in config/tasks.yaml")
    subset = args.subset or rdir.name
    bench = BenchmarkFactory.get_benchmark(args.task, task_cfg, subset)

    preds = [o["generation"] for o in raw]
    refs = [o["target_text"] for o in raw]
    etypes = [o.get("eval_type") for o in raw]
    points = [o.get("points", 1) for o in raw]
    metrics = bench.evaluate(preds, refs, etypes, points)

    # Strip per-example arrays, attach to raw outputs
    per_ex = None
    for key in ("per_example_scores", "per_example_accuracy", "per_example_stats", "per_example_f1"):
        if key in metrics:
            per_ex = metrics.pop(key)
            break

    if isinstance(per_ex, list):
        for i, o in enumerate(raw):
            o["scores"] = {"accuracy" if "accuracy" in args.task or per_ex == per_ex else "score": per_ex[i]}

    print("=== Rescored metrics ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_metrics = rdir / "metrics_rescored.json"
    out_raw = rdir / "raw_outputs_rescored.json"
    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    out_raw.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    print(f"Saved -> {out_metrics}")
    print(f"Saved -> {out_raw}")


if __name__ == "__main__":
    main()
