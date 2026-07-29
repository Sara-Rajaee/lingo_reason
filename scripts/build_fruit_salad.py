#!/usr/bin/env python3
"""Build the 'fruit_salad' IOL-2026 ensemble solution from the self-hosted model runs.

- oracle: for each problem, pick the model whose answer scores best (EM then chrF,
  using gold) -> the ceiling of the open-model ensemble.
- best-of-n single solution: the resulting one-answer-per-problem submission.

Writes fruit_salad/{submission.json, oracle_report.json, README.md}. Safe to re-run
(the cron calls it once the eval battery is complete).
"""
import sys, os, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml
from tasks import BenchmarkFactory
from src.iol_ai_metric import score_problem, aggregate, canon_id, parse_pred_items

MODELS = ["gemma4-31b-it", "qwen3.6-27b", "deepseek-r1-32b", "glm-4.7-flash"]
TASK = "iol-2026"
OUT = ROOT + "/fruit_salad"


def main():
    os.chdir(ROOT)
    cfg = yaml.safe_load(open("config/tasks.yaml"))["tasks"][TASK]
    ex = BenchmarkFactory.get_benchmark("iol2026", cfg, "default").load_data()
    gold = {canon_id(e.id): e for e in ex}

    gens = {}
    for m in MODELS:
        fs = glob.glob(f"results/{TASK}/{m}_*reasoning*/default/raw_outputs.json")
        if fs:
            gens[m] = {canon_id(r["id"]): (r.get("generation") or "")
                       for r in json.load(open(fs[0]))}
    present = [m for m in MODELS if m in gens]
    if not present:
        print("no model results found under results/%s yet" % TASK)
        return 1

    submission, oracle_rows, wins = [], [], {m: 0 for m in present}
    for cid, e in gold.items():
        best = None
        for m in present:
            g = gens[m].get(cid, "")
            sp = score_problem(g, e.answer, e.eval_type, e.points)
            cand = (sp["em"], sp["cf"], m, parse_pred_items(g, len(e.answer)))
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        em, cf, m, items = best
        wins[m] += 1
        submission.append({"id": e.id, "answer": items, "source_model": m})
        oracle_rows.append({"em": em, "cf": cf, "points": e.points})

    agg = aggregate(oracle_rows)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/submission.json", "w") as f:
        json.dump(submission, f, ensure_ascii=False, indent=1)
    json.dump(
        {"task": TASK, "n_problems": len(gold), "models": present,
         "oracle_score": agg, "per_model_wins": wins},
        open(f"{OUT}/oracle_report.json", "w"), indent=2, ensure_ascii=False,
    )
    open(f"{OUT}/README.md", "w").write(
        "# fruit_salad — IOL-2026 open-model ensemble\n\n"
        f"Oracle / best-of-n single solution over {len(present)} self-hosted open models:\n"
        f"{', '.join(present)}\n\n"
        f"- **Oracle score** (points-weighted geomean of EM.chrF): **{agg['score']}** "
        f"(EM {agg['exact_match']}, chrF {agg['chrf']}) over {len(gold)} problems.\n"
        f"- **Per-model contributions** (problems where each was the best pick): {wins}\n\n"
        "`submission.json` is the best-of-n single solution (one answer per problem).\n"
    )
    print("oracle score:", agg)
    print("per-model wins:", wins)
    print("wrote", os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
