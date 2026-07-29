"""Official IOL-AI 2026 scoring helpers (mirrors competition metric.py).

Final score is the geometric mean of points-weighted aggregates on a 0-1 scale:

    chrf_mean = sum_i (points_i * mean_j chrF(pred_ij, gold_ij)) / sum_i points_i
    em_mean   = sum_i (points_i * mean_j EM(pred_ij, gold_ij))   / sum_i points_i
    score     = sqrt(chrf_mean * em_mean)

For eval_type="multi", each gold item is a list of acceptable alternatives.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any, List, Sequence, Union


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("’", "'").replace("‘", "'")
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    s = s.lower()
    s = " ".join(s.split())
    s = s.rstrip(".")
    s = s.strip()
    return s


def _char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    if len(s) < n:
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf_score(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    p_orders, r_orders = [], []
    for n in range(1, max_n + 1):
        h = _char_ngrams(hyp, n)
        r = _char_ngrams(ref, n)
        h_total = sum(h.values())
        r_total = sum(r.values())
        matches = sum((h & r).values())
        if h_total > 0:
            p_orders.append(matches / h_total)
        if r_total > 0:
            r_orders.append(matches / r_total)
    if not p_orders or not r_orders:
        return 0.0
    chr_p = sum(p_orders) / len(p_orders)
    chr_r = sum(r_orders) / len(r_orders)
    if chr_p == 0 and chr_r == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * chr_p + chr_r
    if denom == 0:
        return 0.0
    return (1 + b2) * chr_p * chr_r / denom


def item_scores(pred: Any, gold_alts: Sequence[Any]) -> tuple:
    """Score pred against acceptable gold strings. Returns (em, chrf)."""
    p = normalize_text(pred)
    best_em, best_cf = 0.0, 0.0
    for gold in gold_alts:
        g = normalize_text(gold)
        em = 1.0 if p == g and g != "" else 0.0
        cf = chrf_score(p, g)
        if em > best_em:
            best_em = em
        if cf > best_cf:
            best_cf = cf
    return best_em, best_cf


def parse_pred_items(pred_str: Any, n_items: int) -> List[str]:
    """Split a prediction into n_items answers (JSON list or newline-separated)."""

    def _pad(items):
        items = list(items)
        if len(items) < n_items:
            items += [""] * (n_items - len(items))
        return items[:n_items]

    if not pred_str or not str(pred_str).strip():
        return [""] * n_items
    s = str(pred_str).strip()

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return _pad([("" if x is None else str(x)).strip() for x in parsed])
    except Exception:
        pass

    lines = [re.sub(r"^\d+[\.\)]\s*", "", ln).strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return _pad(lines)


def aggregate(rows: Sequence[dict]) -> dict:
    """Aggregate row-level {em, cf, points} into score / chrf / exact_match (0-1)."""
    wsum = 0.0
    em_acc = 0.0
    chrf_acc = 0.0
    for row in rows:
        w = float(row.get("points", 1.0) or 0.0)
        wsum += w
        em_acc += w * row["em"]
        chrf_acc += w * row["cf"]
    if wsum == 0:
        return {"score": 0.0, "chrf": 0.0, "exact_match": 0.0}
    em_mean = em_acc / wsum
    chrf_mean = chrf_acc / wsum
    score = math.sqrt(max(em_mean, 0.0) * max(chrf_mean, 0.0))
    return {
        "score": round(score, 4),
        "chrf": round(chrf_mean, 4),
        "exact_match": round(em_mean, 4),
    }


GoldItem = Union[str, List[str]]


def score_problem(
    pred_str: Any,
    gold_list: Sequence[GoldItem],
    eval_type: str = "single",
    points: float = 1.0,
) -> dict:
    """Score one problem (row). Returns {em, cf, points} plus per-item lists."""
    is_multi = (eval_type or "single").lower() == "multi"
    pred_items = parse_pred_items(pred_str, len(gold_list))

    em_scores, cf_scores = [], []
    for pred_item, gold_item in zip(pred_items, gold_list):
        alts = gold_item if is_multi else [gold_item]
        if not isinstance(alts, (list, tuple)):
            alts = [alts]
        em, cf = item_scores(pred_item, alts)
        em_scores.append(em)
        cf_scores.append(cf)

    n = len(em_scores) or 1
    return {
        "em": sum(em_scores) / n,
        "cf": sum(cf_scores) / n,
        "points": float(points) if points is not None else 1.0,
        "item_em": em_scores,
        "item_chrf": cf_scores,
        "pred_items": pred_items,
    }


def canon_id(value: Any) -> str:
    """Canonicalize problem ids (strip leading zeros), matching competition metric."""
    s = str(value).strip().lstrip("0")
    return s if s else "0"


def parse_gold_answer(raw: Any) -> list:
    """Parse solution.csv answer cell into a Python list."""
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    return json.loads(s)
