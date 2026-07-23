"""Heuristic (non-LLM) baselines that compose answers from puzzle context."""

from __future__ import annotations

import hashlib
import random
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from .base_provider import BaseProvider

# Common English gloss tokens / function words seen in Linguini contexts.
_ENGLISH_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "my", "your", "our", "i", "you", "we",
    "this", "that", "these", "those", "not", "no", "yes", "as", "by", "at", "into",
    "about", "after", "before", "between", "through", "during", "without", "under",
    "over", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "should", "now", "also", "him", "me", "us", "who", "whom", "which", "what",
    "see", "say", "tell", "open", "trap", "obey", "believe", "play", "steal",
    "roast", "sew", "weave", "learn", "know", "push", "pull", "make", "smooth",
    "out", "fire", "songs", "hearts", "days", "places", "villages", "singing",
    "spirit", "hot", "coal", "part", "garden", "two", "day", "heart", "gardens",
    "many", "things", "grandchildren", "branches", "big", "whole", "world",
    "eyes", "vines", "water", "house", "man", "woman", "gave", "ate", "drank",
    "found", "brought", "help", "ask", "stop", "challenge", "see", "him", "them",
    "if", "we", "pl", "sg", "you_sg", "you_pl", "here", "are", "some", "words",
    "forms", "verbs", "sentences", "numerals", "phrases", "their", "english",
    "translations", "translation", "equivalents", "approximate", "values",
    "language", "dialect", "roman", "transcription", "transliteration", "latin",
    "letters", "below", "above", "following", "given", "consider", "fill",
    "blanks", "translate", "write", "digits", "one", "these", "has", "same",
    "value", "as", "numeral", "verb", "noun", "form", "word", "phrase",
}

_CONTEXT_RE = re.compile(
    r"(?is)Context:\s*(.*?)\n\s*Question:\s*(.*?)(?:\n\s*(?:Give your|Answer with|Use this)|\Z)"
)
_QUESTION_ONLY_RE = re.compile(
    r"(?is)Question:\s*(.*?)(?:\n\s*(?:Give your|Answer with|Use this)|\Z)"
)
_BLANK_RE = re.compile(r"^\((\d+)\)$")
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*\d+\s*[\.\)\:]\s*(.+)$")


def _parse_prompt(prompt: str) -> Tuple[str, str]:
    """Return (context, question) extracted from a Linguini-style prompt."""
    match = _CONTEXT_RE.search(prompt or "")
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = _QUESTION_ONLY_RE.search(prompt or "")
    if match:
        return "", match.group(1).strip()
    return "", (prompt or "").strip()


def _looks_english_phrase(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return False
    if " " in text.strip() or len(tokens) >= 2:
        english = sum(1 for t in tokens if t in _ENGLISH_WORDS)
        return english / len(tokens) >= 0.5
    return tokens[0] in _ENGLISH_WORDS


def _is_likely_gloss(text: str) -> bool:
    """True for English glosses / numeric values in the rightmost table column."""
    text = (text or "").strip()
    if not text:
        return True
    if _looks_english_phrase(text):
        return True
    if re.fullmatch(r"\d+", text):
        return True
    if re.fullmatch(r"[A-Za-z0-9\s\{\}_\-'’,.()/:;!?]+", text):
        return True
    return False


def _clean_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"^\d+[\.\)]\s*", "", token)
    token = token.strip(" \t\"'`")
    return token


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1] (exact match → 1)."""
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def nearest_neighbor(query: str, candidates: Sequence[str]):
    """Return the candidate most similar to ``query``."""
    if not candidates:
        return None
    best, best_score = candidates[0], -1.0
    for cand in candidates:
        score = similarity(query, cand)
        if score > best_score:
            best, best_score = cand, score
    return best


def parse_lexicon_rows(context: str) -> List[Dict]:
    """Parse context tables into lexicon rows: forms + optional gloss."""
    rows: List[Dict] = []
    for raw_line in (context or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 2:
                continue
            if _is_likely_gloss(parts[-1]):
                rows.append({"forms": parts[:-1], "gloss": parts[-1]})
            else:
                rows.append({"forms": parts, "gloss": None})
            continue

        for sep in (" — ", " – ", " - "):
            if sep in line:
                left, right = line.split(sep, 1)
                left, right = left.strip(), right.strip()
                if _is_likely_gloss(right):
                    rows.append({"forms": [left], "gloss": right})
                elif _is_likely_gloss(left):
                    rows.append({"forms": [right], "gloss": left})
                break
    return rows


def extract_context_examples(context: str) -> List[str]:
    """Extract foreign-language example strings from a Linguini context block."""
    examples: List[str] = []
    seen = set()

    def add(token: str):
        token = _clean_token(token)
        if not token or len(token) < 1:
            return
        if _looks_english_phrase(token):
            return
        if re.fullmatch(r"[\W_]+", token):
            return
        key = token.lower()
        if key in seen:
            return
        seen.add(key)
        examples.append(token)

    for row in parse_lexicon_rows(context):
        for cell in row["forms"]:
            add(cell)
    return examples


def infer_num_answers(question: str, default: int = 1) -> int:
    """Infer how many answer lines the question expects."""
    q = question or ""
    match = re.search(r"\(1\s*[-–—]\s*(\d+)\)", q)
    if match:
        return max(1, int(match.group(1)))

    numbered = re.findall(r"(?m)^\s*(\d+)\s*[\.\)\:]\s+\S", q)
    if numbered:
        return len(numbered)

    blanks = [int(x) for x in re.findall(r"\((\d+)\)", q)]
    if blanks:
        return max(blanks)

    # Bare item lists (e.g. num_to_text digit lines, text_to_num forms)
    items = extract_query_items(q)
    if len(items) >= 1:
        return len(items)

    return default


def extract_query_items(question: str) -> List[str]:
    """Extract numbered query items (translation / numeral prompts)."""
    items = [m.group(1).strip() for m in _NUMBERED_ITEM_RE.finditer(question or "")]
    if items:
        return items
    lines = []
    for line in (question or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(
            r"(?i)^(translate|fill|write|determine|here|context|question)\b",
            line,
        ):
            continue
        if _BLANK_RE.match(line):
            continue
        lines.append(line)
    return lines


def _question_direction(question: str) -> str:
    """Return answer direction: english | lrl | digits | text | blanks | other."""
    q = (question or "").lower()
    if "|" in (question or "") and (
        "fill the blank" in q
        or "fill in the blank" in q
        or re.search(r"\(\d+\)", question or "")
    ):
        return "blanks"
    if re.search(r"translate into english\b", q):
        return "english"
    if re.search(r"write in digits\b", q) or re.search(r"write in numerals\b", q):
        return "digits"
    if re.search(r"write out in\b", q) or re.search(
        r"write in (?:the )?[a-zá-ÿ]", q
    ):
        return "text"
    if re.search(r"translate into\b", q):
        return "lrl"
    if re.search(r"determine the correct correspondence", q):
        return "match"
    return "other"


def _fill_blank_nn(
    rows: List[Dict],
    question: str,
    n: int,
    lrl_forms: Sequence[str],
) -> List[str]:
    """Fill blank cells by projecting the nearest complete context row."""
    by_index: Dict[int, str] = {}

    for raw_line in (question or "").splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        blank_idxs = []
        known: Dict[int, str] = {}
        for i, part in enumerate(parts):
            m = _BLANK_RE.match(part)
            if m:
                blank_idxs.append((i, int(m.group(1))))
            elif part:
                known[i] = part
        if not blank_idxs:
            continue

        best_full: Optional[List[str]] = None
        best_score = -1.0
        for row in rows:
            full = list(row["forms"])
            if row["gloss"] is not None:
                full.append(row["gloss"])
            score = 0.0
            compared = 0
            for i, val in known.items():
                if i < len(full):
                    score += similarity(val, full[i])
                    compared += 1
                else:
                    score += max(similarity(val, cell) for cell in full)
                    compared += 1
            if compared:
                score /= compared
            if score > best_score:
                best_score = score
                best_full = full

        for col_i, blank_num in blank_idxs:
            if best_full is not None and col_i < len(best_full):
                by_index[blank_num] = best_full[col_i]
            elif known:
                cue = next(iter(known.values()))
                by_index[blank_num] = nearest_neighbor(cue, lrl_forms) or "?"
            else:
                by_index[blank_num] = lrl_forms[0] if lrl_forms else "?"

    if by_index:
        return [by_index[i] for i in sorted(by_index)][:n]
    return list(lrl_forms[:n]) + (
        [lrl_forms[-1]] * max(0, n - len(lrl_forms)) if lrl_forms else ["?"] * n
    )


def lexicon_nn_answers(
    context: str, question: str, examples: Sequence[str]
) -> List[str]:
    """Lexicon align + nearest-neighbor copy baseline.

    Builds an LRL↔gloss lexicon from the context table, then for each query
    item copies the aligned neighbor (exact match preferred, else highest
    character similarity).
    """
    rows = parse_lexicon_rows(context)
    n = infer_num_answers(question, default=1)
    direction = _question_direction(question)

    lrl_forms: List[str] = []
    glosses: List[str] = []
    for row in rows:
        lrl_forms.extend(row["forms"])
        if row["gloss"] is not None:
            glosses.append(row["gloss"])
    if not lrl_forms:
        lrl_forms = list(examples)

    if direction == "blanks":
        return _fill_blank_nn(rows, question, n, lrl_forms)

    items = extract_query_items(question)
    if not items:
        return list(lrl_forms[:n]) + (
            [lrl_forms[-1]] * max(0, n - len(lrl_forms)) if lrl_forms else ["?"] * n
        )

    answers: List[str] = []
    for item in items[:n]:
        if direction == "english":
            best_row, best_score = None, -1.0
            for row in rows:
                if row["gloss"] is None:
                    continue
                score = max(similarity(item, form) for form in row["forms"])
                if score > best_score:
                    best_row, best_score = row, score
            if best_row is not None:
                answers.append(best_row["gloss"])
            else:
                answers.append(
                    nearest_neighbor(item, glosses)
                    or nearest_neighbor(item, lrl_forms)
                    or "?"
                )
        elif direction == "lrl":
            best_row, best_score = None, -1.0
            for row in rows:
                if row["gloss"] is None:
                    continue
                score = similarity(item, row["gloss"])
                if score > best_score:
                    best_row, best_score = row, score
            if best_row is not None:
                answers.append(" ".join(best_row["forms"]))
            else:
                answers.append(nearest_neighbor(item, lrl_forms) or "?")
        elif direction == "digits":
            best_row, best_score = None, -1.0
            for row in rows:
                if row["gloss"] is None or not re.fullmatch(
                    r"\d+", row["gloss"].strip()
                ):
                    continue
                score = max(similarity(item, form) for form in row["forms"])
                if score > best_score:
                    best_row, best_score = row, score
            if best_row is not None:
                answers.append(best_row["gloss"])
            else:
                digit_glosses = [
                    g for g in glosses if re.fullmatch(r"\d+", g.strip())
                ]
                answers.append(
                    nearest_neighbor(item, digit_glosses)
                    or (digit_glosses[0] if digit_glosses else "?")
                )
        elif direction == "text":
            best_row, best_score = None, -1.0
            for row in rows:
                if row["gloss"] is None or not re.fullmatch(
                    r"\d+", row["gloss"].strip()
                ):
                    continue
                score = similarity(item, row["gloss"])
                if score > best_score:
                    best_row, best_score = row, score
            if best_row is not None:
                answers.append(" ".join(best_row["forms"]))
            else:
                answers.append(nearest_neighbor(item, lrl_forms) or "?")
        else:
            if _is_likely_gloss(item) and glosses:
                best_row, best_score = None, -1.0
                for row in rows:
                    if row["gloss"] is None:
                        continue
                    score = similarity(item, row["gloss"])
                    if score > best_score:
                        best_row, best_score = row, score
                answers.append(
                    " ".join(best_row["forms"])
                    if best_row
                    else (nearest_neighbor(item, lrl_forms) or "?")
                )
            else:
                answers.append(nearest_neighbor(item, lrl_forms) or "?")

    while len(answers) < n:
        answers.append(answers[-1] if answers else "?")
    return answers[:n]


def copy_context_answers(
    examples: Sequence[str],
    n: int,
    rng: random.Random,
) -> List[str]:
    """Baseline 1: copy context example words as answers."""
    if n <= 0:
        return []
    if not examples:
        return ["?"] * n
    pool = list(examples)
    rng.shuffle(pool)
    answers = []
    i = 0
    while len(answers) < n:
        answers.append(pool[i % len(pool)])
        i += 1
    return answers


def _char_ngrams(word: str, n: int) -> List[str]:
    """Return overlapping character n-grams from ``word``."""
    if not word:
        return []
    if len(word) < n:
        return [word]
    return [word[i : i + n] for i in range(len(word) - n + 1)]


def recombine_char_ngram_answers(
    examples: Sequence[str],
    n: int,
    rng: random.Random,
    ngram_size: int = 3,
) -> List[str]:
    """Baseline 2: randomly recombine character n-grams from context words."""
    if n <= 0:
        return []
    if not examples:
        return ["?"] * n

    grams: List[str] = []
    for word in examples:
        grams.extend(_char_ngrams(word, ngram_size))
    grams = [g for g in grams if g]
    if not grams:
        return copy_context_answers(examples, n, rng)

    lengths = [len(w) for w in examples if w]
    answers = []
    for _ in range(n):
        target_len = rng.choice(lengths) if lengths else max(ngram_size, 1)
        target_len = max(target_len, 1)
        pieces = []
        size = 0
        for _attempt in range(64):
            piece = rng.choice(grams)
            pieces.append(piece)
            size += len(piece)
            if size >= target_len:
                break
        answers.append("".join(pieces)[:target_len])
    return answers


def _seeded_rng(seed: int, prompt: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _format_output(answer_lines: Sequence[str], prompt: str) -> str:
    body = "\n".join(answer_lines)
    if re.search(r"(?i)use this format exactly", prompt or "") or (
        re.search(r"(?i)\banswer\s*:", prompt or "")
        and re.search(r"(?i)\bexplanation\s*:", prompt or "")
    ):
        return (
            f"Answer:\n{body}\n\n"
            "Explanation:\nHeuristic baseline answer composed from context examples."
        )
    return body


class HeuristicProvider(BaseProvider):
    """Non-LLM baselines for Linguini-style puzzles.

    Supported ``model_id`` values:
      - ``copy-context``: copy context example words as answers
      - ``recombine-char-ngrams``: randomly recombine character n-grams
      - ``lexicon-nn``: lexicon align + nearest-neighbor copy from context tables
    """

    async def generate(
        self,
        model_id,
        prompt,
        params,
        system_prompt=None,
        reasoning_effort=None,
        thinking_budget=0,
    ):
        context, question = _parse_prompt(prompt)
        examples = extract_context_examples(context)
        if not examples:
            examples = extract_context_examples(question)

        n = infer_num_answers(question, default=1)
        seed = int(params.get("seed", 0) or 0)
        rng = _seeded_rng(seed, prompt or "")
        ngram_size = int(params.get("ngram_size", 3) or 3)

        mode = (model_id or "copy-context").strip().lower()
        if mode in {
            "recombine-char-ngrams",
            "recombine-ngrams",
            "char-ngrams",
            "ngram",
            "recombine",
        }:
            answers = recombine_char_ngram_answers(
                examples, n, rng, ngram_size=ngram_size
            )
        elif mode in {
            "lexicon-nn",
            "lexicon-align-nn",
            "lexicon-align",
            "nearest-neighbor",
            "nn",
        }:
            answers = lexicon_nn_answers(context or question, question, examples)
        else:
            answers = copy_context_answers(examples, n, rng)

        generation = _format_output(answers, prompt or "")
        return {
            "raw_generation": generation,
            "generation": generation,
            "reasoning": None,
            "finish_reason": "stop",
        }
