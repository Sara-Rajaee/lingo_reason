import json
import os
import random

from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

OPENTHOUGHTS2_DATASET = "open-thoughts/OpenThoughts2-1M"
DEFAULT_OPENTHOUGHTS_MATH_CACHE = "data/openthoughts_math_subset.json" #save a cached file of math subset


def create_reasoning_dataloader(distilled_path, distilled_percentage=30, max_samples=-1,
    batch_size=8, num_workers=0, **dataset_kwargs):
    """
    Building dataloader
    """
    dataset = ReasoningDataset(
        distilled_path=distilled_path,
        distilled_percentage=distilled_percentage,
        max_samples=max_samples,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_generation_examples,
    )


class ReasoningDataset(IterableDataset):
    def __init__(
        self,
        distilled_path,
        distilled_percentage=30,
        max_samples=-1,
        openthoughts_dataset=OPENTHOUGHTS2_DATASET,
        openthoughts_split="train",
        openthoughts_subset_path=DEFAULT_OPENTHOUGHTS_MATH_CACHE,
        rebuild_openthoughts_cache=False,
        openthoughts_cache_max_samples=-1,
        seed=42,
    ):
        self.distilled_reasoning = load_distilled_reasoning(distilled_path)
        self.distilled_ratio = _normalize_percentage(distilled_percentage)
        self.max_samples = max_samples
        self.openthoughts_dataset = openthoughts_dataset
        self.openthoughts_split = openthoughts_split
        self.openthoughts_subset_path = openthoughts_subset_path
        self.rebuild_openthoughts_cache = rebuild_openthoughts_cache
        self.openthoughts_cache_max_samples = openthoughts_cache_max_samples
        self.seed = seed

    def __iter__(self):
        distilled_count, openthoughts_count = self._counts()
        rng = random.Random(self.seed)

        distilled_rows = list(self.distilled_reasoning)
        rng.shuffle(distilled_rows)
        distilled_iter = iter(distilled_rows[:distilled_count])

        openthoughts_iter = None
        if openthoughts_count > 0:
            openthoughts_iter = self._iter_openthoughts_math()

        schedule = self._build_source_schedule(
            distilled_count,
            openthoughts_count,
            rng,
        )

        for source_name in schedule:
            if source_name == "distilled":
                row = next(distilled_iter)
                yield format_distilled_row(row)
            else:
                try:
                    yield next(openthoughts_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "OpenThoughts cache does not contain enough usable math rows "
                        "for the requested mix."
                    ) from exc


    def _counts(self):
        distilled_total = len(self.distilled_reasoning)

        if self.max_samples == -1:
            distilled_count = distilled_total

            if self.distilled_ratio == 1:
                return distilled_count, 0

            if self.distilled_ratio == 0:
                raise ValueError("max_samples=-1 requires distilled_percentage > 0.")

            total = max(distilled_count, round(distilled_count / self.distilled_ratio))
            return distilled_count, max(0, total - distilled_count)

        if self.max_samples <= 0:
            raise ValueError("max_samples must be -1 or a positive integer.")

        if self.distilled_ratio == 1:
            return min(distilled_total, self.max_samples), 0

        if self.distilled_ratio == 0:
            return 0, self.max_samples

        requested_distilled = round(self.max_samples * self.distilled_ratio)
        distilled_count = min(distilled_total, requested_distilled)
        openthoughts_count = max(0, self.max_samples - distilled_count)
        return distilled_count, openthoughts_count


    def _iter_openthoughts_math(self):
        self._ensure_openthoughts_subset_cache()
        rows = load_json_rows(self.openthoughts_subset_path)
        rng = random.Random(self.seed)
        rng.shuffle(rows)
        for row in rows:
            if row.get("question") and row.get("reasoning"):
                yield row

    def _ensure_openthoughts_subset_cache(self):
        if (
            not self.rebuild_openthoughts_cache
            and os.path.exists(self.openthoughts_subset_path)
        ):
            return

        build_openthoughts_math_subset(
            output_path=self.openthoughts_subset_path,
            dataset_name=self.openthoughts_dataset,
            split=self.openthoughts_split,
            max_samples=self.openthoughts_cache_max_samples,
            seed=self.seed,
        )

    def _build_source_schedule(self, distilled_count, openthoughts_count, rng):
        schedule = (
            ["distilled"] * distilled_count
            + ["openthoughts"] * openthoughts_count
        )
        rng.shuffle(schedule)
        return schedule


def build_openthoughts_math_subset(
    output_path=DEFAULT_OPENTHOUGHTS_MATH_CACHE,
    dataset_name=OPENTHOUGHTS2_DATASET,
    split="train",
    max_samples=-1,
    seed=42,
):

    ds = load_dataset(dataset_name, split=split)
    ds = ds.filter(lambda row: is_math_source(row.get("source")))
    ds = ds.shuffle(seed=seed)

    rows = []
    for row in ds:
        example = format_openthoughts_row(row)
        if not example["question"] or not example["reasoning"]:
            continue

        rows.append(example)
        if max_samples != -1 and len(rows) >= max_samples:
            break

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return rows


def load_distilled_reasoning(path):
    rows = []
    for single_path in _normalize_paths(path):
        rows.extend(load_json_rows(single_path))
    return rows

def load_json_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("gold_outputs", "sampled_outputs", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Could not find a list of examples in {path}.")

def _normalize_paths(path):
    if isinstance(path, (str, bytes)):
        return [path]
    if isinstance(path, (list, tuple)):
        if not path:
            raise ValueError("distilled_path must contain at least one path.")
        return list(path)
    raise TypeError("distilled_path must be a path string or a list/tuple of path strings.")

def is_math_source(source):
    return isinstance(source, str) and "math" in source.lower()

def format_openthoughts_row(row):
    conversations = row.get("conversations") or []
    question = _find_message_text(conversations, {"user", "human"}) or row.get("question")
    assistant = _find_message_text(conversations, {"assistant", "gpt"})
    reasoning, final_answer = split_thinking_trace(assistant)

    return {
        "dataset": "openthoughts2",
        "id": row.get("id"),
        "source": row.get("source"),
        "question": str(question or "").strip(),
        "prompt": str(question or "").strip(),
        "reasoning": reasoning,
        "final_answer": final_answer,
        "raw_answer": str(assistant or "").strip(),
    }

def format_distilled_row(row):
    question = row.get("question") or row.get("source") or row.get("prompt")
    if row.get("context") and row.get("question"):
        question = (
            "Below is a problem sheet from a lingusitics exam. You will first see the entire sheet, then be asked\n"
            "to respond to specific questions from the sheet. Your answers to the questions should rely only on\n"
            "reasoning about the information provided in the sheet.\n"
            f"{row.get('preamble', '')}\n"
            f"{row.get('context', '')}\n"
            f"{row.get('all_questions', '')}\n"
            "Now respond to the following questions:\n"
            f"{row.get('question', '')}\n"
            "Answer with only the requested translation or phrase. "
            "Do not include explanations in your final answer."
        )
    final_answer = (
        row.get("answer")
        or row.get("target_text")
    )

    formatted = {
        "dataset": "distilled",
        "id": row.get("id"),
        "source": row.get("source"),
        "question": str(question or "").strip(),
        "prompt": str(row.get("prompt") or question or "").strip(),
        "reasoning": str(row.get("reasoning") or "").strip(),
        "final_answer": str(final_answer or "").strip(),
        "raw_answer": str(row.get("raw_generation") or final_answer or "").strip(),
    }

    for key in ("preamble", "context", "all_questions", "overall_question_n"):
        if key in row:
            formatted[key] = row.get(key)

    return formatted


def split_thinking_trace(text):
    text = str(text or "")
    start_token = "<think>"
    end_token = "</think>"

    start = text.find(start_token)
    end = text.find(end_token)

    if start == -1 or end == -1 or end < start:
        return None, None

    reasoning = text[start + len(start_token):end].strip()
    final_answer = text[end + len(end_token):].strip()
    return reasoning, final_answer


def collate_generation_examples(examples):
    keys = sorted({key for example in examples for key in example}) if examples else []
    return {key: [example.get(key) for example in examples] for key in keys}


def _find_message_text(conversations, roles):
    for message in conversations:
        if isinstance(message, dict) and message.get("from") in roles:
            return message.get("value") or message.get("content") or ""
    return ""


def _normalize_percentage(value):
    ratio = float(value)
    if ratio > 1:
        ratio = ratio / 100
    if not 0 <= ratio <= 1:
        raise ValueError("distilled_percentage must be between 0 and 100.")
    return ratio
