#!/usr/bin/env python3
"""Build a human-jury PDF packet from linguini-explain raw_outputs.json.

Pipeline: raw_outputs → Markdown → HTML (pandoc) → PDF (Chrome headless).
Falls back to writing Markdown/HTML only if Chrome/pandoc is unavailable.

Example:
  python scripts/linguini_explain_jury_pdf.py \\
    --raw-outputs results/linguini-explain/gpt5.6-sol-high_reasoning_high/default/raw_outputs.json \\
    --output results/linguini-explain/gpt5.6-sol-high_reasoning_high/default/jury_packet.pdf
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tasks import LinguiniBenchmark  # noqa: E402

_CONTEXT_RE = re.compile(
    r"(?is)Context:\s*(.*?)\n\s*Question:\s*(.*?)(?:\n\s*(?:Give your|Answer with|Use this)|\Z)"
)
_QUESTION_ONLY_RE = re.compile(
    r"(?is)Question:\s*(.*?)(?:\n\s*(?:Give your|Answer with|Use this)|\Z)"
)

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an IOL-style human jury PDF from linguini-explain outputs."
    )
    p.add_argument(
        "--raw-outputs",
        type=Path,
        required=True,
        help="Path to raw_outputs.json from a linguini-explain run",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: alongside raw_outputs as jury_packet.pdf)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only include the first N problems",
    )
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated problem IDs to include (optional filter)",
    )
    p.add_argument(
        "--model-label",
        type=str,
        default="System A",
        help="Blind placeholder name for the model in the title (default: System A)",
    )
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name for the title (default: inferred from path, e.g. linguini-explain)",
    )
    p.add_argument(
        "--show-gold",
        action="store_true",
        help="Include gold answers (off by default for blind jury review)",
    )
    p.add_argument(
        "--show-scores",
        action="store_true",
        help="Include automatic metric scores (off by default)",
    )
    p.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Also write .md and .html next to the PDF",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override document title (default: '<task> -- <model-label>')",
    )
    return p.parse_args()


def load_outputs(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list in {path}")
    return data


def infer_task_name(raw_path: Path) -> str:
    """Infer task folder name from results/.../<task>/<model>/... paths."""
    parts = raw_path.resolve().parts
    for name in (
        "iol-2026-explain",
        "iol-2026",
        "linguini-explain",
        "linguini-nocontext",
        "linguini",
    ):
        if name in parts:
            return name
    # Fallback: parent of model dir → .../<task>/<model>/default/raw_outputs.json
    try:
        return raw_path.parent.parent.parent.name
    except IndexError:
        return "linguini-explain"


def sort_key_id(item: Dict[str, Any]) -> Tuple:
    """Sort by problem id (numeric when possible)."""
    pid = str(item.get("id", ""))
    digits = re.sub(r"\D", "", pid)
    if digits:
        return (0, int(digits), pid)
    return (1, pid)


def parse_problem_subproblem(pid: str) -> Optional[Tuple[int, int]]:
    """Decode IOL-style ids like 12026050200 → (problem=5, subproblem=2).

    Layout (11 digits): ``1`` + year(4) + problem(2) + subproblem(2) + ``00``.
    Falls back to the last 6 digits as PP SS xx when length differs.
    """
    digits = re.sub(r"\D", "", str(pid or ""))
    if len(digits) < 6:
        return None
    try:
        problem = int(digits[-6:-4])
        subproblem = int(digits[-4:-2])
    except ValueError:
        return None
    if problem <= 0 or subproblem <= 0:
        return None
    return problem, subproblem


def problem_heading(pid: str, fallback_index: int) -> str:
    parsed = parse_problem_subproblem(pid)
    if parsed:
        problem, subproblem = parsed
        return f"# Problem {problem}, subproblem {subproblem}"
    return f"# Problem {fallback_index}"


def extract_context_question(item: Dict[str, Any]) -> Tuple[str, str]:
    prompt = item.get("prompt") or ""
    m = _CONTEXT_RE.search(prompt)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _QUESTION_ONLY_RE.search(prompt)
    if m:
        return "", m.group(1).strip()
    source = (item.get("source") or "").strip()
    return "", source


def get_answer(item: Dict[str, Any]) -> str:
    # Prefer re-parsing generation so improved extractors apply to old runs.
    gen = item.get("generation") or item.get("raw_generation") or ""
    if gen:
        answer, _ = LinguiniBenchmark._extract_explained_parts(gen)
        if answer.strip():
            return answer.strip()
    ans = item.get("extracted_answer")
    if ans is not None and str(ans).strip():
        return str(ans).strip()
    return str(gen).strip()


def get_explanation(item: Dict[str, Any]) -> str:
    gen = item.get("generation") or item.get("raw_generation") or ""
    if gen:
        _, explanation = LinguiniBenchmark._extract_explained_parts(gen)
        if explanation.strip():
            return explanation.strip()
    exp = item.get("explanation")
    if exp is not None and str(exp).strip():
        return str(exp).strip()
    return ""


def md_escape_fence(text: str) -> str:
    """Put arbitrary text in a fenced block; raise fence length if needed."""
    text = text or ""
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}\n{text}\n{fence}"


def build_markdown(
    items: List[Dict[str, Any]],
    *,
    title: str,
    show_gold: bool,
    show_scores: bool,
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")

    items = sorted(items, key=sort_key_id)

    for i, item in enumerate(items, start=1):
        pid = str(item.get("id", f"item-{i}"))
        context, question = extract_context_question(item)
        answer = get_answer(item)
        explanation = get_explanation(item)
        gold = (item.get("target_text") or "").strip()
        scores = item.get("scores") or {}

        # One problem per page (title stays on its own first page).
        lines.append('<div class="problem-break"></div>')
        lines.append("")
        lines.append(problem_heading(pid, i))
        lines.append("")
        lines.append(f"**Problem ID:** `{pid}`")
        lines.append("")

        lines.append("## Context")
        lines.append("")
        if context:
            lines.append(md_escape_fence(context))
        else:
            lines.append("*No context provided in this run.*")
        lines.append("")

        lines.append("## Question")
        lines.append("")
        lines.append(md_escape_fence(question or "(empty)"))
        lines.append("")

        lines.append("## System answer")
        lines.append("")
        lines.append(md_escape_fence(answer or "(empty)"))
        lines.append("")

        lines.append("## System explanation")
        lines.append("")
        if explanation:
            # Keep explanation as markdown (tables/schemata are intentional).
            lines.append(explanation)
        else:
            lines.append("*No explanation provided.*")
        lines.append("")

        if show_gold:
            lines.append("## Gold answer")
            lines.append("")
            lines.append(md_escape_fence(gold or "(empty)"))
            lines.append("")

        if show_scores:
            lines.append("## Automatic scores")
            lines.append("")
            lines.append(
                f"- accuracy: `{scores.get('accuracy')}`  \n"
                f"- chrF: `{scores.get('chrf')}`  \n"
                f"- line_correct/total: "
                f"`{scores.get('line_correct')}/{scores.get('line_total')}`"
            )
            lines.append("")

    return "\n".join(lines)


CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}
html { font-size: 11pt; }
body {
  font-family: "Arial Unicode MS", "Arial Unicode", Arial, "Noto Sans",
               "Helvetica Neue", Helvetica, sans-serif;
  line-height: 1.35;
  color: #111;
  max-width: 900px;
  margin: 0 auto;
}
h1 { font-size: 1.45rem; margin-top: 1.4em; }
h1:first-of-type { margin-top: 0; }
.problem-break {
  break-before: page;
  page-break-before: always;
}
h2 { font-size: 1.1rem; margin-top: 1.1em; border-bottom: 1px solid #ccc; padding-bottom: 0.15em; }
pre, code {
  font-family: "Arial Unicode MS", "Arial Unicode", Menlo, Consolas, monospace;
  font-size: 0.92em;
  white-space: pre-wrap;
  word-break: break-word;
}
pre {
  background: #f6f6f6;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 0.7em 0.8em;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.6em 0 1em;
  font-size: 0.95em;
}
th, td {
  border: 1px solid #bbb;
  padding: 0.35em 0.5em;
  vertical-align: top;
}
th { background: #f0f0f0; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
"""


def find_chrome() -> Optional[str]:
    for cand in CHROME_CANDIDATES:
        if "/" in cand or cand.startswith("."):
            if Path(cand).exists():
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def markdown_to_html(md_text: str, title: str) -> str:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        # Minimal fallback: wrap markdown in <pre> (still printable).
        escaped = html.escape(md_text)
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>"
            f"<pre>{escaped}</pre></body></html>"
        )

    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / "packet.md"
        html_path = Path(tmp) / "packet.html"
        css_path = Path(tmp) / "style.css"
        md_path.write_text(md_text, encoding="utf-8")
        css_path.write_text(CSS, encoding="utf-8")
        cmd = [
            pandoc,
            str(md_path),
            "-f",
            "markdown+pipe_tables+fenced_code_blocks+raw_html",
            "-t",
            "html5",
            "-s",
            "--metadata",
            f"title={title}",
            "-c",
            css_path.name,
            "-o",
            str(html_path),
        ]
        subprocess.run(cmd, check=True, cwd=tmp)
        # Inline CSS so Chrome print does not depend on relative CSS path.
        body = html_path.read_text(encoding="utf-8")
        body = body.replace(
            f'<link rel="stylesheet" href="{css_path.name}" />',
            f"<style>{CSS}</style>",
        )
        body = body.replace(
            f'<link rel="stylesheet" href="{css_path.name}">',
            f"<style>{CSS}</style>",
        )
        return body


def html_to_pdf(html_text: str, pdf_path: Path) -> None:
    chrome = find_chrome()
    if not chrome:
        raise RuntimeError(
            "No Chrome/Chromium found for PDF rendering. "
            "HTML/Markdown intermediates were still written; print to PDF from a browser."
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        html_file = Path(tmp) / "packet.html"
        html_file.write_text(html_text, encoding="utf-8")
        # Chrome requires absolute file:// URL
        url = html_file.resolve().as_uri()
        out_pdf = Path(tmp) / "out.pdf"
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_pdf}",
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_pdf.exists():
            raise RuntimeError(
                "Chrome PDF export failed.\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        pdf_path.write_bytes(out_pdf.read_bytes())


def main() -> None:
    args = parse_args()
    raw_path = args.raw_outputs.expanduser().resolve()
    if not raw_path.exists():
        raise SystemExit(f"File not found: {raw_path}")

    items = load_outputs(raw_path)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [it for it in items if str(it.get("id", "")) in wanted]
    if args.limit is not None:
        items = items[: max(0, args.limit)]
    if not items:
        raise SystemExit("No problems to include after filtering.")

    task = args.task or infer_task_name(raw_path)
    model_label = args.model_label
    title = args.title or f"{task} -- {model_label}"
    out_pdf = (
        args.output.expanduser().resolve()
        if args.output
        else raw_path.parent / "jury_packet.pdf"
    )

    md = build_markdown(
        items,
        title=title,
        show_gold=args.show_gold,
        show_scores=args.show_scores,
    )
    html_doc = markdown_to_html(md, title)

    if args.keep_intermediates:
        md_out = out_pdf.with_suffix(".md")
        html_out = out_pdf.with_suffix(".html")
        md_out.write_text(md, encoding="utf-8")
        html_out.write_text(html_doc, encoding="utf-8")
        print(f"Wrote {md_out}")
        print(f"Wrote {html_out}")

    try:
        html_to_pdf(html_doc, out_pdf)
        print(f"Wrote {out_pdf}")
    except RuntimeError as exc:
        # Always leave readable intermediates if PDF fails.
        md_out = out_pdf.with_suffix(".md")
        html_out = out_pdf.with_suffix(".html")
        md_out.write_text(md, encoding="utf-8")
        html_out.write_text(html_doc, encoding="utf-8")
        print(f"Wrote {md_out}")
        print(f"Wrote {html_out}")
        print(f"PDF not created: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
