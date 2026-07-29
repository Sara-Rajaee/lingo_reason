# fruit_salad — IOL-2026 open-model oracle / best-of-n

Oracle / best-of-n single solution over 4 self-hosted open models
(gemma4-31b-it, qwen3.6-27b, deepseek-r1-32b, glm-4.7-flash). For each problem
the answer is taken from the model that scored best vs gold (EM then chrF).

## Files
- **`submission.json`** — one entry per problem: `id`, `answer` (final best-of-n
  answer), `source_model` (winning model), `explanation` (that model's reasoning,
  verbatim), `has_explanation`.
- **`oracle_report.json`** — oracle score: 0.3353 (EM 0.2318, chrF 0.485) over 14
  problems; per-model wins gemma 6 / qwen 5 / deepseek 2 / glm 1.

## Notes
- `has_explanation` is `false` for one problem (qwen `12026050100`), where the
  explanation pass degenerated; its raw reasoning is kept verbatim.
- glm `12026040100` also degenerated after a clean answer/explanation head; kept
  verbatim, which is why `submission.json` is a few MB.
