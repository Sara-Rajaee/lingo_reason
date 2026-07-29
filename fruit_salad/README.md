# fruit_salad — IOL-2026 open-model ensemble

Oracle / best-of-n single solution over 4 self-hosted open models:
gemma4-31b-it, qwen3.6-27b, deepseek-r1-32b, glm-4.7-flash

- **Oracle score** (points-weighted geomean of EM.chrF): **0.2906** (EM 0.2001, chrF 0.4221) over 14 problems.
- **Per-model contributions** (problems where each was the best pick): {'gemma4-31b-it': 8, 'qwen3.6-27b': 4, 'deepseek-r1-32b': 2, 'glm-4.7-flash': 0}

`submission.json` is the best-of-n single solution (one answer per problem).
