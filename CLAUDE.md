# Lingo Reason

Multilingual LLM evaluation framework.

## Environment Setup

- Local venv: `.venv` (managed by `uv`)
- Install deps: `uv add <package>`
- Run scripts: `uv run python run.py ...`

## Running Evaluations

```bash
# Disable proxy and use cached datasets
GPT_OSS_API_BASE="http://<node>:<port>/v1/" \
HF_HUB_OFFLINE=1 \
http_proxy="" HTTP_PROXY="" https_proxy="" HTTPS_PROXY="" \
uv run python run.py --model <model> --task <task>
```

- Get the server node from: `squeue -j <jobid> --format="%.10i %.8T %R %k"`
- gpt-oss-20b default port: 19743
- gpt-oss-120b default port: 19744

## Serving GPT-OSS Models (SLURM)

```bash
PORT=19743 sbatch scripts/serve_slurm.sh openai/gpt-oss-20b
PORT=19744 sbatch scripts/serve_slurm.sh openai/gpt-oss-120b
```

Required env vars in the script:
- `HF_HUB_OFFLINE=1` — use cached model weights
- `TIKTOKEN_ENCODINGS_BASE=$HOME/.cache/tiktoken-rs-cache/` — local tiktoken vocab for openai_harmony

## Caching Datasets

PyPI and HuggingFace are blocked behind a proxy on this cluster. Cache datasets from a machine with internet access using:

```bash
uv run hf download --repo-type dataset <dataset_name>
```

Datasets to cache:
```bash
uv run hf download --repo-type dataset juletxara/mgsm
uv run hf download --repo-type dataset xcopa
uv run hf download --repo-type dataset juletxara/xstory_cloze
uv run hf download --repo-type dataset americas_nli
uv run hf download --repo-type dataset Davlan/sib200
uv run hf download --repo-type dataset mkqa
uv run hf download --repo-type dataset Maxwell-Jia/AIME_2024
uv run hf download --repo-type dataset livecodebench/code_generation_lite
uv run hf download --repo-type dataset EleutherAI/frontiermath
uv run hf download --repo-type dataset Muennighoff/flores200
uv run hf download --repo-type dataset xnli
uv run hf download --repo-type dataset facebook/belebele
uv run hf download --repo-type dataset GAIR/OlympiadBench
uv run hf download --repo-type dataset Idavidrein/gpqa
```

Already cached: `CohereLabs/Global-MMLU-Lite`, `Qwen/PolyMath`, `google/wmt24pp`
