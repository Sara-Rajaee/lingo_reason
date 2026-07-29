# Linguistic Reasoning

A framework for evaluating language models on different tasks.

## Requirements

- Python 3.x
- API keys for the model providers you intend to use

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sara-Rajaee/lingo_reason
   cd your-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys as environment variables:
   ```bash
   export TOGETHER_API_KEY=your_key_here
   export GEMINI_API_KEY=your_key_here
   export OPENAI_API_KEY=your_key_here
   ```
   You only need to set the keys for the model providers you plan to use.

   For GPT-OSS models via a local vLLM proxy:
   ```bash
   export GPT_OSS_API_BASE=http://localhost:19743/v1/   # proxy URL (default)
   export GPT_OSS_API_KEY=no-key                        # optional, proxy usually has no auth
   ```
4. (optional) register polymath repo as a module (needed for polymath evaluation)
   ```bash
   git submodule add https://github.com/QwenLM/PolyMath.git third_party/PolyMath
   git submodule update --init --recursive
   ```
## Serving GPT-OSS Models

GPT-OSS models are served locally via [vLLM](https://github.com/vllm-project/vllm). A launch script is provided at `scripts/serve.sh`.

**Prerequisites:** Install vLLM (`pip install vllm`) and have GPU(s) available.

```bash
# Serve gpt-oss-20b (auto-detects available GPUs for tensor parallelism)
bash scripts/serve.sh openai/gpt-oss-20b

# Serve gpt-oss-120b on 4 GPUs, custom port
TP=4 PORT=8000 bash scripts/serve.sh openai/gpt-oss-120b

# Serve any HuggingFace model
bash scripts/serve.sh meta-llama/Llama-3.3-70B-Instruct
```

The server exposes an OpenAI-compatible API at `http://localhost:19743/v1/` (by default).
Once the server is running, use `run.py` in a separate terminal to evaluate against it.

**Environment variables for `scripts/serve.sh`:**

| Variable | Default | Description |
|-----------|---------|-------------|
| `PORT` | `19743` | API server port |
| `TP` | auto-detect | Tensor-parallel GPU count |
| `DTYPE` | `auto` | Model dtype (`auto`, `float16`, `bfloat16`) |
| `MAX_MODEL_LEN` | vLLM default | Max sequence length |
| `GPU_UTIL` | `0.9` | GPU memory utilization (0.0–1.0) |
| `EXTRA_ARGS` | | Additional `vllm serve` arguments |

## Usage

Run an evaluation by specifying a model and a task:

```bash
python run.py --model gemini-2.5-flash --task polymath

# GPT-OSS models (requires a running vLLM proxy)
python run.py --model gpt-oss-20b --task mmmlu
python run.py --model gpt-oss-120b --task polymath

# IOL 2026 contest problems (private HF datasets; needs HF_TOKEN with access)
python run.py --model gemini-2.5-flash --task iol-2026
python run.py --model gemini-2.5-flash --task iol-2026-explain
```

To see all available models and tasks:

```bash
python run.py --list
```

To collect high-temperature samples for distillation:

```bash
python run.py --model gpt-oss-120b --task mgsm --distillation --distillation-samples 32
```

Distillation runs write to `distilled_results/<model_name>/<task>/<subset>/` with `sampled_outputs.json` for every sampled output and `gold_outputs.json` for one gold-matching output per example. The gold file keeps the original task columns plus the selected reasoning trace.

For distillation sampling params, add `distillation_temperature` and `distillation_top_p` under the task's `defaults` in `config/tasks.yaml`. If they are not set, distillation falls back to the task's normal `temperature` and `top_p`.

## Data Generation

Use `create_reasoning_dataloader` to mix distilled reasoning data with math examples from OpenThoughts2-1M:

```python
from src.data_generation import create_reasoning_dataloader

dataloader = create_reasoning_dataloader(
    distilled_path=[
        "distilled_results/gpt-oss-120b/lingoly/default/all_gold_outputs.json",
        "distilled_results/gpt-oss-120b/iolbench/default/all_gold_outputs.json",
    ],
    distilled_percentage=30,
    max_samples=1000,
    batch_size=8,
    seed=42,
)

for batch in dataloader:
    print(batch["dataset"])
    print(batch["question"][0])
    print(batch["reasoning"][0])
    print(batch["final_answer"][0])
    break
```

`distilled_percentage=30` means 30% distilled examples and 70% OpenThoughts math examples. The first mixed run builds `data/openthoughts_math_subset.json`. `max_samples=-1` includes all rows from the distilled linguistic reasoning JSON file and loads enough OpenThoughts2 math rows to satisfy the requested mix. later runs reuse it.


## Configuration

Task configurations (e.g. languages, number of evaluation samples) can be modified in the config files located in the `config/` directory.
