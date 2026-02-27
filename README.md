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
```

To see all available models and tasks:

```bash
python run.py --list
```

## Configuration

Task configurations (e.g. languages, number of evaluation samples) can be modified in the config files located in the `config/` directory.
