#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Launch a vLLM OpenAI-compatible API server for GPT-OSS models.
#
# Usage:
#   # Serve a model with defaults (auto tensor-parallel, port 19743)
#   bash scripts/serve.sh openai/gpt-oss-20b
#
#   # Override tensor parallelism and port
#   TP=4 PORT=8000 bash scripts/serve.sh openai/gpt-oss-120b
#
#   # Serve any HuggingFace model
#   bash scripts/serve.sh meta-llama/Llama-3.3-70B-Instruct
#
# Environment variables:
#   MODEL           Model name/path           (positional arg or env var)
#   PORT            API server port            (default: 19743)
#   TP              Tensor-parallel GPUs       (default: auto-detect)
#   DTYPE           Model dtype                (default: auto)
#   MAX_MODEL_LEN   Max sequence length        (default: vLLM auto)
#   GPU_UTIL        GPU memory utilization     (default: 0.9)
#   EXTRA_ARGS      Additional vLLM arguments  (default: "")
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${1:-${MODEL:-openai/gpt-oss-20b}}"
PORT="${PORT:-19743}"
DTYPE="${DTYPE:-auto}"
GPU_UTIL="${GPU_UTIL:-0.9}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Auto-detect tensor parallelism from available GPUs if not set
if [ -z "${TP:-}" ]; then
    if command -v nvidia-smi &> /dev/null; then
        TP=$(nvidia-smi -L 2>/dev/null | wc -l)
        [ "$TP" -eq 0 ] && TP=1
    else
        TP=1
    fi
fi

echo "════════════════════════════════════════════════════════════"
echo "  vLLM Server"
echo "════════════════════════════════════════════════════════════"
echo "  Model:              ${MODEL}"
echo "  Port:               ${PORT}"
echo "  Tensor Parallel:    ${TP}"
echo "  Dtype:              ${DTYPE}"
echo "  GPU Mem Utilization: ${GPU_UTIL}"
echo "  Extra args:         ${EXTRA_ARGS:-<none>}"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  API will be available at: http://localhost:${PORT}/v1/"
echo ""

CMD=(
    vllm serve "${MODEL}"
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --dtype "${DTYPE}"
    --gpu-memory-utilization "${GPU_UTIL}"
    --trust-remote-code
)

# Optional: max model length
if [ -n "${MAX_MODEL_LEN:-}" ]; then
    CMD+=(--max-model-len "${MAX_MODEL_LEN}")
fi

# Append any extra arguments
if [ -n "${EXTRA_ARGS}" ]; then
    read -ra EXTRA_ARRAY <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA_ARRAY[@]}")
fi

exec "${CMD[@]}"
