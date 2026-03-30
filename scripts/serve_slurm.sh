#!/bin/bash
#SBATCH --job-name=lingo-vllm
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --account=omnilingual
#SBATCH --qos=h100_omnilingual_high
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# Single-node vLLM server for lingo_reason evals
set -eo pipefail

MODEL="${1:-openai/gpt-oss-20b}"
PORT="${PORT:-19743}"

echo "════════════════════════════════════════════════════════════"
echo "  lingo_reason vLLM Server"
echo "════════════════════════════════════════════════════════════"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Node:     $(hostname -f)"
echo "  GPUs:     $SLURM_GPUS_ON_NODE"
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "════════════════════════════════════════════════════════════"

nvidia-smi

TP=$(nvidia-smi -L | wc -l)

# Store port in job comment for easy discovery
scontrol update JobId=$SLURM_JOB_ID Comment=$PORT

echo ""
echo "Server will be available at: http://$(hostname -f):${PORT}/v1/"
echo ""

export HF_HUB_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="${HOME}/.cache/tiktoken-rs-cache"
export TIKTOKEN_ENCODINGS_BASE="${HOME}/.cache/tiktoken-rs-cache/"

cd /storage/home/eduardosanchez/workspace/lingo_reason

/storage/home/eduardosanchez/workspace/omnilingual/.venv/bin/vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
