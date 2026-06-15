#!/bin/bash
#SBATCH --job-name=lingo-vllm
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-task=4
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# Single-node vLLM server for lingo_reason evals
set -eo pipefail
mkdir -p logs

MODEL="${1:-openai/gpt-oss-120b}"
PORT="${PORT:-19743}"

echo "════════════════════════════════════════════════════════════"
echo "  lingo_reason vLLM Server"
echo "════════════════════════════════════════════════════════════"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Node:     $(hostname -f)"
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "════════════════════════════════════════════════════════════"

nvidia-smi

TP=$(nvidia-smi -L | wc -l)

# write endpoint to a file so eval jobs can find this server
ENDPOINT_FILE=~/lingo_reason/.server_endpoint
echo "$(hostname -f):${PORT}" > "$ENDPOINT_FILE"
echo "Endpoint written to: $ENDPOINT_FILE"
echo "Server will be available at: http://$(hostname -f):${PORT}/v1/"
echo ""
# activate conda env (run `which conda` once on login node to confirm path)
export PATH="/home/srajaee/.conda/envs/lingo-reason/bin:$PATH"
# source activate lingo-reason

# put HF cache on scratch — home is too small for 120b weights (~63 GB)
export HF_HOME="/scratch-shared/srajaee/HF/cache/hub/"

# leave HF_HUB_OFFLINE off for first run so weights download;
# after that you can re-enable it for faster startup:
# export HF_HUB_OFFLINE=1

# cd ~/lingo_reason

vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code