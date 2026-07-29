#!/bin/bash
#SBATCH --job-name=ling-serve
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --account=omnilingual
#SBATCH --qos=h100_omnilingual_high
#SBATCH --output=logs/lingserve_%j.out
#SBATCH --error=logs/lingserve_%j.err

# Self-contained vLLM server for the linguini runs on newer-arch models
# (gemma4-31B, qwen3.6-27B, deepseek-r1-32B, glm-4.7-flash).
#
# lingo_reason's own serving venv ships vLLM 0.11.0, which predates these
# architectures. PyPI is blocked on this cluster, so we invoke a vLLM 0.23.0
# *binary* directly (VLLM_BIN). This runs only the inference server — no
# external project/orchestration code.
set -eo pipefail

MODEL="${1:?usage: MODEL as positional arg (path or HF repo id)}"
PORT="${PORT:-19760}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-72000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
VLLM_BIN="${VLLM_BIN:-/checkpoint/omnilingual/users/eduardosanchez/guac_eval_env/bin/vllm}"

echo "════════════════════════════════════════════════════════════"
echo "  lingo_reason linguini vLLM server"
echo "════════════════════════════════════════════════════════════"
echo "  Job ID:         $SLURM_JOB_ID"
echo "  Node:           $(hostname -f)"
echo "  vLLM binary:    $VLLM_BIN"
echo "  Model:          $MODEL"
echo "  Served name:    $SERVED_NAME"
echo "  Port:           $PORT"
echo "  TP:             $TP"
echo "  max-model-len:  $MAX_MODEL_LEN"
echo "  extra flags:    $EXTRA_FLAGS"
echo "════════════════════════════════════════════════════════════"
nvidia-smi

# Record the port in the job comment for discovery (squeue -o %k)
scontrol update JobId=$SLURM_JOB_ID Comment=$PORT || true

echo ""
echo "Server will be available at: http://$(hostname -f):${PORT}/v1/"
echo ""

export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- CUDA env for the borrowed vLLM 0.23.0 env ---------------------------------
# Invoking the env's vllm bare leaves torch unable to find the CUDA runtime
# ("No CUDA runtime is found" -> "Failed to infer device type"). Replicate the
# env setup the guac_eval_env expects: put its nvidia pip libs on
# LD_LIBRARY_PATH, add the CUDA driver dir, and LD_PRELOAD cublas.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export FLASHINFER_JIT_CACHE_DIR="/tmp/flashinfer_cache_$$"
_ENV_ROOT="$(dirname "$(dirname "$VLLM_BIN")")"
_NVIDIA_BASE="${_ENV_ROOT}/lib/python3.12/site-packages/nvidia"
if [ -d "${_NVIDIA_BASE}" ]; then
    for _nv_lib in "${_NVIDIA_BASE}"/*/lib; do
        [ -d "${_nv_lib}" ] && export LD_LIBRARY_PATH="${_nv_lib}:${LD_LIBRARY_PATH:-}"
    done
fi
for _d in /usr/lib64 /usr/lib/x86_64-linux-gnu; do
    if [ -f "${_d}/libcuda.so.1" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${_d}"
        break
    fi
done
_CUBLAS="${_NVIDIA_BASE}/cublas/lib/libcublas.so.12"
[ -f "${_CUBLAS}" ] && export LD_PRELOAD="${_CUBLAS}:${LD_PRELOAD:-}"

cd /storage/home/eduardosanchez/workspace/lingo_reason

"${VLLM_BIN}" serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype auto \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --trust-remote-code \
    ${EXTRA_FLAGS}
