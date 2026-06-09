#!/bin/bash
#SBATCH --job-name=lingo-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --account=omnilingual
#SBATCH --qos=h100_omnilingual_high
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -eo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
MODE="${MODE:-lora}"
DISTILLED_PCT="${DISTILLED_PCT:-100}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
NO_REASONING_FLAG=""
if [ -n "$NO_REASONING" ] && [ "$NO_REASONING" != "0" ]; then
    NO_REASONING_FLAG="--no-reasoning"
fi

DISTILLED_PATHS_DEFAULT=(
    distilled_results/gpt-oss-120b/lingoly/default/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/en/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/de/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/es/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/fr/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/ja/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/ko/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/pt/all_gold_outputs.json
    distilled_results/gpt-oss-120b/mulr/zh/all_gold_outputs.json
)
if [ -n "$DISTILLED_PATHS_OVERRIDE" ]; then
    read -ra DISTILLED_PATHS <<< "$DISTILLED_PATHS_OVERRIDE"
else
    DISTILLED_PATHS=("${DISTILLED_PATHS_DEFAULT[@]}")
fi

MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
TS=$(date +%Y%m%d_%H%M%S)
OUT="train_runs/${MODEL_SLUG}_${MODE}_distilled-${DISTILLED_PCT}${NO_REASONING:+_noreason}_${TS}"

echo "════════════════════════════════════════════════════════════"
echo "  lingo_reason SFT Training"
echo "════════════════════════════════════════════════════════════"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Node:          $(hostname -f)"
echo "  Model:         $MODEL"
echo "  Mode:          $MODE"
echo "  Distilled %:   $DISTILLED_PCT"
echo "  Epochs:        $EPOCHS"
echo "  Output:        $OUT"
echo "════════════════════════════════════════════════════════════"
nvidia-smi

export HF_HUB_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="${HOME}/.cache/tiktoken-rs-cache"
export TIKTOKEN_ENCODINGS_BASE="${HOME}/.cache/tiktoken-rs-cache/"
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

uv run --no-sync torchrun \
    --standalone \
    --nproc_per_node=8 \
    scripts/train_sft.py \
    --model "$MODEL" \
    --mode "$MODE" \
    --distilled-paths "${DISTILLED_PATHS[@]}" \
    --distilled-pct "$DISTILLED_PCT" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-length "$MAX_SEQ_LEN" \
    $NO_REASONING_FLAG \
    --output-dir "$OUT"
