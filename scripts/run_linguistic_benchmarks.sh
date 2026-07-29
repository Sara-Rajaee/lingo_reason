#!/bin/bash
# Run linguistic reasoning benchmarks on gpt-oss models
# Usage: bash scripts/run_linguistic_benchmarks.sh <node_20b> <node_120b>
set -e

NODE_20B="${1:?Usage: $0 <node_20b> <node_120b>}"
NODE_120B="${2:?Usage: $0 <node_20b> <node_120b>}"

export HF_HUB_OFFLINE=1
export http_proxy="" HTTP_PROXY="" https_proxy="" HTTPS_PROXY=""

echo "========== LOBSTER on gpt-oss-20b =========="
GPT_OSS_API_BASE="http://${NODE_20B}.fair-sc-compute-compute.tenant-slurm.svc.cluster.local:19743/v1/" \
uv run python run.py --model gpt-oss-20b --task lobster

echo "========== LOBSTER on gpt-oss-120b =========="
GPT_OSS_API_BASE="http://${NODE_120B}.fair-sc-compute-compute.tenant-slurm.svc.cluster.local:19744/v1/" \
uv run python run.py --model gpt-oss-120b --task lobster

echo "========== LingGym on gpt-oss-20b =========="
GPT_OSS_API_BASE="http://${NODE_20B}.fair-sc-compute-compute.tenant-slurm.svc.cluster.local:19743/v1/" \
uv run python run.py --model gpt-oss-20b --task linggym

echo "========== LingGym on gpt-oss-120b =========="
GPT_OSS_API_BASE="http://${NODE_120B}.fair-sc-compute-compute.tenant-slurm.svc.cluster.local:19744/v1/" \
uv run python run.py --model gpt-oss-120b --task linggym

echo "========== LingOly on gpt-oss-20b (via lm-eval-harness) =========="
lm_eval run \
    --model local-completions \
    --model_args model=openai/gpt-oss-20b,base_url=http://${NODE_20B}.fair-sc-compute-compute.tenant-slurm.svc.cluster.local:19743/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
    --tasks lingoly \
    --batch_size 16 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --output_path ./results/lingoly/gpt-oss-20b/ \
    --log_samples

echo "========== LingOly on gpt-oss-120b (via lm-eval-harness) =========="
lm_eval run \
    --model local-completions \
    --model_args model=openai/gpt-oss-120b,base_url=http://${NODE_120B}.fair-sc-compute-compute.tenant-slurm.svc.cluster.local:19744/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
    --tasks lingoly \
    --batch_size 16 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --output_path ./results/lingoly/gpt-oss-120b/ \
    --log_samples

echo "All linguistic benchmarks complete!"
