#!/bin/bash
# Fan-out driver: for each linguini vLLM serve job, wait for the node + health,
# then launch its linguini eval (lingo_reason's own run.py). Self-contained.
cd /storage/home/eduardosanchez/workspace/lingo_reason
set -u

# "jobid port served_model_name" (served name == models.yaml model_id), one per
# line in /tmp/ling_jobs.txt (written by the submit step).
mapfile -t ENTRIES < "${LING_JOBS_FILE:-/tmp/ling_jobs.txt}"

wait_and_eval() {
  local jid=$1 port=$2 name=$3
  local node="" waited=0 st
  # 1) wait for the job to be RUNNING and get its node
  while [ -z "$node" ] && [ $waited -lt 2400 ]; do
    st=$(squeue -j "$jid" -h -o "%T" 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      node=$(squeue -j "$jid" -h -o "%N" 2>/dev/null)
    elif [ -z "$st" ]; then
      echo "[$name] job $jid ended before running"; return 1
    fi
    [ -z "$node" ] && { sleep 20; waited=$((waited+20)); }
  done
  [ -z "$node" ] && { echo "[$name] timed out waiting for node"; return 1; }
  echo "[$name] node=$node port=$port; waiting for health (torch.compile ~7min)..."
  # 2) poll the OpenAI /v1/models endpoint until the model is served
  waited=0
  while [ $waited -lt 2400 ]; do
    if curl -s --connect-timeout 3 "http://$node:$port/v1/models" 2>/dev/null | grep -q "$name"; then
      echo "[$name] HEALTHY at http://$node:$port/v1/ — launching eval"
      GPT_OSS_API_BASE="http://$node:$port/v1/" HF_HUB_OFFLINE=1 \
        http_proxy="" HTTP_PROXY="" https_proxy="" HTTPS_PROXY="" \
        nohup uv run python run.py --model "$name" --task linguini ${CONCURRENCY:+--concurrency $CONCURRENCY} \
        > "logs/eval_${name}_linguini.out" 2>&1 &
      echo "[$name] eval launched pid $!"
      return 0
    fi
    if [ -z "$(squeue -j "$jid" -h -o %T 2>/dev/null)" ]; then
      echo "[$name] job died during model load (check logs/lingserve_${jid}.err)"; return 1
    fi
    sleep 20; waited=$((waited+20))
  done
  echo "[$name] health timeout"; return 1
}

for e in "${ENTRIES[@]}"; do
  read -r jid port name <<< "$e"
  wait_and_eval "$jid" "$port" "$name" &
done
wait
echo "=== fan-out driver done: all healthy servers have evals launched ==="
