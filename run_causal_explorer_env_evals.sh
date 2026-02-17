#!/bin/bash

models=(
  "allenai/olmo-3.1-32b-instruct"
  "meta-llama/llama-3.1-70b-instruct"
  "openai/gpt-5.1-chat"
  "allenai/olmo-3.1-32b-thinking"
  "Qwen/Qwen3-4B-Thinking-2507"
)

EVALS_DIR="./environments/CausalExplorerEnv/outputs/evals"

for model in "${models[@]}"; do
  short_name="${model#*/}"
  if ls "$EVALS_DIR" 2>/dev/null | grep -q "$short_name"; then
    echo "Skipping $model (already has eval results for $short_name)"
  else
    echo "Running eval with model: $model"
    prime eval run CausalExplorerEnv -n 50 -r 3 -a '{"num_objects_range":[4,10], "num_examples": 50}' -m "$model"
  fi
  echo ""
done
