#!/bin/bash

models=(
  "allenai/olmo-3.1-32b-instruct"
  "allenai/olmo-3-7b-instruct"
)

EVALS_DIR="./environments/CausalExplorerEnv/outputs/evals"

for model in "${models[@]}"; do
  short_name="${model#*/}"
  if ls "$EVALS_DIR" 2>/dev/null | grep -q "$short_name"; then
    echo "Skipping $model (already has eval results for $short_name)"
  else
    echo "Running eval with model: $model"
    prime eval run irfanjamil/CausalExplorerEnv -n 100 -r 3 -m "$model"
  fi
  echo ""
done
