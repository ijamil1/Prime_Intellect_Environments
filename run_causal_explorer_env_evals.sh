#!/bin/bash

models=(
  "allenai/olmo-3.1-32b-instruct"
  "meta-llama/llama-3.1-70b-instruct"
  "openai/gpt-5.1-chat"
)

for model in "${models[@]}"; do
  echo "Running eval with model: $model"
  prime eval run CausalExplorerEnv -n 50 -r 3 -a '{"num_objects_range":[4,10], "num_examples": 50}' -m "$model"
  echo ""
done
