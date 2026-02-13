#!/bin/bash

models=(
  "allenai/olmo-3.1-32b-think"
  "qwen/qwen3-vl-30b-a3b-thinking"
  "prime-intellect/intellect-3"
  "openai/gpt-4.1-mini"
  "openai/gpt-4.1"
)

for model in "${models[@]}"; do
  echo "Running eval with model: $model"
  prime eval run num-seq-env -c 10 -n 100 -m "$model"
  echo ""
done