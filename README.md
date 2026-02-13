# Prime Intellect Environments

Creating RL environments for LLM training and evaluation, built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework. Workspace scaffolded via `prime lab setup`.

## Setup

```bash
uv sync
```

## Usage

```bash
# Create a new environment
prime env init <env-name>

# Install locally
prime env install <env-name>

# Run evaluation
prime eval run <env-name>

# Push to Prime Hub
prime env push -p ./environments/<env_name>
```

## Environments

| Environment | Description |
| ----------- | ----------- |
| [num-seq-env](environments/num_seq_env/) | Inductive reasoning over numeric sequences governed by order-2 linear recurrence relations |
| [CausalExplorerEnv](environments/CausalExplorerEnv/) |  Multi-turn causal reasoning environment based on the Blicket detector paradigm from developmental psychology. Tests an LLM's ability to design experiments, reason causally, and identify hidden causal structure. Inspired by  [Do LLMs Think Like Scientists? Causal Reasoning and Hypothesis Testing in LLMs](https://arxiv.org/pdf/2505.09614) |
