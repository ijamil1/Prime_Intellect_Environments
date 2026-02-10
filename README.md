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
