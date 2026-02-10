# num-seq-env

### Overview
- **Environment ID**: `num-seq-env`
- **Short description**: Inductive reasoning over numeric sequences governed by order-2 linear recurrence relations.
- **Tags**: single-turn, math, reasoning, eval

### Datasets
- **Primary dataset(s)**: Programmatically generated. Each example contains 5 consecutive terms from a sequence defined by `a(k) = c1*a(k-1) + c2*a(k-2)` with randomized coefficients and initial values.
- **Source links**: N/A (generated at load time)
- **Split sizes**: 500 examples by default (configurable via `num_examples` argument)

### Task
- **Type**: single-turn
- **Parser**: `XMLParser` with `<reasoning>` and `<answer>` fields
- **Rubric overview**: Single `exact_match` reward function — 1.0 if the parsed `<answer>` matches the ground truth integer, 0.0 otherwise.

The model sees 5 consecutive terms and is asked to predict the n-th next term (n in {1, 2, 3}). It must identify the underlying recurrence relation from the given terms and apply it to compute the answer.

### Quickstart

```bash
prime eval run num-seq-env
```

Configure model and sampling:

```bash
prime eval run num-seq-env -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Pass environment-specific args:

```bash
prime eval run num-seq-env -a '{"num_examples": 100, "seed": 123}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `500` | Number of dataset examples to generate |
| `seed` | int | `42` | Random seed for reproducible dataset generation |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `exact_match` | 1.0 if parsed answer matches ground truth, 0.0 otherwise |
