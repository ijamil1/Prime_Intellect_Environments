# num-seq-env

### Overview
- **Environment ID**: `num-seq-env`
- **Short description**: Inductive reasoning over numeric sequences governed by variable-order linear recurrence relations.
- **Tags**: single-turn, math, reasoning, eval

### Datasets
- **Primary dataset(s)**: Programmatically generated. Each example contains consecutive terms from a sequence defined by an order-k linear recurrence `a(n) = c1*a(n-1) + c2*a(n-2) + ... + ck*a(n-k)`, where k is sampled uniformly from {2, 3, 4, 5} with randomized coefficients and initial values.
- **Source links**: N/A (generated at load time)
- **Split sizes**: 500 examples by default (configurable via `num_examples` argument)

### Task
- **Type**: single-turn
- **Parser**: `XMLParser` with `<reasoning>` and `<answer>` fields
- **Rubric overview**: Single `exact_match` reward function — 1.0 if the parsed `<answer>` matches the ground truth integer, 0.0 otherwise.

The model sees consecutive terms from a known position in the sequence (e.g., "terms 10 through 20") and is asked to compute a specific term by its absolute position. The target term may be **before or after** the shown window. A successful model will likely first identify the underlying recurrence relation — including its order — from the given terms, and then use that relation to compute the requested term.

**Shown terms and identifiability.** `max_k` (default 5) is the maximum recurrence order. By default, `2*max_k + 1` terms are shown (11 for max_k=5). An order-k recurrence has k unknown coefficients; fitting it to L consecutive terms yields L-k equations in k unknowns. Showing at least 2k terms guarantees the system is (over-)determined so the coefficients — and therefore every future and past term — are uniquely recoverable. Showing `2*max_k + 1` terms ensures this holds for all k up to max_k.

**Generation-time checks.** Each generated sequence is validated before inclusion:
- **Genuine order check**: the k x k Hankel determinant of the shown terms is verified to be non-zero, confirming the sequence is truly order k which implies it is not expressible by a shorter recurrence and the coefficients are uniquely determinable.
- **Periodicity handling**: if the characteristic polynomial has roots on the unit circle (roots of unity by Kronecker's theorem), the sequence is periodic. In this case, the shown window is truncated to fewer than one full period so the model cannot exploit repeating patterns. If the period is too short to show enough terms for identifiability (`period - 1 < 2k + 1`), the sequence is rejected.

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
prime eval run num-seq-env -a '{"num_examples": 100, "seed": 123, "min_k": 2, "max_k": 5}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `500` | Number of dataset examples to generate |
| `seed` | int | `42` | Random seed for reproducible dataset generation |
| `min_k` | int | `2` | Minimum recurrence order |
| `max_k` | int | `5` | Maximum recurrence order |

### Baseline Results

| Model | Accuracy | Details |
| ----- | -------- | ------- |
| `gpt-4.1-mini` (default) | 69.2% | `prime eval run num-seq-env` with no CLI overrides; num_examples (250) and rollouts_per_example (1) determined by the environment's `pyproject.toml` |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `exact_match` | 1.0 if parsed answer matches ground truth, 0.0 otherwise |
