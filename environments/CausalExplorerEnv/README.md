# CausalExplorerEnv

### Overview
- **Environment ID**: `CausalExplorerEnv`
- **Short description**: Multi-turn causal reasoning environment based on the Blicket detector paradigm from developmental psychology. Tests an LLM's ability to design experiments, reason causally, and identify which objects are Blickets.
- **Tags**: multi-turn, causal-reasoning, eval, train

### Reference
Based on [Do LLMs Think Like Scientists? Causal Reasoning and Hypothesis Testing in LLMs](https://arxiv.org/pdf/2505.09614).

### Task
- **Type**: multi-turn
- **Parser**: XMLParser (fields: `reasoning`, `action`)
- **Rubric overview**: Reward is split equally between Blicket identification accuracy (0.5) and hypothesis elimination efficiency (0.5).

The agent interacts with a simulated "Blicket-detecting machine" across two phases:
1. **Exploration phase** — toggle objects on/off the machine one at a time, observe whether the machine activates, and exit when ready.
2. **Answer phase** — declare which objects are Blickets. The agent has up to `MAX_ANSWER_ATTEMPTS` (3) retries to produce a correctly-formatted answer before the episode ends with no score.

The machine activates according to a hidden rule (disjunctive OR or conjunctive AND over Blicket objects). The rule type is hidden from the agent.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run CausalExplorerEnv
```

Configure model and sampling:

```bash
prime eval run CausalExplorerEnv \
  -m openai/gpt-4.1-mini \
  -n 50 -r 3 -t 4096 -T 0.7 \
  -a '{"num_objects_range": [4, 10], "num_examples": 50}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_objects_range` | tuple[int, int] | `(4, 10)` | Inclusive (min, max) range for number of objects per row. Must satisfy 4 <= lo <= hi <= 10 |
| `num_examples` | int | `100` | Number of dataset rows to generate |
| `seed` | int | `42` | RNG seed for reproducible dataset generation |

Per-row configuration is sampled dynamically at dataset generation time:
- **num_objects**: uniformly sampled from `num_objects_range`
- **num_blickets**: uniformly sampled from [2, floor(num_objects/2)]
- **rule_type**: randomly chosen as `"disjunctive"` or `"conjunctive"`
- **max_num_steps**: computed as `ceil(1.5 * optimal_steps)` where `optimal_steps` comes from a greedy info-gain simulation

### Architecture

`BlicketEnv` subclasses `vf.MultiTurnEnv`. The verifiers `max_turns` is set to `global_max_steps + 1 + MAX_ANSWER_ATTEMPTS` (the largest step budget across all rows + transition turn + up to 3 answer-phase retries).

**Rollout lifecycle:**

1. `setup_state()` reads per-row config from the dataset `info` field (blickets, rule type, step budget are pre-assigned at dataset generation). Initializes zeroed object states, the full hypothesis space (2^N blicket assignments × 2 rule types), and tracking counters.
2. `env_response()` drives the game loop across both phases. All turns increment `exploration_and_answer_count`:
   - **Exploration**: calls `parse_response(..., "exploration", ...)` which strips reasoning blocks, requires exactly one `<action>` tag, and delegates to `parse_action`. Validates the action, toggles object state, computes machine activation, filters the hypothesis space against the observation, and returns a compact observation. Invalid and redundant actions still consume a step.
   - **Answer**: calls `parse_response(..., "answer", ...)` which applies the same strict tag rules then delegates to `parse_predictions`. On successful parse, scores and terminates. On failure, sends a reformat message and loops up to `MAX_ANSWER_ATTEMPTS` (3) total attempts; if all exhausted, exits with score 0.
3. Termination is handled by the base class `has_final_env_response` stop condition.

**Strict action parsing (`parse_response`):**

All action extraction goes through `parse_response(content, phase, num_objects)` which enforces:
- All `<reasoning>...</reasoning>` blocks are stripped before searching for `<action>` tags.
- Exactly one `<action>...</action>` must remain — zero or multiple tags yield `None` (unparseable).
- The extracted action string is then passed to the phase-specific parser (`parse_action` or `parse_predictions`).

**Machine activation logic:**
- **Disjunctive** (OR): machine ON if *any* Blicket is on the machine.
- **Conjunctive** (AND): machine ON only if *all* Blickets are on the machine.

**Transition to answer phase** happens when the agent sends `exit` or exhausts `max_num_steps`. The transition message includes a full observation history recap so the agent can reason over all experiments at once.

**Action format (exploration):**
```xml
<reasoning>...</reasoning>
<action>put 3 on</action>
```
Valid actions: `put {id} on|off` (1-indexed) or `exit`.

**Answer format:**
```xml
<reasoning>...</reasoning>
<action>1: True, 2: False, 3: True, 4: False</action>
```

### File Structure (`CausalExplorerEnv.py`)

**Module-level constants:**
- `MAX_ANSWER_ATTEMPTS = 3` — maximum answer-phase retries before the episode ends with score 0.

**Entry point:**
- `load_environment(num_objects_range, num_examples, seed)` — validates parameter constraints, generates a diverse dataset with per-row config sampling (num_objects, num_blickets, rule_type, max_num_steps, blicket assignments), builds the parser/rubric, and returns a `BlicketEnv` instance.

**Environment class:**
- `BlicketEnv(vf.MultiTurnEnv)`
  - `setup_state()` — reads pre-computed per-row config from dataset info. Initializes blicket array, zeroed object/machine states, step counter, phase tracker, history log, hypothesis space, and action-tracking counters (`total_action_count`, `exploration_and_answer_count`, `parseable_action_count`, `valid_action_count`, `redundant_action_count`, `out_of_range_count`, `answer_attempt_count`).
  - `env_response()` — core game loop. Handles exploration (parse action via `parse_response`, validate, toggle, compute machine state, filter hypotheses, return observation) and answer phase (parse predictions via `parse_response`, retry loop up to `MAX_ANSWER_ATTEMPTS`, score and signal termination on success).
  - `_build_transition_message()` — assembles the observation history recap when moving to answer phase.

**Helper functions:**
- `compute_machine_state()` — vectorized OR/AND activation over Blicket positions.
- `is_consistent()` — checks if a blicket assignment is consistent with an observed machine state.
- `parse_action()` — regex parser for `put {id} on|off` and `exit`.
- `parse_predictions()` — regex parser for `1: True, 2: False, ...` answer format.
- `parse_response()` — strict action extractor: strips reasoning blocks, requires exactly one `<action>` tag, then delegates to `parse_action` (exploration) or `parse_predictions` (answer). Returns `None` on any violation.
- `build_system_prompt()` — constructs the system prompt with rules, action format, and strategy hint (does not reveal the rule type).
- `build_initial_message()` — constructs the opening user message presenting the game.
- `format_observation()` — formats a compact single-line observation (step counter, action, object lists, machine state).
- `format_history()` — formats the full observation history for the transition message.
- `compute_optimal_steps()` — simulates a greedy info-gain-maximizing agent to determine the optimal number of exploration steps for a given configuration.

**Reward functions:**
- `blicket_identification()` — reads `state["final_score"]`, set by `env_response` when a valid answer is submitted. Returns 0.0 if no valid answer was recorded.
- `step_budget_utilization()` — when `final_score < 1.0`: returns `step_count / max_steps` to encourage more exploration; when `final_score == 1.0`: returns 1.0.
- `exploration_efficiency()` — `1 - (wasted / parseable_action_count)`, where waste = redundant actions + out-of-range object IDs + non-contiguous configuration revisits. Higher is better.
- `format_compliance()` — `parseable_action_count / exploration_and_answer_count` across all turns in both phases. Higher is better.
- `hypotheses_eliminated()` — fraction of hypotheses eliminated relative to the optimal greedy info-gain agent. Higher means more informative exploration.

### Metrics / Counters

| Counter | Incremented when |
|---|---|
| `exploration_and_answer_count` | Every `env_response` call (both phases) |
| `total_action_count` | Every exploration-phase `env_response` call |
| `parseable_action_count` | Exploration: `parse_response` returns non-None. Answer: valid predictions submitted |
| `valid_action_count` | Exit action or non-redundant in-range toggle |
| `redundant_action_count` | Toggle targeting object already in requested state |
| `out_of_range_count` | Toggle with object ID outside [1, num_objects] |
| `answer_attempt_count` | Each answer-phase `env_response` call |

### Reward Table

| Component | Weight | Meaning |
| --------- | ------ | ------- |
| `blicket_identification` | 0.5 | Per-object accuracy of Blicket predictions |
| `step_budget_utilization` | 0.0 | — |
| `exploration_efficiency` | 0.0 | — |
| `format_compliance` | 0.0 | — |
| `hypotheses_eliminated` | 0.5 | Fraction of hypotheses eliminated vs. the optimal greedy info-gain agent. Higher is better |
