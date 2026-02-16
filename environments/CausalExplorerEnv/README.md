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
- **Rubric overview**: Primary reward is per-object Blicket identification accuracy. Metrics track step budget utilization, exploration inefficiency, format compliance, and hypotheses eliminated.

The agent interacts with a simulated "Blicket-detecting machine" across two phases:
1. **Exploration phase** — toggle objects on/off the machine one at a time, observe whether the machine activates, and exit when ready.
2. **Answer phase** — declare which objects are Blickets.

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
| `num_objects_range` | tuple[int, int] | `(3, 6)` | Inclusive (min, max) range for number of objects per row. Must satisfy 2 <= lo <= hi <= 10 |
| `num_examples` | int | `100` | Number of dataset rows to generate |
| `seed` | int | `42` | RNG seed for reproducible dataset generation |

Per-row configuration is sampled dynamically at dataset generation time:
- **num_objects**: uniformly sampled from `num_objects_range`
- **num_blickets**: uniformly sampled from [2, num_objects]
- **rule_type**: randomly chosen as `"disjunctive"` or `"conjunctive"`
- **max_num_steps**: computed as `ceil(1.2 * optimal_steps)` where `optimal_steps` comes from a greedy info-gain simulation

### Architecture

`BlicketEnv` subclasses `vf.MultiTurnEnv`. The verifiers `max_turns` is set to `global_max_steps + 2` (the largest step budget across all rows + answer-phase transition + answer response).

**Rollout lifecycle:**

1. `setup_state()` reads per-row config from the dataset `info` field (blickets, rule type, step budget are pre-assigned at dataset generation). Initializes zeroed object states, the full hypothesis space (2^N blicket assignments × 2 rule types), and tracking counters.
2. `env_response()` drives the game loop across both phases:
   - **Exploration**: parses XML actions (`put N on/off` or `exit`), validates them, toggles object state, computes machine activation, filters the hypothesis space against the observation, and returns a compact single-line observation. Invalid and redundant actions still consume a step.
   - **Answer**: parses `1: True, 2: False, ...` predictions and terminates via `state["final_env_response"]`.
3. Termination is handled by the base class `has_final_env_response` stop condition.

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

**Entry point:**
- `load_environment(num_objects_range, num_examples, seed)` — validates parameter constraints, generates a diverse dataset with per-row config sampling (num_objects, num_blickets, rule_type, max_num_steps, blicket assignments), builds the parser/rubric, and returns a `BlicketEnv` instance.

**Environment class:**
- `BlicketEnv(vf.MultiTurnEnv)`
  - `setup_state()` — reads pre-computed per-row config from dataset info. Initializes blicket array, zeroed object/machine states, step counter, phase tracker, history log, hypothesis space, and action-tracking counters.
  - `env_response()` — core game loop. Handles exploration (parse action, validate, toggle, compute machine state, filter hypotheses, return observation) and answer phase (parse predictions, signal termination).
  - `_build_transition_message()` — assembles the observation history recap when moving to answer phase.

**Helper functions:**
- `compute_machine_state()` — vectorized OR/AND activation over Blicket positions.
- `is_consistent()` — checks if a blicket assignment is consistent with an observed machine state.
- `parse_action()` — regex parser for `put {id} on|off` and `exit`.
- `parse_predictions()` — regex parser for `1: True, 2: False, ...` answer format.
- `build_system_prompt()` — constructs the system prompt with rules, action format, and strategy hint (does not reveal the rule type).
- `build_initial_message()` — constructs the opening user message presenting the game.
- `format_observation()` — formats a compact single-line observation (step counter, action, object lists, machine state).
- `format_history()` — formats the full observation history for the transition message.
- `compute_optimal_steps()` — simulates a greedy info-gain-maximizing agent to determine the optimal number of exploration steps for a given configuration.

**Reward & metrics:**
- `blicket_identification()` — primary reward (weight 1.0): per-object accuracy comparing predictions to ground truth.
- `step_budget_utilization()` — metric: `steps_used / max_steps`. Value of 1.0 means the entire budget was used; lower means the agent exited early.
- `exploration_inefficiency()` — metric: fraction of parseable actions that were wasted (redundant no-ops + non-contiguous revisits of previously seen configurations). Lower is better.
- `format_compliance()` — metric: fraction of exploration turns with parseable AND valid actions.
- `hypotheses_eliminated()` — metric: fraction of hypotheses eliminated relative to the optimal info-gain agent. Higher means more informative exploration.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `blicket_identification` | Per-object accuracy of Blicket predictions (primary reward, weight 1.0) |
| `step_budget_utilization` | `steps_used / max_steps` — fraction of step budget consumed |
| `exploration_inefficiency` | `(redundant + revisits) / parseable_actions` — fraction of wasted actions. Lower is better |
| `format_compliance` | Fraction of exploration turns with parseable AND valid actions |
| `hypotheses_eliminated` | Fraction of hypotheses eliminated vs. the optimal greedy info-gain agent. Higher is better |
