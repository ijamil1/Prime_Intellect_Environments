# CausalExplorerEnv

### Overview
- **Environment ID**: `CausalExplorerEnv`
- **Short description**: Multi-turn causal reasoning environment based on the Blicket detector paradigm from developmental psychology. Tests an LLM's ability to design experiments, reason causally, and identify hidden causal structure.
- **Tags**: multi-turn, causal-reasoning, eval, train

### Reference
Based on [Do LLMs Think Like Scientists? Causal Reasoning and Hypothesis Testing in LLMs](https://arxiv.org/pdf/2505.09614).

### Task
- **Type**: multi-turn
- **Parser**: XMLParser (fields: `reasoning`, `action`)
- **Rubric overview**: Primary reward is per-object Blicket identification accuracy. Metrics track exploration efficiency, format compliance, and (placeholder) hypotheses eliminated.

The agent interacts with a simulated "Blicket-detecting machine" across two phases:
1. **Exploration phase** — toggle objects on/off the machine one at a time, observe whether the machine activates, and exit when ready.
2. **Answer phase** — declare which objects are Blickets.

The machine activates according to a hidden rule (disjunctive OR or conjunctive AND over Blicket objects). The agent must discover both which objects are Blickets and the nature of the rule through experimentation.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run CausalExplorerEnv
```

Configure model and sampling:

```bash
prime eval run CausalExplorerEnv \
  -m openai/gpt-4.1-mini \
  -n 20 -r 5 -t 4096 -T 0.7 \
  -a '{"num_objects": 4, "num_blickets": 2}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_objects` | int | `4` | Number of objects (N). Range: [2, 10] |
| `num_blickets` | int | `2` | How many objects are Blickets. Range: [2, num_objects] |
| `max_num_steps` | int | `32` | Max exploration steps before forced answer. Range: [2^N, 2^(N+1)] |
| `rule_type` | str \| None | `None` | `"disjunctive"`, `"conjunctive"`, or `None` (random per episode) |
| `num_examples` | int | `100` | Number of dataset rows |
| `seed` | int | `42` | RNG seed for reproducibility |

### Architecture

`BlicketEnv` subclasses `vf.MultiTurnEnv`. The verifiers `max_turns` is set to `max_num_steps + 2` (exploration steps + answer-phase transition + answer response).

**Rollout lifecycle:**

1. `setup_state()` initializes a random Blicket assignment per rollout (unique RNG via `seed + rollout_counter`), zeroed object states, and tracking counters.
2. `env_response()` drives the game loop across both phases:
   - **Exploration**: parses XML actions (`put N on/off` or `exit`), validates them, toggles object state, computes machine activation, and returns a full state observation. Invalid actions and no-ops still consume a step.
   - **Answer**: parses `1: True, 2: False, ...` predictions and terminates via `state["final_env_response"]`.
3. `@vf.stop exploration_complete` fires when the answer has been submitted (phase transitions to `"done"`).

**Machine activation logic:**
- **Disjunctive** (OR): machine ON if *any* Blicket is on the machine.
- **Conjunctive** (AND): machine ON only if *all* Blickets are on the machine.
- The rule type is hidden from the agent — discovering it is part of the challenge.

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
- `load_environment()` — validates parameter constraints, builds the dataset/parser/rubric, and returns a `BlicketEnv` instance.

**Environment class:**
- `BlicketEnv(vf.MultiTurnEnv)` — stores game config and a per-rollout counter for RNG seeding.
  - `setup_state()` — random Blicket assignment, zeroed object/machine states, step counter, phase tracker, history log, and action-tracking counters.
  - `env_response()` — core game loop. Handles exploration (parse action, validate, toggle, compute machine state, return observation) and answer phase (parse predictions, signal termination).
  - `_build_transition_message()` — assembles the observation history recap when moving to answer phase.
  - `@vf.stop exploration_complete` — stops the rollout after the answer is submitted.

**Helper functions:**
- `compute_machine_state()` — vectorized OR/AND activation over Blicket positions.
- `parse_action()` — regex parser for `put {id} on|off` and `exit`.
- `parse_predictions()` — regex parser for `1: True, 2: False, ...` answer format.
- `build_system_prompt()` — constructs the system prompt with rules, action format, and strategy hint (does not reveal the rule type).
- `build_initial_message()` — constructs the opening user message presenting the game.
- `format_observation()` — formats a single-step observation (step counter, action description, object lists, machine state).
- `format_history()` — formats the full observation history for the transition message.

**Reward & metrics:**
- `blicket_identification()` — primary reward (weight 1.0): per-object accuracy comparing predictions to ground truth.
- `exploration_efficiency()` — metric: `1.0 - (steps_used / max_steps)`.
- `format_compliance()` — metric: fraction of exploration turns with parseable actions.
- `hypotheses_eliminated()` — metric: placeholder returning 0.0.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `blicket_identification` | Per-object accuracy of Blicket predictions (primary reward, weight 1.0) |
| `exploration_efficiency` | `1.0 - (steps_used / max_steps)` — higher means fewer steps used |
| `format_compliance` | Fraction of exploration turns with parseable valid XML actions |
| `hypotheses_eliminated` | Placeholder (returns 0.0) for future information-theoretic analysis |
