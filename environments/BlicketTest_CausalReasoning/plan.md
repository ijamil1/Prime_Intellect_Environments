# CausalExplorerEnv — Implementation Plan

Reference paper: [Do LLMs Think Like Scientists? Causal Reasoning and Hypothesis Testing in LLMs](https://arxiv.org/pdf/2505.09614)

## 1. What This Environment Is

A multi-turn text-based game that tests an LLM's ability to **reason causally and explore hypotheses efficiently**. The agent interacts with a simulated "Blicket-detecting machine" to discover which objects (out of N) are "Blickets" — objects that activate the machine according to a hidden rule.

The environment has two phases:
1. **Exploration phase** — the agent toggles objects on/off the machine one at a time, observes machine state (on/off), and can exit early.
2. **Answer phase** — the agent declares which objects are Blickets.

## 2. Environment Parameters

`load_environment` accepts these arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_objects` | `int` | 4 | Number of objects (N) |
| `num_blickets` | `int` | 2 | How many objects are Blickets |
| `max_num_steps` | `int` | 32 | Max exploration steps before forced answer |
| `rule_type` | `str \| None` | `None` | `"disjunctive"`, `"conjunctive"`, or `None` (random) |
| `num_examples` | `int` | 100 | Number of dataset rows |
| `seed` | `int` | 42 | RNG seed |

### Parameter Constraints (validate in `load_environment`)

```
2 <= num_blickets <= num_objects
2 <= num_objects <= 10
2**num_objects <= max_num_steps <= 2**(num_objects + 1)
```

Raise `ValueError` with a clear message if any constraint is violated.

If `rule_type is None`, each episode randomly selects `"disjunctive"` or `"conjunctive"`.

## 3. Architecture Overview

```
vf.MultiTurnEnv (subclass: BlicketEnv)
├── Dataset: repeated prompt rows (each gets a fresh random Blicket assignment)
├── System prompt: rules of the game, action format, answer format
├── env_response(): processes agent actions, returns machine feedback
├── setup_state(): initializes Blicket assignment, object states, step counter
├── @vf.stop conditions: exit action or max steps reached
└── Rubric: scores final Blicket identification accuracy
```

## 4. Game Mechanics

### 4.1 State Representation

Per-rollout state (initialized in `setup_state`):

```python
state["blickets"]       # np.array of shape (num_objects,), binary; 1 = Blicket
state["object_states"]  # np.array of shape (num_objects,), binary; 1 = on machine
state["machine_state"]  # int, 0 or 1
state["rule_type"]      # "disjunctive" or "conjunctive" (resolved from None if needed)
state["step_count"]     # int, current exploration step
state["phase"]          # "exploration" or "answer"
state["history"]        # list of (action, machine_state) tuples for observation log
```

### 4.2 Machine Activation Logic

Given `object_states` (x) and `blickets` (b):

```python
active_blickets = object_states[blickets == 1]  # extract blicket positions from x
if rule_type == "disjunctive":
    machine_state = int(active_blickets.any())   # OR: any Blicket on → machine on
elif rule_type == "conjunctive":
    machine_state = int(active_blickets.all())   # AND: all Blickets on → machine on
```

### 4.3 Action Space (Exploration Phase)

The agent can do exactly one of:
- **Toggle one object**: place or remove a single object (flips exactly one index in `object_states`)
- **Exit**: end exploration early and move to answer phase

### 4.4 Action Format

Use `vf.XMLParser` with fields `["reasoning", "action"]`.

Objects are identified by 1-indexed integers: `1`, `2`, ..., `N`.

During **exploration**, the agent outputs:
```xml
<reasoning>My hypothesis is...</reasoning>
<action>put 3 on</action>
```
or
```xml
<reasoning>I have enough information.</reasoning>
<action>exit</action>
```

Valid action syntax: `put {id} on|off` or `exit`, where `{id}` is an integer in `[1, N]`.

During the **answer phase**, the agent outputs:
```xml
<reasoning>Based on my observations...</reasoning>
<action>1: True, 2: False, 3: True, 4: False</action>
```

### 4.5 Exploration Phase Flow

Start state: all objects off the machine, machine off.

Each step:
1. Agent sends an action (parsed via XMLParser).
2. `env_response()` validates the action:
   - If `exit` → transition to answer phase.
   - If `put {id} on|off` → validate the object ID is in range `[1, N]` and that this actually toggles the object (i.e., it's not already in the requested state).
   - If invalid or no-op → return an error message. **The step still counts toward the limit** (incentivizes clean, intentional actions).
3. Apply the toggle to `object_states`.
4. Compute `machine_state` using the activation logic.
5. Increment `step_count`.
6. Return full state observation:
   ```
   Step {n}/{max}: You placed object {id} on / removed object {id} from the machine.
   Objects currently on the machine: [1, 4]
   Objects currently off the machine: [2, 3]
   Machine state: ON / OFF
   ```

### 4.6 Answer Phase Flow

When the agent exits (or hits `max_num_steps`), `env_response` provides a transition message:

```
Exploration complete. You used {n} of {max} steps.

Here is your full observation history:
Step 1: put 1 on → Objects on: [1] | Objects off: [2, 3, 4] → Machine: OFF
Step 2: put 2 on → Objects on: [1, 2] | Objects off: [3, 4] → Machine: ON
...

Now identify which objects are Blickets. For each object, respond True or False.
```

The full observation history is included to mirror the paper's setup and give the agent a consolidated view for its final reasoning.

The agent's next response is its final answer. After parsing the answer, set `state["final_env_response"]` to signal termination.

## 5. System Prompt

The system prompt should convey:
1. **Setting**: "You are interacting with a Blicket-detecting machine with N objects."
2. **Rules**: Explain that some objects are Blickets, and the machine activates based on a hidden rule. Do NOT reveal whether the rule is disjunctive or conjunctive.
3. **Goal**: "Determine which objects are Blickets through experimentation."
4. **Action format**: Specify the XML format and valid actions.
5. **Constraint**: "You can place or remove exactly one object per step."
6. **Strategy hint**: "Plan your experiments carefully to gather maximum information efficiently."

**Important**: Do NOT reveal whether the rule is disjunctive or conjunctive. Discovering the rule type is part of the challenge.

## 6. Dataset

Each row in the dataset corresponds to one episode. Since the Blicket assignment is randomized per rollout in `setup_state`, the dataset rows are essentially identical prompts — only the random state differs.

Structure:
```python
dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": initial_message}],
        "info": json.dumps({
            "num_objects": num_objects,
            "num_blickets": num_blickets,
            "max_num_steps": max_num_steps,
            "rule_type": rule_type,  # None or specific
        })
    }
    for _ in range(num_examples)
])
```

The `initial_message` presents the game:
```
You are in front of a Blicket-detecting machine with {N} objects: 1, 2, ..., {N}.
Some of these objects are "Blickets" that activate the machine according to a hidden rule.
Currently, no objects are on the machine. The machine is OFF.

Begin your exploration.
```

The dataset uses a simple fixed-config approach — all rows share the same parameters. Varied-config (nested) design is a future extension.

## 7. Reward Function

The reward evaluates the agent's final Blicket identification. Parse the answer phase response to extract per-object True/False predictions and compare against ground truth.

### Primary reward: per-object accuracy

```python
async def blicket_identification(completion, state, parser) -> float:
    parsed = parser.parse(completion)
    predictions = parse_predictions(parsed.action, state["num_objects"])  # dict: obj_id -> bool
    ground_truth = state["blickets"]  # binary array

    if predictions is None:
        return 0.0  # failed to parse

    correct = sum(
        1 for i in range(len(ground_truth))
        if predictions.get(i + 1) == bool(ground_truth[i])
    )
    return correct / len(ground_truth)
```

This naturally rewards true positives and true negatives, and penalizes false positives and false negatives — each misclassified object reduces the score by `1/N`.

### Metrics (weight 0.0 — tracked but do not affect reward)

These are registered via `rubric.add_metric()` so they appear in rollout results for analysis.

| Metric | Formula | Notes |
|--------|---------|-------|
| `exploration_efficiency` | `1.0 - (steps_used / max_steps)` | Higher = fewer steps used. Key signal of causal reasoning per the paper. |
| `format_compliance` | Fraction of exploration turns where the agent produced parseable XML with a valid action | Tracks how well the agent follows the protocol. |
| `hypotheses_eliminated` | **Placeholder** — return `0.0` for now | Future: compute information-theoretic measure of how many candidate Blicket assignments are ruled out per turn. |

## 8. Implementation Checklist

### File: `CausalExplorerEnv.py`

- [ ] **Constants and imports**: `import verifiers as vf`, `import numpy as np`, `import json`, `from datasets import Dataset`
- [ ] **`SYSTEM_PROMPT`**: As described in Section 5
- [ ] **`BlicketEnv(vf.MultiTurnEnv)` class**:
  - [ ] `__init__`: Accept and store `num_objects`, `num_blickets`, `max_num_steps`, `rule_type`. Call `super().__init__(dataset=..., rubric=..., parser=..., system_prompt=..., max_turns=...)`. Note: `max_turns` should be `max_num_steps + 2` (exploration steps + answer phase transition + answer).
  - [ ] `setup_state(state)`: Initialize blicket assignment (random), object states (zeros), machine state (0), step count, phase, history. Use `state["info"]` for per-row config if using the nested approach.
  - [ ] `env_response(messages, state)`: Core game logic. Parse agent action, validate, update state, return observation or transition to answer phase. Handle the answer phase termination with `state["final_env_response"]`.
  - [ ] `@vf.stop` **`exploration_complete`**: Return `True` when phase is `"answer"` and agent has submitted final answer.
- [ ] **`compute_machine_state(object_states, blickets, rule_type)`**: Pure function, vectorized.
- [ ] **`parse_action(action_str)`**: Parse `put {id} on|off` or `exit`.
- [ ] **`parse_predictions(answer_str, num_objects)`**: Parse `1: True, 2: False, ...` into dict.
- [ ] **Reward function**:
  - [ ] `blicket_identification`: Per-object accuracy (weight 1.0)
- [ ] **Metrics** (weight 0.0, via `rubric.add_metric()`):
  - [ ] `exploration_efficiency`: `1.0 - (steps_used / max_steps)`
  - [ ] `format_compliance`: Fraction of turns with parseable valid XML actions
  - [ ] `hypotheses_eliminated`: Dummy placeholder, returns `0.0`
- [ ] **`load_environment(**kwargs)`**: Validate params, build dataset, build rubric, return `BlicketEnv`.

### File: `pyproject.toml`

- [ ] Update `description` and `tags` (`["multi-turn", "causal-reasoning", "eval", "train"]`)
- [ ] Add `numpy` to `dependencies`
- [ ] Set eval defaults: `num_examples = 20`, `rollouts_per_example = 5`

## 9. Key Implementation Notes

1. **XMLParser shared between env and rubric**: Instantiate once, pass to both `BlicketEnv(parser=...)` and `vf.Rubric(parser=...)`. Use `self.parser` in `env_response` and `parser` argument in reward functions.

2. **`max_turns` vs `max_num_steps`**: The verifiers `max_turns` counts model response turns. Set it to `max_num_steps + 2` to accommodate exploration + answer transition + answer response.

3. **Invalid actions count as steps**: If the agent gives an invalid action or a no-op (e.g., "put 1 on" when object 1 is already on), return a corrective message AND increment `step_count`. This incentivizes clean, intentional actions.

4. **Forced transition**: When `step_count >= max_num_steps` and the agent hasn't exited, `env_response` should force the transition to answer phase regardless.

5. **Randomness per rollout**: The Blicket assignment must be randomized in `setup_state`, not in dataset generation. This ensures different rollouts of the same dataset row get different Blicket configurations.

6. **Answer phase is exactly one turn**: After the transition prompt, the agent responds once with its predictions, and the episode ends.

## 10. Resolved Decisions

| Question | Decision |
|----------|----------|
| Object IDs | Plain integers `1` through `N` |
| Reveal rule type to agent? | No |
| Observation history on transition? | Yes, include full history |
| Dataset strategy | Simple fixed-config (nested is a future extension) |
| Invalid actions count as steps? | Yes |
| Observation format | Full state (objects on + objects off + machine state) |
| Extra reward components | Tracked as metrics (weight 0.0): `exploration_efficiency`, `format_compliance`, `hypotheses_eliminated` (placeholder) |
