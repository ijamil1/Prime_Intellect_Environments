import json
import math
import re
from itertools import product

import numpy as np
import verifiers as vf
from datasets import Dataset


# --- Helper functions ---


def compute_machine_state(
    object_states: np.ndarray, blickets: np.ndarray, rule_type: str
) -> int:
    """Compute whether the machine is ON (1) or OFF (0).

    Args:
        object_states: Binary array of shape (num_objects,); 1 = on machine.
        blickets: Binary array of shape (num_objects,); 1 = is a Blicket.
        rule_type: "disjunctive" or "conjunctive".
    """
    active_blickets = object_states[blickets == 1]
    if rule_type == "disjunctive":
        return int(active_blickets.any())
    else:  # conjunctive
        return int(active_blickets.all())


def is_consistent(
    object_states: np.ndarray,
    blickets: np.ndarray,
    rule_type: str,
    machine_state: int,
) -> int:
    """Check if a blicket assignment is consistent with an observed machine state.

    Returns 1 if compute_machine_state(object_states, blickets, rule_type) equals
    machine_state, 0 otherwise.
    """
    predicted = compute_machine_state(object_states, blickets, rule_type)
    return 1 if predicted == machine_state else 0


def parse_action(action_str: str) -> dict | None:
    """Parse an exploration-phase action string.

    Returns a dict with keys:
        - type: "toggle" | "exit"
        - id: int (1-indexed, only for toggle)
        - target: "on" | "off" (only for toggle)
    Returns None if unparseable.
    """
    action_str = action_str.strip().lower()
    if action_str == "exit":
        return {"type": "exit"}
    m = re.match(r"^put\s+(\d+)\s+(on|off)$", action_str)
    if m:
        return {"type": "toggle", "id": int(m.group(1)), "target": m.group(2)}
    return None


def parse_predictions(answer_str: str, num_objects: int) -> dict | None:
    """Parse answer-phase predictions like '1: True, 2: False, ...'.

    Returns dict mapping 1-indexed object id to bool, or None on failure.
    """
    predictions = {}
    for part in answer_str.split(","):
        part = part.strip()
        m = re.match(r"^(\d+)\s*:\s*(true|false)$", part, re.IGNORECASE)
        if not m:
            return None
        obj_id = int(m.group(1))
        value = m.group(2).lower() == "true"
        predictions[obj_id] = value
    if set(predictions.keys()) != set(range(1, num_objects + 1)):
        return None
    return predictions


def build_system_prompt(num_objects: int, max_num_steps: int) -> str:
    """Build the system prompt for the Blicket game."""
    object_list = ", ".join(str(i) for i in range(1, num_objects + 1))
    return f"""\

You are an intelligent, curious agent. You are playing a game where you are in a room with \
{num_objects} different objects, and a machine. The objects are labeled as such: {object_list}. Some of these objects are blickets. \
You can't tell which object is a blicket just by looking at it. \
Blickets make the machine turn on, following some hidden rule.

More precisely, a blicket is defined as an object whose state is not independent of the
state of the machine (in other words, the object's state (on/off the machine) distribution and the machine's distribution have nonzero mutual
information)

Your goal is to determine exactly which objects are Blickets through exploration.
You have a maximum of {max_num_steps} steps to conduct the exploration phase so you must act efficiently. You can also exit this phase early if you think you understand the relationship between the
objects and the machine. After the exploration phase is done, you will be asked which objects are blickets.

RULES:
- In each action, you can place exactly one object onto the machine or remove exactly one object off the machine.
- After each action, you will observe which objects are on the machine and whether the machine is ON or OFF.
- When you have gathered enough information to determine which objects are Blickets, you can exit the exploration phase to submit your answer.

ACTION FORMAT (use XML tags):

During exploration, respond with:
<reasoning>Your reasoning about the decision to place an on object on the machine...</reasoning>
<action>put N on</action>
or
<reasoning>Your reasoning to remove an object from the machines...</reasoning>
<action>put N off</action>
Where N is an object number in ({object_list}).

To exit the exploration phase and enter the answer phase, respond with: 
<reasoning>Your reasoning for stopping...</reasoning>
<action>exit</action>

During the answer phase, respond with:
<reasoning>Your analysis of which objects are Blickets...</reasoning>
<action>1: True, 2: False, ...</action>

Where True means the object is a Blicket and False means it is not. You must provide a prediction for every object.

STRATEGY: Plan your experiments carefully to gather maximum information efficiently since you are limited by the number of actions you can take. \
Reason about what actions will give you the most information and what each observation tells you about the hidden rule and which objects might be Blickets."""


def build_initial_message(num_objects: int) -> str:
    """Build the initial user message presenting the game."""
    object_list = ", ".join(str(i) for i in range(1, num_objects + 1))
    return f"""\
You are in front of a Blicket-detecting machine with {num_objects} objects: {object_list}.
Currently, no objects are on the machine. The machine is OFF. Your task is to determine which objects \
are blickets.

Begin"""


def format_observation(
    step_count: int,
    max_num_steps: int,
    action_desc: str,
    object_states: np.ndarray,
    machine_state: int,
) -> str:
    """Format the observation after an exploration action."""
    on_objects = [str(i + 1) for i in range(len(object_states)) if object_states[i] == 1]
    off_objects = [str(i + 1) for i in range(len(object_states)) if object_states[i] == 0]
    on_list = f"[{', '.join(on_objects)}]" if on_objects else "[]"
    off_list = f"[{', '.join(off_objects)}]" if off_objects else "[]"
    machine = "ON" if machine_state else "OFF"
    return (
        f"Step {step_count}/{max_num_steps}: {action_desc}\n"
        f"Objects currently on the machine: {on_list}\n"
        f"Objects currently off the machine: {off_list}\n"
        f"Machine state: {machine}"
    )


def format_history(history: list) -> str:
    """Format the full observation history for the transition message."""
    lines = []
    for entry in history:
        step = entry["step"]
        action = entry["action"]
        on_objs = entry["on_objects"]
        off_objs = entry["off_objects"]
        machine = "ON" if entry["machine_state"] else "OFF"
        on_list = f"[{', '.join(str(o) for o in on_objs)}]" if on_objs else "[]"
        off_list = f"[{', '.join(str(o) for o in off_objs)}]" if off_objs else "[]"
        lines.append(
            f"Step {step}: {action} → Objects on: {on_list} | Objects off: {off_list} → Machine: {machine}"
        )
    return "\n".join(lines)


def compute_optimal_steps(
    num_objects: int,
    blickets: list[int],
    rule_type: str,
    num_samples: int = 50,
    seed: int = 0,
) -> float:
    """Compute average exploration steps for a greedy info-gain-maximizing agent.

    Simulates an agent that at each step selects the single-object toggle which
    best bisects the active hypothesis space.  Ties are broken by:
      1. Preferring actions that lead to a never-before-seen object configuration.
      2. Randomly choosing among remaining tied actions.
    Because of (2), the simulation is run *num_samples* times with different RNG
    seeds and the results are averaged.

    Outcomes are determined by the true *blickets* and *rule_type*.  Exploration
    stops when only one hypothesis remains.
    """
    rng = np.random.default_rng(seed)
    run_seeds = rng.integers(0, 2**31, size=num_samples)

    def predict(obj_states, blicket_bits, rule):
        """Pure-Python machine-state prediction for a hypothesis."""
        active = [obj_states[i] for i in range(num_objects) if blicket_bits[i]]
        return int(any(active)) if rule == "disjunctive" else int(all(active))

    # Full hypothesis space: 2^N blicket assignments × 2 rule types
    all_hypotheses = [
        (bits, rule)
        for bits in product((0, 1), repeat=num_objects)
        for rule in ("disjunctive", "conjunctive")
    ]

    # Pre-filter against the free initial observation (shared across runs)
    init_state = tuple([0] * num_objects)
    init_machine = predict(init_state, blickets, rule_type)
    init_active = [
        (bits, rule) for bits, rule in all_hypotheses
        if predict(init_state, bits, rule) == init_machine
    ]

    max_iters = 2 ** (num_objects + 1)  # safety bound
    total_steps = 0

    for rs in run_seeds:
        run_rng = np.random.default_rng(int(rs))
        obj_states = list(init_state)
        active = list(init_active)
        visited = {init_state}
        steps = 0

        while steps < max_iters and len(active) > 1:
            # Compute balance for every possible toggle
            candidates = []
            for i in range(num_objects):
                trial = list(obj_states)
                trial[i] = 1 - trial[i]
                on_count = sum(
                    1 for bits, rule in active if predict(trial, bits, rule) == 1
                )
                bal = min(on_count, len(active) - on_count)
                is_new = tuple(trial) not in visited
                candidates.append((i, bal, is_new))

            best_bal = max(b for _, b, _ in candidates)
            tied = [(i, is_new) for i, b, is_new in candidates if b == best_bal]

            # Tiebreak 1: prefer actions leading to unseen configurations
            unseen = [i for i, is_new in tied if is_new]
            if unseen:
                action = int(run_rng.choice(unseen))
            else:
                action = int(run_rng.choice([i for i, _ in tied]))

            obj_states[action] = 1 - obj_states[action]
            visited.add(tuple(obj_states))
            actual = predict(obj_states, blickets, rule_type)
            active = [
                (b, r) for b, r in active if predict(obj_states, b, r) == actual
            ]
            steps += 1

        total_steps += steps

    return total_steps / num_samples, len(all_hypotheses) - 1


# --- Environment class ---


class BlicketEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        # Read per-row config from dataset info (pre-computed in load_environment)
        info = state["info"]
        num_objects = info["num_objects"]
        max_num_steps = info["max_num_steps"]
        blickets = np.array(info["blickets"], dtype=int)

        state["blickets"] = blickets
        state["object_states"] = np.zeros(num_objects, dtype=int)
        state["machine_state"] = 0
        state["rule_type"] = info["rule_type"]
        state["step_count"] = 0
        state["phase"] = "exploration"
        state["history"] = []
        state["num_objects"] = num_objects
        state["max_num_steps"] = max_num_steps
        state["valid_action_count"] = 0
        state["parseable_action_count"] = 0
        state["total_action_count"] = 0
        state["redundant_action_count"] = 0
        state["optimal_hypotheses_eliminated"] = info["optimal_hypotheses_eliminated"]

        # Initialize hypothesis space: all (blicket_assignment, rule_type) pairs
        # 2^N blicket assignments × 2 rule types = 2^(N+1) hypotheses
        state["valid_hypotheses"] = [
            (bits, rule)
            for bits in product((0, 1), repeat=num_objects)
            for rule in ("disjunctive", "conjunctive")
        ]
        state["hypotheses_eliminated_per_step"] = []

        return await super().setup_state(state, **kwargs)

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        # Extract the most recent assistant message and parse its XML tags
        content = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = str(msg["content"])
                break
        parsed = self.parser.parse(content)
        action_str = parsed.action if parsed.action else ""

        # --- Answer phase: agent has submitted predictions ---
        if state["phase"] == "answer":
            predictions = parse_predictions(action_str, state["num_objects"])
            if predictions is not None:
                correct = sum(
                    1 for i in range(state["num_objects"])
                    if predictions.get(i + 1) == bool(state["blickets"][i])
                )
                score = correct / state["num_objects"]
                blicket_list = [str(i + 1) for i in range(state["num_objects"]) if state["blickets"][i] == 1]
                final_msg = (
                    f"Your answer has been recorded. "
                    f"You correctly identified {correct}/{state['num_objects']} objects. "
                    f"Score: {score:.2f}\n"
                    f"The Blickets were: [{', '.join(blicket_list)}]\n"
                    f"The rule was: {state['rule_type']}"
                )
            else:
                blicket_list = [str(i + 1) for i in range(state["num_objects"]) if state["blickets"][i] == 1]
                final_msg = (
                    f"Could not parse your answer. Please use the format: 1: True, 2: False, ...\n"
                    f"The Blickets were: [{', '.join(blicket_list)}]\n"
                    f"The rule was: {state['rule_type']}"
                )
            final_response = [{"role": "user", "content": final_msg}]
            state["final_env_response"] = final_response
            return final_response

        # --- Exploration phase ---
        state["total_action_count"] += 1
        action = parse_action(action_str)

        # Handle exit action
        if action is not None and action["type"] == "exit":
            state["parseable_action_count"] += 1
            state["valid_action_count"] += 1
            state["phase"] = "answer"
            return self._build_transition_message(state)

        # Check if we've hit the step limit (before processing the action)
        # The current action still counts as a step
        state["step_count"] += 1

        if action is None:
            # Unparseable action
            error_msg = (
                f"Step {state['step_count']}/{state['max_num_steps']}: "
                f"Invalid action format. Expected <action>put N on</action>, "
                f"<action>put N off</action>, or <action>exit</action>, "
                f"where N is an object number between 1 and {state['num_objects']}."
            )
            if state["step_count"] >= state["max_num_steps"]:
                state["phase"] = "answer"
                return [{"role": "user", "content": error_msg + "\n\n"}] + self._build_transition_message(state)
            return [{"role": "user", "content": error_msg}]

        state["parseable_action_count"] += 1

        # Validate toggle action
        obj_id = action["id"]
        target = action["target"]

        if obj_id < 1 or obj_id > state["num_objects"]:
            error_msg = (
                f"Step {state['step_count']}/{state['max_num_steps']}: "
                f"Invalid object ID {obj_id}. Must be between 1 and {state['num_objects']}."
            )
            if state["step_count"] >= state["max_num_steps"]:
                state["phase"] = "answer"
                return [{"role": "user", "content": error_msg + "\n\n"}] + self._build_transition_message(state)
            return [{"role": "user", "content": error_msg}]

        current_state = state["object_states"][obj_id - 1]
        target_state = 1 if target == "on" else 0

        if current_state == target_state:
            state["redundant_action_count"] += 1
            already = "on" if current_state == 1 else "off"
            error_msg = (
                f"Step {state['step_count']}/{state['max_num_steps']}: "
                f"Object {obj_id} is already {already} the machine. This is a no-op as it is redundant!"
            )
            if state["step_count"] >= state["max_num_steps"]:
                state["phase"] = "answer"
                return [{"role": "user", "content": error_msg + "\n\n"}] + self._build_transition_message(state)
            return [{"role": "user", "content": error_msg}]

        # Valid toggle - apply it
        state["valid_action_count"] += 1
        state["object_states"][obj_id - 1] = target_state
        state["machine_state"] = compute_machine_state(
            state["object_states"], state["blickets"], state["rule_type"]
        )

        # Filter hypotheses against this observation
        prev_count = len(state["valid_hypotheses"])
        state["valid_hypotheses"] = [
            (bits, rule) for bits, rule in state["valid_hypotheses"]
            if is_consistent(state["object_states"], np.array(bits), rule, state["machine_state"])
        ]
        state["hypotheses_eliminated_per_step"].append(prev_count - len(state["valid_hypotheses"]))

        # Build action description
        if target == "on":
            action_desc = f"You placed object {obj_id} on the machine."
        else:
            action_desc = f"You removed object {obj_id} from the machine."

        # Record history
        on_objects = [i + 1 for i in range(state["num_objects"]) if state["object_states"][i] == 1]
        off_objects = [i + 1 for i in range(state["num_objects"]) if state["object_states"][i] == 0]
        state["history"].append({
            "step": state["step_count"],
            "action": f"put {obj_id} {target}",
            "on_objects": on_objects,
            "off_objects": off_objects,
            "machine_state": state["machine_state"],
        })

        observation = format_observation(
            state["step_count"],
            state["max_num_steps"],
            action_desc,
            state["object_states"],
            state["machine_state"],
        )

        # Check if step limit reached after this step
        if state["step_count"] >= state["max_num_steps"]:
            state["phase"] = "answer"
            return [{"role": "user", "content": observation + "\n\n"}] + self._build_transition_message(state)

        return [{"role": "user", "content": observation}]

    def _build_transition_message(self, state: vf.State) -> list[dict]:
        """Build the transition message from exploration to answer phase."""
        history_str = format_history(state["history"]) if state["history"] else "No valid exploration steps were taken."
        msg = (
            f"Exploration complete. You used {state['step_count']} of {state['max_num_steps']} steps.\n\n"
            f"Here is your full observation history:\n"
            f"{history_str}\n\n"
            f"Now identify which objects are Blickets. For each object, respond True or False.\n"
            f"Use the format: 1: True, 2: False, ..."
        )
        return [{"role": "user", "content": msg}]


# --- Reward and metric functions ---


async def blicket_identification(completion, state, parser) -> float:
    """Primary reward: per-object accuracy of Blicket identification."""
    action_str = parser.parse_answer(completion)
    if action_str is None:
        action_str = ''
    predictions = parse_predictions(action_str, state["num_objects"])
    ground_truth = state["blickets"]

    if predictions is None:
        return 0.0

    correct = sum(
        1 for i in range(len(ground_truth))
        if predictions.get(i + 1) == bool(ground_truth[i])
    )
    return correct / len(ground_truth)


async def step_budget_utilization(state) -> float:
    """Metric: fraction of the step budget consumed (steps_used / max_steps).

    Returns a value in [0, 1]. A value of 1.0 means the agent used the entire
    budget; lower values indicate the agent exited early.
    """
    steps_used = state.get("step_count", 0)
    max_steps = state.get("max_num_steps", 1)
    return steps_used / max_steps


async def exploration_inefficiency(state) -> float:
    """Metric: fraction of parseable actions that were wasted.

    Counts two disjoint sources of waste relative to parseable actions:
    1. Redundant actions — no-ops where the object was already in the
       requested state (blocked by env_response, not added to history).
    2. Non-contiguous revisits — valid actions that reproduce a machine
       configuration already seen earlier in the history.

    Returns (redundant + revisits) / parseable_action_count, a value in
    [0, 1].  Lower is better (0 = no wasted actions).  Returns 0.0 when
    no parseable actions were taken.
    """
    parseable = state.get("parseable_action_count", 0)
    if parseable == 0:
        return 0.0

    redundant = state.get("redundant_action_count", 0)

    # Count non-contiguous revisits from history
    history = state.get("history", [])
    seen = set()
    non_contiguous = 0
    for entry in history:
        config = frozenset(entry["on_objects"])
        if config in seen:
            non_contiguous += 1
        seen.add(config)

    return float(redundant + non_contiguous) / parseable


async def format_compliance(state) -> float:
    """Metric: fraction of exploration turns with parseable AND valid actions."""
    total = state.get("total_action_count", 0)
    if total == 0:
        return 1.0
    valid = state.get("valid_action_count", 0)
    return valid / total


async def hypotheses_eliminated(state) -> float:
    """Metric: fraction of hypotheses eliminated relative to the optimal agent.

    The denominator is the number of hypotheses that need to be eliminated to
    uniquely identify the true hypothesis (2^(N+1) - 1, pre-computed at
    dataset generation time from the greedy info-gain simulation).  The
    numerator is the cumulative count of hypotheses the agent actually
    eliminated across all valid exploration steps.

    Returns a value in [0, 1].  Higher = more informative exploration.
    """
    optimal = state.get("optimal_hypotheses_eliminated", 1)
    eliminated = float(sum(state.get("hypotheses_eliminated_per_step", [])))
    return min(1.0, eliminated / optimal)


# --- Entry point ---


def load_environment(
    num_objects_range: tuple[int, int] = (3, 6),
    num_examples: int = 100,
    seed: int = 42,
) -> vf.Environment:
    """Load the CausalExplorerEnv (Blicket machine) environment.

    Each dataset row is generated with a unique configuration sampled from
    the provided ranges, producing a diverse evaluation across varying
    numbers of objects, blickets, step budgets, and rule types.

    Args:
        num_objects_range: Inclusive (min, max) range for number of objects per row.
        num_examples: Number of dataset rows to generate.
        seed: RNG seed for reproducible dataset generation.
    """

    # Validate ranges
    obj_lo, obj_hi = num_objects_range
    if not (2 <= obj_lo <= obj_hi <= 10):
        raise ValueError(
            f"num_objects_range must satisfy 2 <= lo <= hi <= 10, got {num_objects_range}"
        )

    # Generate diverse dataset rows
    rng = np.random.default_rng(seed)
    rows = []
    global_max_steps = 0

    for _ in range(num_examples):
        # Sample num_objects
        n_obj = int(rng.integers(obj_lo, obj_hi + 1))

        # Auto-derive num_blickets: [2, n_obj]
        n_blick = int(rng.integers(2, n_obj + 1))

        # Sample rule type for this row
        row_rule = rng.choice(["disjunctive", "conjunctive"])

        # Assign blickets at dataset generation time
        blicket_indices = sorted(rng.choice(n_obj, size=n_blick, replace=False).tolist())
        blickets = [0] * n_obj
        for idx in blicket_indices:
            blickets[idx] = 1

        # Compute optimal exploration steps via greedy info-gain simulation,
        # then give the agent a 20% budget cushion
        row_seed = int(rng.integers(0, 2**31))
        optimal_steps, hyps_elim_by_opt_agent = compute_optimal_steps(n_obj, blickets, row_rule, seed=row_seed)
        max_steps = max(1, math.ceil(1.2 * optimal_steps))
        global_max_steps = max(global_max_steps, max_steps)

        # Build per-row prompt with embedded system prompt
        system_msg = {"role": "system", "content": build_system_prompt(n_obj, max_steps)}
        user_msg = {"role": "user", "content": build_initial_message(n_obj)}

        rows.append({
            "prompt": [system_msg, user_msg],
            "info": json.dumps({
                "num_objects": n_obj,
                "num_blickets": n_blick,
                "max_num_steps": max_steps,
                "rule_type": row_rule,
                "blickets": blickets,
                "optimal_hypotheses_eliminated": hyps_elim_by_opt_agent
            }),
        })

    dataset = Dataset.from_list(rows)

    # Build parser (shared between env and rubric)
    parser = vf.XMLParser(fields=["reasoning", "action"], answer_field="action")

    # Build rubric
    rubric = vf.Rubric(funcs=[blicket_identification], weights=[1.0], parser=parser)
    rubric.add_metric(step_budget_utilization)
    rubric.add_metric(exploration_inefficiency)
    rubric.add_metric(format_compliance)
    rubric.add_metric(hypotheses_eliminated)

    # max_turns = max possible steps across all rows + 2 (transition + answer)
    max_turns = global_max_steps + 2

    return BlicketEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        max_turns=max_turns,
    )
