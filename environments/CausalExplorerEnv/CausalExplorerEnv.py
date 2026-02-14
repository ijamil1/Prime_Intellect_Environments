import json
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


def build_system_prompt(num_objects: int) -> str:
    """Build the system prompt for the Blicket game."""
    object_list = ", ".join(str(i) for i in range(1, num_objects + 1))
    return f"""\

You are an intelligent, curious agent. You are playing a game where you are in a room with \
{num_objects} different objects, and a machine. The objects are labeled as such: {object_list}. Some of these objects are blickets. \
You can't tell which object is a blicket just by looking at it. \
Blickets make the machine turn on, following some hidden rule.

Your goal is to determine exactly which objects are Blickets through experimentation.

RULES:
- In each action, you can place exactly one object onto the machine or remove exactly one object off the machine.
- After each action, you will observe which objects are on the machine and whether the machine is ON or OFF.
- When you have gathered enough information, you can exit the exploration phase to submit your answer.

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

STRATEGY: Plan your experiments carefully to gather maximum information efficiently. \
Think about what each observation tells you about the hidden rule and which objects might be Blickets."""


def build_initial_message(num_objects: int) -> str:
    """Build the initial user message presenting the game."""
    object_list = ", ".join(str(i) for i in range(1, num_objects + 1))
    return f"""\
You are in front of a Blicket-detecting machine with {num_objects} objects: {object_list}.
Some of these objects are "Blickets" that activate the machine according to a hidden rule.
Currently, no objects are on the machine. The machine is OFF.

Begin your exploration."""


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


# --- Environment class ---


class BlicketEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        num_objects: int,
        num_blickets: int,
        max_num_steps: int,
        rule_type: str | None,
        seed: int,
        **kwargs,
    ):
        self.num_objects = num_objects
        self.num_blickets = num_blickets
        self.max_num_steps = max_num_steps
        self.rule_type = rule_type
        self.seed = seed
        self._rollout_counter = 0
        super().__init__(**kwargs)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        # Derive a unique RNG per rollout
        rng = np.random.default_rng(self.seed + self._rollout_counter)
        self._rollout_counter += 1

        # Resolve rule type
        if self.rule_type is None:
            resolved_rule = rng.choice(["disjunctive", "conjunctive"])
        else:
            resolved_rule = self.rule_type

        # Random Blicket assignment: choose num_blickets positions
        blicket_indices = rng.choice(self.num_objects, size=self.num_blickets, replace=False)
        blickets = np.zeros(self.num_objects, dtype=int)
        blickets[blicket_indices] = 1

        state["blickets"] = blickets
        state["object_states"] = np.zeros(self.num_objects, dtype=int)
        state["machine_state"] = 0
        state["rule_type"] = resolved_rule
        state["step_count"] = 0
        state["phase"] = "exploration"
        state["history"] = []
        state["num_objects"] = self.num_objects
        state["max_num_steps"] = self.max_num_steps
        state["valid_action_count"] = 0
        state["parseable_action_count"] = 0
        state["total_action_count"] = 0
        state["redundant_action_count"] = 0

        # Initialize hypothesis space: all (blicket_assignment, rule_type) pairs
        # 2^N blicket assignments × 2 rule types = 2^(N+1) hypotheses
        state["valid_hypotheses"] = [
            (bits, rule)
            for bits in product((0, 1), repeat=self.num_objects)
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
            predictions = parse_predictions(action_str, self.num_objects)
            if predictions is not None:
                correct = sum(
                    1 for i in range(self.num_objects)
                    if predictions.get(i + 1) == bool(state["blickets"][i])
                )
                score = correct / self.num_objects
                blicket_list = [str(i + 1) for i in range(self.num_objects) if state["blickets"][i] == 1]
                final_msg = (
                    f"Your answer has been recorded. "
                    f"You correctly identified {correct}/{self.num_objects} objects. "
                    f"Score: {score:.2f}\n"
                    f"The Blickets were: [{', '.join(blicket_list)}]\n"
                    f"The rule was: {state['rule_type']}"
                )
            else:
                blicket_list = [str(i + 1) for i in range(self.num_objects) if state["blickets"][i] == 1]
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
                f"Step {state['step_count']}/{self.max_num_steps}: "
                f"Invalid action format. Expected <action>put N on</action>, "
                f"<action>put N off</action>, or <action>exit</action>, "
                f"where N is an object number between 1 and {self.num_objects}."
            )
            if state["step_count"] >= self.max_num_steps:
                state["phase"] = "answer"
                return [{"role": "user", "content": error_msg + "\n\n"}] + self._build_transition_message(state)
            return [{"role": "user", "content": error_msg}]

        state["parseable_action_count"] += 1

        # Validate toggle action
        obj_id = action["id"]
        target = action["target"]

        if obj_id < 1 or obj_id > self.num_objects:
            error_msg = (
                f"Step {state['step_count']}/{self.max_num_steps}: "
                f"Invalid object ID {obj_id}. Must be between 1 and {self.num_objects}."
            )
            if state["step_count"] >= self.max_num_steps:
                state["phase"] = "answer"
                return [{"role": "user", "content": error_msg + "\n\n"}] + self._build_transition_message(state)
            return [{"role": "user", "content": error_msg}]

        current_state = state["object_states"][obj_id - 1]
        target_state = 1 if target == "on" else 0

        if current_state == target_state:
            state["redundant_action_count"] += 1
            already = "on" if current_state == 1 else "off"
            error_msg = (
                f"Step {state['step_count']}/{self.max_num_steps}: "
                f"Object {obj_id} is already {already} the machine. This is a no-op as it is redundant!"
            )
            if state["step_count"] >= self.max_num_steps:
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
        on_objects = [i + 1 for i in range(self.num_objects) if state["object_states"][i] == 1]
        off_objects = [i + 1 for i in range(self.num_objects) if state["object_states"][i] == 0]
        state["history"].append({
            "step": state["step_count"],
            "action": f"put {obj_id} {target}",
            "on_objects": on_objects,
            "off_objects": off_objects,
            "machine_state": state["machine_state"],
        })

        observation = format_observation(
            state["step_count"],
            self.max_num_steps,
            action_desc,
            state["object_states"],
            state["machine_state"],
        )

        # Check if step limit reached after this step
        if state["step_count"] >= self.max_num_steps:
            state["phase"] = "answer"
            return [{"role": "user", "content": observation + "\n\n"}] + self._build_transition_message(state)

        return [{"role": "user", "content": observation}]

    def _build_transition_message(self, state: vf.State) -> list[dict]:
        """Build the transition message from exploration to answer phase."""
        history_str = format_history(state["history"]) if state["history"] else "No valid exploration steps were taken."
        msg = (
            f"Exploration complete. You used {state['step_count']} of {self.max_num_steps} steps.\n\n"
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
    """Metric: 1.0 - (steps_used / max_steps). Higher = fewer steps used."""
    steps_used = state.get("step_count", 0)
    max_steps = state.get("max_num_steps", 1)
    return 1.0 - (steps_used / max_steps)


async def exploration_inefficiency(state) -> float:
    """Metric: total number of revisited configurations.

    Counts two sources of revisits:
    1. Redundant actions (no-ops blocked by env_response) — tracked in state.
    2. Non-contiguous revisits — valid actions that produce a configuration
       already seen earlier in the history.

    Lower is better (0 = no wasted actions).
    """
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

    return float(redundant + non_contiguous)


async def format_compliance(state) -> float:
    """Metric: fraction of exploration turns with parseable AND valid actions."""
    total = state.get("total_action_count", 0)
    if total == 0:
        return 1.0
    valid = state.get("valid_action_count", 0)
    return valid / total


async def hypotheses_eliminated(state) -> float:
    """Metric: total number of hypotheses eliminated during exploration.

    A hypothesis is a (blicket_assignment, rule_type) pair. After each valid action,
    hypotheses inconsistent with the observed machine state are removed. This metric
    returns the cumulative count of eliminated hypotheses across all steps.

    Higher = more informative exploration.
    """
    return float(sum(state.get("hypotheses_eliminated_per_step", [])))


# --- Entry point ---


def load_environment(
    num_objects: int = 4,
    num_blickets: int = 2,
    max_num_steps: int = 32,
    rule_type: str | None = None,
    num_examples: int = 100,
    seed: int = 42,
) -> vf.Environment:
    """Load the CausalExplorerEnv (Blicket machine) environment."""

    # Validate parameters
    if not (2 <= num_blickets <= num_objects):
        raise ValueError(
            f"num_blickets must satisfy 2 <= num_blickets <= num_objects, "
            f"got num_blickets={num_blickets}, num_objects={num_objects}"
        )
    if not (2 <= num_objects <= 10):
        raise ValueError(f"num_objects must satisfy 2 <= num_objects <= 10, got {num_objects}")
    if not (2**num_objects <= max_num_steps <= 2 ** (num_objects + 1)):
        raise ValueError(
            f"max_num_steps must satisfy 2^num_objects <= max_num_steps <= 2^(num_objects+1), "
            f"got max_num_steps={max_num_steps}, num_objects={num_objects} "
            f"(valid range: [{2**num_objects}, {2**(num_objects + 1)}])"
        )
    if rule_type is not None and rule_type not in ("disjunctive", "conjunctive"):
        raise ValueError(
            f"rule_type must be 'disjunctive', 'conjunctive', or None, got '{rule_type}'"
        )

    # Build dataset
    initial_message = build_initial_message(num_objects)
    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": initial_message}],
            "info": json.dumps({
                "num_objects": num_objects,
                "num_blickets": num_blickets,
                "max_num_steps": max_num_steps,
                "rule_type": rule_type,
            }),
        }
        for _ in range(num_examples)
    ])

    # Build parser (shared between env and rubric)
    parser = vf.XMLParser(fields=["reasoning", "action"], answer_field="action")

    # Build rubric
    rubric = vf.Rubric(funcs=[blicket_identification], weights=[1.0], parser=parser)
    rubric.add_metric(step_budget_utilization)
    rubric.add_metric(exploration_inefficiency)
    rubric.add_metric(format_compliance)
    rubric.add_metric(hypotheses_eliminated)

    # Build system prompt
    system_prompt = build_system_prompt(num_objects)

    # max_turns = max_num_steps + 2 (exploration + transition + answer)
    max_turns = max_num_steps + 2

    return BlicketEnv(
        num_objects=num_objects,
        num_blickets=num_blickets,
        max_num_steps=max_num_steps,
        rule_type=rule_type,
        seed=seed,
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
        max_turns=max_turns,
    )
