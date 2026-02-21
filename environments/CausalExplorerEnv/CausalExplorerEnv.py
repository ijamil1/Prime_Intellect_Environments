import hashlib
import json
import math
import re
from itertools import product

import numpy as np
import verifiers as vf
from datasets import Dataset

MAX_ANSWER_ATTEMPTS = 3


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


def parse_response(content: str, phase: str, num_objects: int) -> dict | None:
    """Extract and parse the agent's action with strict tag validation.

    Rules:
    - Strips all <reasoning>...</reasoning> blocks first.
    - Requires exactly one <action>...</action> tag in the remaining text.
    - Branches by phase: calls parse_action (exploration) or parse_predictions (answer).

    Returns a parsed dict or None if any rule is violated or parsing fails.
    """
    stripped = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
    matches = re.findall(r'<action>\s*(.*?)\s*</action>', stripped, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    action_str = matches[0].strip()
    if phase == "exploration":
        return parse_action(action_str)
    else:
        return parse_predictions(action_str, num_objects)


def build_system_prompt(num_objects: int, max_num_steps: int) -> str:
    """Build the system prompt for the Blicket game."""
    object_list = ", ".join(str(i) for i in range(1, num_objects + 1))
    return f"""\

You are an intelligent, curious agent. You are playing a game where you are in a room with \
{num_objects} different objects, and a machine. The objects are labeled as such: {object_list}. Some of these objects are blickets. \
You can't tell which object is a blicket just by looking at it. \
Blickets make the machine turn on following some hidden rule that may require all, some, or any of the blickets being on the machine.

To be precise, a blicket is defined as an object whose state is not independent of the
state of the machine (in other words, the object's state (on/off the machine) distribution and the machine's distribution have nonzero mutual
information)

Your goal is to determine exactly which objects are Blickets through exploration.
You have a maximum of {max_num_steps} steps to conduct the exploration phase so you must act efficiently. You can also exit this phase early if you think you understand the relationship between the
objects and the machine. After the exploration phase is done, you will be asked which objects are blickets.

RULES:
- In each action, you can place exactly one object onto the machine or remove exactly one object off the machine.
- After each action, you will observe which objects are on the machine and whether the machine is ON or OFF.
- When you have gathered enough information to determine which objects are Blickets, you can exit the exploration phase to submit your answer.

RESPONSE FORMAT (strict — violations count as wasted steps):

Every response must contain exactly one <reasoning> block followed by exactly one <action> block. No other XML tags are permitted. The <action> tag must not appear inside the <reasoning> block.

During exploration, your response must be one of these three forms:

  Place an object on the machine:
    <reasoning>Your reasoning here.</reasoning>
    <action>put N on</action>

  Remove an object from the machine:
    <reasoning>Your reasoning here.</reasoning>
    <action>put N off</action>

  Exit exploration and move to the answer phase:
    <reasoning>Your reasoning for stopping here.</reasoning>
    <action>exit</action>

Where N is a single integer in ({object_list}). Do not include any text outside the XML tags.

During the answer phase, your response must be exactly:
    <reasoning>Your analysis of which objects are Blickets.</reasoning>
    <action>1: True, 2: False, 3: True, ...</action>

You must include a True or False prediction for every object ({object_list}), separated by commas, in ascending order by object number. True means the object is a Blicket; False means it is not.

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
        state["exploration_and_answer_count"] = 0
        state["phase"] = "exploration"
        state["history"] = []
        state["num_objects"] = num_objects
        state["max_num_steps"] = max_num_steps
        state["valid_action_count"] = 0
        state["parseable_action_count"] = 0
        state["total_action_count"] = 0
        state["redundant_action_count"] = 0
        state["out_of_range_count"] = 0
        state["answer_attempt_count"] = 0
        state["max_answer_attempts"] = MAX_ANSWER_ATTEMPTS
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
        # Extract the most recent assistant message
        content = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = str(msg["content"])
                break

        state["exploration_and_answer_count"] += 1

        # --- Answer phase: agent has submitted predictions ---
        if state["phase"] == "answer":
            state["answer_attempt_count"] += 1
            predictions = parse_response(content, "answer", state["num_objects"])

            if predictions is None:
                # Bad format — retry or exhaust budget
                if state["answer_attempt_count"] < state["max_answer_attempts"]:
                    object_list = ", ".join(str(i) for i in range(1, state["num_objects"] + 1))
                    example = ", ".join(f"{i}: True" if i % 2 == 1 else f"{i}: False" for i in range(1, state["num_objects"] + 1))
                    retry_msg = (
                        f"Could not parse your answer "
                        f"(attempt {state['answer_attempt_count']}/{state['max_answer_attempts']}). "
                        f"You have {state['max_answer_attempts'] - state['answer_attempt_count']} attempt(s) remaining.\n\n"
                        f"Your response must contain exactly one <reasoning> block and exactly one <action> block. "
                        f"The <action> block must NOT appear inside <reasoning>. "
                        f"The action must list every object ({object_list}) with a True or False prediction, "
                        f"comma-separated, in ascending order.\n\n"
                        f"Example of correct format:\n"
                        f"<reasoning>Your analysis of which objects are Blickets.</reasoning>\n"
                        f"<action>{example}</action>"
                    )
                    return [{"role": "user", "content": retry_msg}]
                else:
                    blicket_list = [str(i + 1) for i in range(state["num_objects"]) if state["blickets"][i] == 1]
                    final_msg = (
                        f"Maximum answer attempts reached. No valid answer recorded.\n"
                        f"The Blickets were: [{', '.join(blicket_list)}]\n"
                        f"The rule was: {state['rule_type']}"
                    )
                    state["final_env_response"] = [{"role": "user", "content": final_msg}]
                    return state["final_env_response"]

            # Valid predictions — score and exit
            state["parseable_action_count"] += 1
            correct = sum(
                1 for i in range(state["num_objects"])
                if predictions.get(i + 1) == bool(state["blickets"][i])
            )
            score = correct / state["num_objects"]
            state["final_score"] = score
            blicket_list = [str(i + 1) for i in range(state["num_objects"]) if state["blickets"][i] == 1]
            final_msg = (
                f"Your answer has been recorded. "
                f"You correctly identified {correct}/{state['num_objects']} objects. "
                f"Score: {score:.2f}\n"
                f"The Blickets were: [{', '.join(blicket_list)}]\n"
                f"The rule was: {state['rule_type']}"
            )
            state["final_env_response"] = [{"role": "user", "content": final_msg}]
            return state["final_env_response"]

        # --- Exploration phase ---
        state["total_action_count"] += 1
        action = parse_response(content, "exploration", state["num_objects"])

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
                f"Invalid action format. Every response must contain exactly one <reasoning> block followed by exactly one <action> block. No other XML tags are permitted. The <action> tag must not appear inside the <reasoning> block. Expected one of: <action>put N on</action>, "
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
            state["out_of_range_count"] += 1
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
            f"You MUST respond using XML tags. For example:\n"
            f"<reasoning>Your analysis of which objects are Blickets...</reasoning>\n"
            f"<action>1: True, 2: False, ...</action>"
        )
        return [{"role": "user", "content": msg}]


# --- Reward and metric functions ---


async def blicket_identification(state) -> float:
    """Reward: per-object accuracy of Blicket identification.

    Reads state["final_score"], which is set by env_response when the agent
    submits a correctly-formatted answer within the allowed retry budget.
    Returns 0.0 if no valid answer was recorded (exhausted retries or no answer).
    """
    return state.get("final_score", 0.0)


async def step_budget_utilization(state) -> float:
    """Reward: step-budget utilization, conditioned on identification accuracy.

    Behavior is split by whether the agent achieved perfect identification:
    - Imperfect (final_score < 1.0): returns step_count / max_steps, rewarding
      more exploration (the agent is encouraged to use its full budget).
    - Perfect (final_score == 1.0): returns 1.0 unconditionally, since the agent
      fully solved the task regardless of how many steps it took.
    """
    # measures utilization as fraction of exploration steps used relative to max num steps (unless we were completely accurate in which case we return a non-differentiable? 1)
    final_score = state.get("final_score", 0.0)
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_num_steps", 1)
    if final_score != 1.0:
        #if we didn't correctly identify blickets, we want to encourage more exploration by rewarding more steps
        return step_count/max_steps
    return 1.0


async def exploration_efficiency(state) -> float:
    """Reward: fraction of parseable actions that were productive.

    Counts three disjoint sources of waste relative to parseable exploration actions:
    1. Redundant actions — no-ops where the object was already in the requested
       state (not added to history).
    2. Out-of-range object IDs — syntactically valid toggles referencing an
       object number outside [1, num_objects].
    3. Non-contiguous revisits — valid toggles that reproduce a machine
       configuration already seen earlier in the history.

    Returns 1 - (wasted / parseable_action_count), a value in [0, 1].
    Higher is better (1.0 = no wasted actions). Returns 0.0 when no parseable
    actions were taken (avoids division by zero).

    Note: parseable_action_count includes the successful answer submission,
    so a perfect episode with no exploration waste scores 1.0.
    """
    #measures efficiency as the fraction of parseable actions that were to unique object state

    redundant = state.get("redundant_action_count", 0)
    parseable = state.get("parseable_action_count", 0) #parseable ~=~ out of range + redundant + non_contiguous revisit + unique
    out_of_range = state.get("out_of_range_count", 0)

    if parseable == 0:
        return 0

    # Count non-contiguous revisits from history
    history = state.get("history", [])
    seen = set()
    non_contiguous = 0
    for entry in history:
        config = frozenset(entry["on_objects"])
        if config in seen:
            non_contiguous += 1
        seen.add(config)

    return 1 - (float(redundant + out_of_range + non_contiguous) / parseable)


async def format_compliance(state) -> float:
    """Reward: fraction of all turns (exploration + answer) with parseable actions.

    Denominator is exploration_and_answer_count — every env_response call across
    both phases. Numerator is parseable_action_count, which includes: parseable
    exploration actions (exit, valid toggles, out-of-range toggles, redundant toggles)
    plus the single successful answer submission.

    Returns parseable / total, in [0, 1]. Higher is better. Returns 1.0 when
    no turns were taken (vacuously compliant).
    """
    total = state.get("exploration_and_answer_count", 0)
    if total == 0:
        return 1.0
    parseable = state.get("parseable_action_count", 0)
    return parseable / total


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


# --- Normalized rubric ---


class NormalizedRubric(vf.Rubric):
    """Rubric that normalizes advantages by std within each group after scoring.

    The base Rubric only subtracts the group mean. When rewards cluster in a
    narrow range the resulting advantages are tiny (~±0.05), producing near-zero
    gradients. Dividing by the group std makes advantages unit-scale regardless
    of how spread the rewards are, giving the trainer a consistent signal.
    """

    async def score_group(self, states, *args, **kwargs):
        await super().score_group(states, *args, **kwargs)
        advantages = [s["advantage"] for s in states]
        # advantages are already mean-subtracted, so std = RMS
        std = (sum(a ** 2 for a in advantages) / len(advantages)) ** 0.5
        for s in states:
            s["advantage"] = s["advantage"] / (std + 1e-8)
            for t in s["trajectory"]:
                if t["advantage"] is not None:
                    t["advantage"] = t["advantage"] / (std + 1e-8)


# --- Entry point ---

# --- Dataset helpers ---


def sample_unique_configs(
    num_objects_range: tuple[int, int],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Sample n unique (n_obj, rule, blickets) configs via rejection sampling.

    Uses a master RNG seeded with `seed` to draw configs; duplicates (same
    n_obj, rule, and blicket_indices) are rejected until n distinct configs
    are collected.
    """
    rng = np.random.default_rng(seed)
    lo, hi = num_objects_range
    seen: set[tuple] = set()
    configs = []
    while len(configs) < n:
        n_obj = int(rng.integers(lo, hi + 1))
        max_b = n_obj // 2
        b = int(rng.integers(2, max_b + 1))
        rule = str(rng.choice(["disjunctive", "conjunctive"]))
        blicket_indices = tuple(sorted(
            rng.choice(n_obj, size=b, replace=False).tolist()
        ))
        key = (n_obj, rule, blicket_indices)
        if key not in seen:
            seen.add(key)
            blickets = [0] * n_obj
            for idx in blicket_indices:
                blickets[idx] = 1
            configs.append({
                "n_obj": n_obj,
                "rule": rule,
                "blickets": blickets,
                "blicket_indices": list(blicket_indices),
            })
    return configs


def build_rows(configs: list[dict]) -> tuple[list[dict], int]:
    """Build dataset rows from a list of configs, returning rows and global max_steps.

    Each row's compute_optimal_steps seed is derived deterministically from the
    config contents via MD5, so seeds are stable regardless of list ordering.
    """
    rows = []
    global_max_steps = 0
    for cfg in configs:
        seed_str = f"{cfg['n_obj']}_{cfg['rule']}_{cfg['blicket_indices']}"
        row_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        optimal_steps, hyps_elim = compute_optimal_steps(
            cfg["n_obj"], cfg["blickets"], cfg["rule"], seed=row_seed
        )
        max_steps = max(1, math.ceil(1.5 * optimal_steps))
        global_max_steps = max(global_max_steps, max_steps)

        system_msg = {"role": "system", "content": build_system_prompt(cfg["n_obj"], max_steps)}
        user_msg = {"role": "user", "content": build_initial_message(cfg["n_obj"])}

        rows.append({
            "prompt": [system_msg, user_msg],
            "info": json.dumps({
                "num_objects": cfg["n_obj"],
                "num_blickets": len(cfg["blicket_indices"]),
                "max_num_steps": max_steps,
                "rule_type": cfg["rule"],
                "blickets": cfg["blickets"],
                "optimal_hypotheses_eliminated": hyps_elim,
            }),
        })
    return rows, global_max_steps


# --- Entry point ---

def load_environment(num_examples: int = 250) -> vf.Environment:
    """Load the CausalExplorerEnv (Blicket machine) environment.

    Training and eval datasets are fixed and fully distinct from each other.
    Both are generated via rejection sampling to ensure uniqueness:
      - Training + eval part 1: (num_examples + 50) unique configs from n ∈ [4, 10],
        sampled with seed=42. First num_examples → training; last 50 → eval part 1
        (guaranteed non-overlapping for any valid num_examples).
      - Eval part 2: 50 unique configs from n ∈ [11, 15], sampled with seed=42
        (distinct from training by construction due to disjoint n range).

    Args:
        num_examples: Number of training examples (clamped to [100, 500]).
    """
    num_examples = max(100, min(num_examples, 500))

    # --- Training + eval part 1 (sampled together to guarantee no overlap) ---
    pool = sample_unique_configs((4, 10), num_examples + 50, seed=42)
    train_configs = pool[:num_examples]
    eval_configs_part1 = pool[num_examples:]

    train_rows, train_max_steps = build_rows(train_configs)
    dataset = Dataset.from_list(train_rows)

    # --- Eval part 2: different n range, automatically distinct from training ---
    eval_configs_part2 = sample_unique_configs((11, 15), 50, seed=42)

    eval_rows, eval_max_steps = build_rows(eval_configs_part1 + eval_configs_part2)
    eval_dataset = Dataset.from_list(eval_rows)

    # Build parser (shared between env and rubric)
    parser = vf.XMLParser(fields=["reasoning", "action"], answer_field="action")

    # Build rubric
    rubric = NormalizedRubric(
        funcs=[blicket_identification, step_budget_utilization, exploration_efficiency, format_compliance, hypotheses_eliminated],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
        parser=parser,
    )

    # max_turns derived from the maximum max_steps across both datasets,
    # so eval examples from [11,15] are fully accommodated
    global_max_steps = max(train_max_steps, eval_max_steps)
    max_turns = global_max_steps + 1 + MAX_ANSWER_ATTEMPTS

    return BlicketEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        parser=parser,
        max_turns=max_turns,
    )
