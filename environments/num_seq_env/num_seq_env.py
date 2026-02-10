import random

import verifiers as vf
from datasets import Dataset


SYSTEM_PROMPT = (
    "You are given 5 consecutive terms of a numeric sequence governed by a "
    "linear recurrence relation of order 2 (each term is a fixed linear "
    "combination of the two preceding terms). Your task is to determine the "
    "requested future term.\n\n"
    "Think step-by-step about what recurrence relation generates the sequence, "
    "then compute the answer.\n\n"
    "Respond using the following format:\n"
    "<reasoning>\n...\n</reasoning>\n"
    "<answer>\n...\n</answer>\n\n"
    "The <answer> tag must contain only the integer value, nothing else."
)


def _generate_dataset(num_examples: int = 500, seed: int = 42) -> Dataset:
    """Generate a dataset of order-2 linear recurrence sequence problems."""
    rng = random.Random(seed)
    coeff_range = range(-3, 4)  # -3 to 3 inclusive
    init_range = range(-10, 11)  # -10 to 10 inclusive
    n_choices = [1, 2, 3]
    max_abs_value = 10_000

    examples: list[dict] = []
    seen: set[tuple] = set()

    while len(examples) < num_examples:
        c1 = rng.choice(coeff_range)
        c2 = rng.choice(coeff_range)
        if c1 == 0 and c2 == 0:
            continue

        a0 = rng.choice(init_range)
        a1 = rng.choice(init_range)
        n = rng.choice(n_choices)

        # Deduplicate on the full parameter tuple
        key = (c1, c2, a0, a1, n)
        if key in seen:
            continue
        seen.add(key)

        # Build the sequence: 5 shown terms + n extra terms
        seq = [a0, a1]
        overflow = False
        for _ in range(3 + n):
            next_val = c1 * seq[-1] + c2 * seq[-2]
            if abs(next_val) > max_abs_value:
                overflow = True
                break
            seq.append(next_val)

        if overflow or len(seq) < 5 + n:
            continue

        shown = seq[:5]
        answer = seq[5 + n - 1]  # the n-th term after the shown 5

        prompt_text = (
            f"Here are 5 consecutive terms of a sequence:\n"
            f"{shown[0]}, {shown[1]}, {shown[2]}, {shown[3]}, {shown[4]}\n\n"
            f"What is the next term (term number {n} after the last shown term)?"
            if n == 1
            else (
                f"Here are 5 consecutive terms of a sequence:\n"
                f"{shown[0]}, {shown[1]}, {shown[2]}, {shown[3]}, {shown[4]}\n\n"
                f"What is term number {n} after the last shown term?"
            )
        )

        examples.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": str(answer),
            }
        )

    return Dataset.from_list(examples)


def load_environment(num_examples: int = 500, seed: int = 42) -> vf.Environment:
    """Load the numeric sequence inductive reasoning environment."""
    dataset = _generate_dataset(num_examples=num_examples, seed=seed)

    parser = vf.XMLParser(["reasoning", "answer"])

    async def exact_match(completion, answer, parser) -> float:
        predicted = parser.parse_answer(completion)
        if predicted is None:
            return 0.0
        return 1.0 if predicted.strip() == answer.strip() else 0.0

    rubric = vf.Rubric(funcs=[exact_match], parser=parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=SYSTEM_PROMPT,
    )
