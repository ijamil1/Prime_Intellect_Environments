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


def _generate_dataset(
    num_examples: int = 500, seed: int = 42, max_start_idx: int = 24
) -> Dataset:
    """Generate a dataset of order-2 linear recurrence sequence problems."""
    rng = random.Random(seed)

    # Biased coefficient sampling: positives 3x more likely, zero excluded
    positives = [1, 2, 3, 4, 5]
    negatives = [-1, -2, -3, -4, -5]
    coeff_pool = positives * 3 + negatives

    init_range = range(-4, 5)  # -5 to 5 inclusive
    n_choices = range(1, 11)
    max_abs_value = 10_000

    examples: list[dict] = []
    seen: set[tuple] = set()

    while len(examples) < num_examples:
        c1 = rng.choice(coeff_pool)
        c2 = rng.choice(coeff_pool)

        a0 = rng.choice(init_range)
        a1 = rng.choice(init_range)
        start_idx = rng.randint(1, max_start_idx)
        n = rng.choice(n_choices)

        # Deduplicate on the full parameter tuple
        key = (c1, c2, a0, a1, start_idx, n)
        if key in seen:
            continue
        seen.add(key)

        # Build the sequence from initial seeds up to the needed length
        offset = start_idx - 1
        total_needed = offset + 5 + n
        seq = [a0, a1]
        overflow = False
        for _ in range(total_needed - 2):
            next_val = c1 * seq[-1] + c2 * seq[-2]
            if abs(next_val) > max_abs_value:
                overflow = True
                break
            seq.append(next_val)

        if overflow or len(seq) < offset + 5 + n:
            continue

        shown = seq[offset : offset + 5]
        answer = seq[offset + 5 + n - 1]  # the n-th term after the shown 5

        # Identifiability check: Hankel determinant must be non-zero
        # so the recurrence coefficients are uniquely determined
        det = shown[1] ** 2 - shown[2] * shown[0]
        if det == 0:
            continue

        prompt_text = (
            f"Here are 5 consecutive terms of a sequence:\n"
            f"{shown[0]}, {shown[1]}, {shown[2]}, {shown[3]}, {shown[4]}\n\n"
            f"What is the next term in the sequence?"
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
