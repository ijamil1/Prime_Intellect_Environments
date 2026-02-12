import random

import numpy as np
import verifiers as vf
from datasets import Dataset


def _det(matrix: list[list[int]]) -> int:
    """Exact integer determinant via cofactor expansion (fine for n <= 5)."""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    result = 0
    for col in range(n):
        minor = [
            [matrix[r][c] for c in range(n) if c != col] for r in range(1, n)
        ]
        sign = 1 if col % 2 == 0 else -1
        result += sign * matrix[0][col] * _det(minor)
    return result


def _hankel_det(seq: list[int], k: int) -> int:
    """Determinant of the k x k Hankel matrix H[i][j] = seq[i+j].

    Non-zero guarantees the sequence is genuinely order k.
    """
    matrix = [[seq[i + j] for j in range(k)] for i in range(k)]
    return _det(matrix)


def _has_unit_roots(coeffs: list[int]) -> bool:
    """Check if the characteristic polynomial has any roots with |root| = 1.

    For integer coefficients, roots on the unit circle are roots of unity
    (Kronecker's theorem), so the sequence is periodic.
    """
    char_poly = [1] + [-c for c in coeffs]
    roots = np.roots(char_poly)
    return any(abs(abs(r) - 1) < 1e-6 for r in roots)



SYSTEM_PROMPT = (
    "You are a mathematician who is given consecutive terms of a numeric sequence governed by a "
    "linear recurrence relation. You are told which positions in the "
    "sequence the shown terms occupy. Your task is to calculate the "
    "value of a requested term in the sequence.\n\n"
    "Think step-by-step about what recurrence relation generates the sequence, "
    "then compute the answer.\n\n"
    "Respond using the following format:\n"
    "<reasoning>\n...\n</reasoning>\n"
    "<answer>\n...\n</answer>\n\n"
    "The <answer> tag must contain only the integer value, nothing else."
)

def _generate_dataset(
    num_examples: int = 500,
    seed: int = 42,
    max_start_idx: int = 24,
    min_k: int = 2,
    max_k: int = 5,
) -> Dataset:
    """Generate a dataset of variable-order linear recurrence sequence problems."""
    rng = random.Random(seed)

    # Biased coefficient sampling: positives 3x more likely, zero excluded
    positives = [1, 2, 3, 4, 5]
    negatives = [-1, -2, -3, -4, -5]
    coeff_pool = positives + negatives

    init_range = range(-4, 5)  # -4 to 4 inclusive
    max_abs_value = 100_000
    max_lookahead = 10
    max_num_shown = 2 * max_k + 1  # show the same count for all k by default

    examples: list[dict] = []
    seen: set[tuple] = set()

    while len(examples) < num_examples:
        # Sample recurrence order uniformly
        k = rng.randint(min_k, max_k)

        # Sample k coefficients; coeff_pool excludes zero so coeffs[-1] != 0
        coeffs = [rng.choice(coeff_pool) for _ in range(k)]

        # Sample k initial values
        inits = [rng.choice(init_range) for _ in range(k)]

        start_idx = rng.randint(1, max_start_idx)

        # Build sequence long enough for max shown + max forward lookahead
        offset = start_idx - 1
        total_needed = offset + max_num_shown + max_lookahead
        seq = list(inits)
        overflow = False
        for _ in range(total_needed - k):
            next_val = sum(coeffs[i] * seq[-(i + 1)] for i in range(k))
            if abs(next_val) > max_abs_value:
                overflow = True
                break
            seq.append(next_val)

        if overflow or len(seq) < total_needed:
            continue

        full_shown = seq[offset : offset + max_num_shown]

        # Identifiability: k x k Hankel determinant must be non-zero
        if _hankel_det(full_shown, k) == 0:
            continue

        # Reject periodic sequences: if the characteristic polynomial has
        # roots on the unit circle (roots of unity), the sequence is periodic
        # and the model could exploit repeating patterns.
        if _has_unit_roots(coeffs):
            continue

        num_shown = max_num_shown
        shown = full_shown

        # Valid target positions (1-indexed): before and after the shown window
        first_shown = start_idx
        last_shown = start_idx + num_shown - 1
        backward = list(range(max(1, first_shown - max_lookahead), first_shown))
        forward = list(range(last_shown + 1, last_shown + max_lookahead + 1))
        target_pos = rng.choice(backward + forward)

        # Deduplicate on the full parameter tuple
        key = (tuple(coeffs), tuple(inits), start_idx, target_pos)
        if key in seen:
            continue
        seen.add(key)

        answer = seq[target_pos - 1]

        terms_str = ", ".join(str(t) for t in shown)
        prompt_text = (
            f"Here are terms {first_shown} through {last_shown} of a sequence:\n"
            f"{terms_str}\n\n"
            f"What is term {target_pos} of the sequence?"
        )

        examples.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": str(answer),
            }
        )

    return Dataset.from_list(examples)


def load_environment(
    num_examples: int = 500,
    seed: int = 42,
    min_k: int = 2,
    max_k: int = 5,
) -> vf.Environment:
    """Load the numeric sequence inductive reasoning environment."""
    dataset = _generate_dataset(
        num_examples=num_examples, seed=seed, min_k=min_k, max_k=max_k
    )

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
