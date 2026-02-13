"""Rescore intellect-3 eval results using <answer> tags with \\boxed{} fallback."""

import json
import re
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "outputs" / "evals" / "num-seq-env--prime-intellect--intellect-3" / "a60a8e36" / "results.jsonl"

answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
boxed_pattern = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)

records = []
with open(RESULTS_FILE) as f:
    for line in f:
        records.append(json.loads(line))
records.sort(key=lambda r: r["example_id"])

correct = 0
parsed_via_xml = 0
parsed_via_boxed = 0
unparseable = 0

for rec in records:
    completion_text = "\n".join(
        msg["content"] for msg in rec["completion"] if msg.get("content") is not None
    )
    answer = rec["answer"]

    # Try <answer> tags first
    match = answer_pattern.search(completion_text)
    if match:
        predicted = match.group(1).strip()
        parsed_via_xml += 1
    else:
        # Fallback to \boxed{}
        match = boxed_pattern.search(completion_text)
        if match:
            predicted = match.group(1).strip()
            parsed_via_boxed += 1
        else:
            predicted = None
            unparseable += 1

    if predicted is not None and predicted == answer.strip():
        correct += 1

total = len(records)
print(f"Total examples: {total}")
print(f"Parsed via <answer>: {parsed_via_xml}")
print(f"Parsed via \\boxed{{}}: {parsed_via_boxed}")
print(f"Unparseable: {unparseable}")
print(f"\nOriginal accuracy (xml only): {sum(r['reward'] for r in records) / total:.2f}")
print(f"Rescored accuracy (xml + boxed): {correct / total:.2f}")
