"""Load eval results across models, verify consistency, and extract completions."""

import json
import re
from pathlib import Path

EVALS_DIR = Path(__file__).parent / "outputs" / "evals"


def load_results() -> dict[str, list[dict]]:
    """Load results.jsonl from each model subfolder. Returns {model_name: [records]}."""
    model_results = {}
    for model_dir in sorted(EVALS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name_list = model_dir.name.split('--')
        model_name = model_name_list[1] + '/' + model_name_list[2]
        # Each model dir has one hash-named subfolder containing results.jsonl
        jsonl_files = list(model_dir.glob("*/results.jsonl"))
        if len(jsonl_files) != 1:
            raise FileNotFoundError(f"Expected 1 results.jsonl in {model_dir}, found {len(jsonl_files)}")
        records = []
        with open(jsonl_files[0]) as f:
            for line in f:
                records.append(json.loads(line))
        # Sort by example_id for consistent ordering
        records.sort(key=lambda r: r["example_id"])
        model_results[model_name] = records
    return model_results


def verify_consistency(model_results: dict[str, list[dict]]) -> None:
    """Verify that prompts and answers match across all models for each example_id."""
    models = list(model_results.keys())
    ref_model = models[0]
    ref_records = model_results[ref_model]

    # Check all models have the same number of examples
    for model in models:
        n = len(model_results[model])
        assert n == len(ref_records), f"{model} has {n} examples, expected {len(ref_records)}"

    # Check example_ids, prompts, and answers match
    mismatches = []
    for i, ref_rec in enumerate(ref_records):
        eid = ref_rec["example_id"]
        for model in models[1:]:
            rec = model_results[model][i]
            if rec["example_id"] != eid:
                mismatches.append(f"  example index {i}: {ref_model} has id={eid}, {model} has id={rec['example_id']}")
                continue
            if rec["prompt"] != ref_rec["prompt"]:
                mismatches.append(f"  example_id={eid}: prompt mismatch between {ref_model} and {model}")
            if rec["answer"] != ref_rec["answer"]:
                mismatches.append(f"  example_id={eid}: answer mismatch between {ref_model} and {model}")

    if mismatches:
        raise ValueError("Consistency check failed:\n" + "\n".join(mismatches))


def extract_completions(model_results: dict[str, list[dict]]) -> dict[str, dict[int, str]]:
    """Extract completion content per model per example_id.

    Returns {model_name: [list of completions]}.
    """
    completions = {}
    for model, records in model_results.items():
        completions[model] = []
        for rec in records:
            # completion is a list of message dicts; extract the assistant content
            comp_messages = rec["completion"]
            content = "\n".join(msg["content"] for msg in comp_messages if msg.get("content") is not None)
            completions[model].append(content)
    return completions


def extract_rewards(model_results: dict[str, list[dict]]) -> dict[str, dict[int, float]]:
    """Extract reward per model per example_id.

    Returns {model_name: [list of rewards]}.
    """
    rewards = {}
    for model, records in model_results.items():
        rewards[model] = [rec["reward"] for rec in records]
    return rewards


OUTPUT_FILE = Path(__file__).parent / "format_errors.txt"


if __name__ == "__main__":
    model_results = load_results()
    verify_consistency(model_results)

    completions = extract_completions(model_results)
    rewards = extract_rewards(model_results)
    format_errors = {m: [] for m in completions.keys()}

    # Count completions where <answer>...</answer> tags could not be parsed
    answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Loaded results for {len(model_results)} models: {', '.join(model_results.keys())}\n\n")

        f.write("Format failures (no <answer> tag parsed) per model:\n")
        for model in sorted(completions):
            unparseable = 0
            for completion in completions[model]:
                if not answer_pattern.search(completion):
                    unparseable += 1
                    format_errors[model].append(completion)
            total = len(completions[model])
            f.write(f"  {model}: {unparseable}/{total} completions could not be parsed\n")

        for model in sorted(format_errors):
            completions_with_format_errors = format_errors[model]
            if not len(completions_with_format_errors):
                continue
            empty_ct = sum(1 for c in completions_with_format_errors if c == "")
            non_empty_ct = len(completions_with_format_errors) - empty_ct
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Format errors for {model} ({len(completions_with_format_errors)} total: {empty_ct} empty, {non_empty_ct} non-empty)\n")
            f.write(f"{'=' * 80}\n")
            j = 0
            for completion in completions_with_format_errors:
                if completion == "":
                    continue
                j += 1
                f.write(f"\n--- [{model}] error {j}/{non_empty_ct} ---\n")
                f.write(completion + "\n")

        # Check how many intellect-3 non-empty format errors used \boxed{}
        intellect_key = "prime-intellect/intellect-3"
        if intellect_key in format_errors:
            non_empty = [c for c in format_errors[intellect_key] if c != ""]
            boxed_pattern = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)
            boxed_ct = sum(1 for c in non_empty if boxed_pattern.search(c))
            non_boxed_non_empty = [c for c in non_empty if not boxed_pattern.search(c)]
            f.write(f"\n{'=' * 80}\n")
            f.write(f"\\boxed analysis for {intellect_key}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"  Non-empty format errors: {len(non_empty)}\n")
            f.write(f"  Contained \\boxed{{}}: {boxed_ct}/{len(non_empty)}\n")
            f.write(f"  Neither <answer> nor \\boxed{{}}: {len(non_empty) - boxed_ct}/{len(non_empty)}\n")
            j = 0
            non_boxed_non_empty_ct = len(non_empty) - boxed_ct
            if  non_boxed_non_empty_ct:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Non-boxed non-empty format errors for {intellect_key}\n")
                f.write(f"{'=' * 80}\n")
                for completion in non_boxed_non_empty:
                    j += 1
                    f.write(f"\n--- [{intellect_key}] non-boxed non-empty format error {j}/{non_boxed_non_empty_ct} ---\n")
                    f.write(completion + "\n")
    print(f"Output written to {OUTPUT_FILE}")

