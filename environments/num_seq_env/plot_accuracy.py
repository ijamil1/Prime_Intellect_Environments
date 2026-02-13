"""Bar chart of test accuracy per model on num_seq_env."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

EVALS_DIR = Path(__file__).parent / "outputs" / "evals"
INTELLECT3_RESCORED = 0.73

# Load model names and avg_reward from metadata.json
models = []
accuracies = []
for model_dir in sorted(EVALS_DIR.iterdir()):
    if not model_dir.is_dir():
        continue
    parts = model_dir.name.split("--")
    model_name = parts[1] + "/" + parts[2]
    metadata_files = list(model_dir.glob("*/metadata.json"))
    with open(metadata_files[0]) as f:
        meta = json.load(f)
    models.append(model_name)
    accuracies.append(meta["avg_reward"])

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, accuracies, color="#4C72B0", zorder=3)

# Add semi-transparent rescored bar for intellect-3
intellect_idx = models.index("prime-intellect/intellect-3")
ax.bar(
    models[intellect_idx],
    INTELLECT3_RESCORED,
    color="#4C72B0",
    alpha=0.3,
    zorder=2,
    label=f"intellect-3 rescored (xml + \\boxed{{}}): {INTELLECT3_RESCORED:.0%}",
)

# Labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.0%}", ha="center", va="bottom", fontweight="bold")

ax.set_ylabel("Test Accuracy")
ax.set_title("num_seq_env — Test Accuracy by Model")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
ax.legend()

plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "outputs" / "accuracy_by_model.png", dpi=150)
plt.show()
print("Saved to outputs/accuracy_by_model.png")
