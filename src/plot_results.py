from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Output directory
# ============================================================

FIGURE_DIR = Path("./results/figures")
FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Verified NegGrad+ sensitivity results
# ============================================================

steps = [100, 150, 200, 500]

forget_corrupted_accuracy = [
    34.40,
    23.72,
    18.68,
    11.26
]

retain_accuracy = [
    94.14,
    91.24,
    88.74,
    83.66
]

# ============================================================
# Forgetting-utility trade-off figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 4.2)
)

ax.plot(
    steps,
    forget_corrupted_accuracy,
    marker="o",
    linewidth=2,
    label="Corrupted-label accuracy (Forget set)"
)

ax.plot(
    steps,
    retain_accuracy,
    marker="s",
    linewidth=2,
    label="Retain-set accuracy"
)

ax.set_xlabel("NegGrad+ Unlearning Steps")
ax.set_ylabel("Accuracy (%)")

ax.set_xticks(steps)

ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

ax.legend(
    loc="lower center",
    fontsize=8,
    frameon=True
)

fig.tight_layout()

output_path = (
    FIGURE_DIR /
    "neggrad_forgetting_utility_tradeoff.pdf"
)

fig.savefig(
    output_path,
    bbox_inches="tight"
)
preview_path = (
    FIGURE_DIR /
    "neggrad_forgetting_utility_tradeoff.png"
)

fig.savefig(
    preview_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Preview saved to: {preview_path}")

plt.close(fig)

print(f"Figure saved to: {output_path}")