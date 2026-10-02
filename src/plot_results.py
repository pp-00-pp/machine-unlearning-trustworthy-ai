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

# ============================================================
# Original vs. Unlearned vs. Retrained comparison
# ============================================================

model_names = [
    "Original",
    "Unlearned",
    "Retrained"
]

forget_true = [
    10.68,
    52.96,
    72.46
]

forget_corrupted = [
    84.00,
    23.72,
    3.32
]

retain_acc = [
    97.84,
    91.24,
    97.88
]

test_acc = [
    64.76,
    67.93,
    71.34
]

# ============================================================
# Grouped comparison figure
# ============================================================

import numpy as np

x = np.arange(len(model_names))
width = 0.20

fig, ax = plt.subplots(
    figsize=(7.0, 4.4)
)

ax.bar(
    x - 1.5 * width,
    forget_true,
    width,
    label="Forget: true-label accuracy"
)

ax.bar(
    x - 0.5 * width,
    forget_corrupted,
    width,
    label="Forget: corrupted-label accuracy"
)

ax.bar(
    x + 0.5 * width,
    retain_acc,
    width,
    label="Retain-set accuracy"
)

ax.bar(
    x + 1.5 * width,
    test_acc,
    width,
    label="Clean test accuracy"
)

ax.set_ylabel("Accuracy (%)")

ax.set_xticks(x)
ax.set_xticklabels(model_names)

ax.set_ylim(0, 105)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.legend(
    fontsize=8,
    ncol=2,
    loc="upper center"
)

fig.tight_layout()

comparison_pdf = (
    FIGURE_DIR /
    "model_performance_comparison.pdf"
)

comparison_png = (
    FIGURE_DIR /
    "model_performance_comparison.png"
)

fig.savefig(
    comparison_pdf,
    bbox_inches="tight"
)

fig.savefig(
    comparison_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"Comparison figure saved to: "
    f"{comparison_pdf}"
)

print(
    f"Comparison preview saved to: "
    f"{comparison_png}"
)