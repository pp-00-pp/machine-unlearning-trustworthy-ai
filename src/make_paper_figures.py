"""Generate single-seed paper figures from experiment JSONs, without training."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


KEYS = ["forget_true_label_accuracy", "forget_corrupted_label_accuracy",
        "retain_accuracy", "clean_test_accuracy"]
STEPS = [100, 150, 200, 500]
LABELS = ["Forget: true", "Forget: corrupted", "Retain", "Test"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.results_dir.resolve()
    sources = {}

    def read(name):
        path = root / name
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        sources[name] = {"absolute_path": str(path),
                         "sha256": hashlib.sha256(raw).hexdigest(),
                         "data": data}
        return data

    def number(data, key, context, lower=0, upper=None):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context}: {key} must be numeric")
        if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
            raise ValueError(f"{context}: invalid {key}={value}")
        return value

    evaluations = {}
    filenames = {"Original": "original_pre_unlearning_evaluation.json",
                 "Retrained": "retrained_reference_evaluation.json"}
    filenames.update({f"NegGrad+ {s}": f"neggrad_plus_{s}steps_evaluation.json" for s in STEPS})
    counts = None
    for model, filename in filenames.items():
        data = read(filename)
        evaluations[model] = [number(data, k, filename, upper=100) for k in KEYS]
        current = tuple(data[k] for k in ["forget_samples", "retain_samples", "test_samples"])
        if counts is None:
            counts = current
        if current != counts:
            raise ValueError(f"Inconsistent split sizes in {filename}")

    unl = read("neggrad_plus_150steps_unlearning_results.json")
    ret = read("retrained_reference_results.json")
    if unl["unlearning_steps"] != 150:
        raise ValueError("Selected unlearning JSON does not describe 150 steps")
    if unl["seed"] != ret["seed"]:
        raise ValueError("Unlearning and retraining seeds differ")
    seed = unl["seed"]
    times = [number(unl, "unlearning_time_seconds", "unlearning"),
             number(ret, "training_time_seconds", "retraining")]
    if min(times) <= 0:
        raise ValueError("Both elapsed times must be positive")
    speedup = times[1] / times[0]
    behavior = read("model_behavior_comparison.json")
    if behavior["unlearned_model"] != "neggrad_plus_150steps":
        raise ValueError("Behavior comparison does not identify the 150-step model")
    splits = ["forget", "retain", "test"]
    similarity = {}
    for metric in ["prediction_agreement_percent", "mean_js_divergence"]:
        similarity[metric] = {}
        for name in ["original_vs_retrained", "unlearned_vs_retrained"]:
            similarity[metric][name] = [
                number(behavior[split][name], metric, f"{split}/{name}",
                       upper=100 if metric == "prediction_agreement_percent" else None)
                for split in splits]
    for split, count in zip(splits, counts):
        if behavior[split]["num_samples"] != count:
            raise ValueError(f"Behavior/evaluation sample-count mismatch: {split}")

    print(f"Validated {len(sources)} JSON files; recorded seed: {seed}.")
    print("All figure values come from those JSON files; no result values are hard-coded.")
    print(f"Computed recorded-time speedup: {speedup!r}")
    if args.validate_only:
        print("Validation complete. No figures or experiment files were changed.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, name):
        for ext in ["png", "pdf"]:
            path = out / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(str(path))
            print(f"Saved: {path}")
        plt.close(fig)

    def tidy(ax):
        ax.grid(axis="y", linestyle="--", alpha=.25)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 7.0), layout="constrained")
    x = np.arange(len(KEYS))
    for i, model in enumerate(["Original", "NegGrad+ 150", "Retrained"]):
        axes[0].bar(x + (i - 1) * .25, evaluations[model], width=.24,
                    label=model, color=COLORS[i])
    axes[0].set_xticks(x, ["Forget\ntrue", "Forget\ncorrupted", "Retain", "Test"])
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("(a) Original, unlearned, and retrained models", pad=44)
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    for j, (label, color, marker) in enumerate(zip(LABELS, COLORS, ["o", "s", "^", "D"])):
        axes[1].plot(STEPS, [evaluations[f"NegGrad+ {s}"][j] for s in STEPS],
                     marker=marker, color=color, linewidth=1.8, label=label)
    axes[1].axvline(150, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xticks(STEPS)
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("NegGrad+ update steps")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("(b) Effect of the update budget", pad=48)
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    for ax in axes:
        tidy(ax)
    save(fig, "figure_1_accuracy_tradeoff")

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.5), layout="constrained")
    for ax, metric, title, ylabel in zip(axes,
            ["prediction_agreement_percent", "mean_js_divergence"],
            ["(a) Prediction agreement with retraining", "(b) Jensen–Shannon divergence from retraining"],
            ["Prediction agreement (%)", "Mean JS divergence"]):
        x = np.arange(3)
        for i, (key, label) in enumerate(zip(
                ["original_vs_retrained", "unlearned_vs_retrained"], ["Original", "NegGrad+ 150"])):
            ax.bar(x + (i - .5) * .34, similarity[metric][key], width=.32,
                   label=label, color=COLORS[i])
        ax.set_xticks(x, ["Forget", "Retain", "Test"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=40)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
        tidy(ax)
    axes[0].set_ylim(0, 105)
    save(fig, "figure_2_behavioral_similarity")

    fig, ax = plt.subplots(figsize=(5.6, 2.9), layout="constrained")
    bars = ax.barh(["NegGrad+\n150 steps", "Retain-only\nretraining"], times,
                   color=[COLORS[1], COLORS[2]], height=.55)
    ax.invert_yaxis()
    ax.set_xlim(0, max(times) * 1.23)
    ax.set_xlabel("Recorded elapsed time (s)")
    ax.set_title(f"Recorded-time speedup: {speedup:.2f}×")
    for bar, value in zip(bars, times):
        ax.text(value + max(times) * .02, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} s", va="center")
    ax.grid(axis="x", linestyle="--", alpha=.25)
    ax.set_axisbelow(True)
    save(fig, "figure_3_update_time")

    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "recorded_seed": seed, "single_seed": True,
                "sources": sources, "plotted_accuracies": evaluations,
                "plotted_similarity": similarity, "plotted_times_seconds": times,
                "computed_speedup": speedup, "outputs": outputs,
                "note": "Full source precision used in plots. Only visible time annotations are formatted to two decimals. Retraining timer includes per-epoch test evaluation. No repeated-seed uncertainty is estimated."}
    path = out / "paper_figures_provenance.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Source data and hashes saved: {path}")


if __name__ == "__main__":
    main()
