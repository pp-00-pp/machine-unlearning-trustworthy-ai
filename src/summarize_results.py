import json
from pathlib import Path


RESULT_DIR = Path("./results")

FILES = {
    "original": RESULT_DIR / "original_pre_unlearning_evaluation.json",
    "retrained": RESULT_DIR / "retrained_reference_evaluation.json",

    "neggrad_100": RESULT_DIR / "neggrad_plus_100steps_evaluation.json",
    "neggrad_150": RESULT_DIR / "neggrad_plus_150steps_evaluation.json",
    "neggrad_200": RESULT_DIR / "neggrad_plus_200steps_evaluation.json",
    "neggrad_500": RESULT_DIR / "neggrad_plus_500steps_evaluation.json",

    "unlearning_100": RESULT_DIR / "neggrad_plus_100steps_unlearning_results.json",
    "unlearning_150": RESULT_DIR / "neggrad_plus_150steps_unlearning_results.json",
    "unlearning_200": RESULT_DIR / "neggrad_plus_200steps_unlearning_results.json",
    "unlearning_500": RESULT_DIR / "neggrad_plus_500steps_unlearning_results.json",

    "retrained_training": RESULT_DIR / "retrained_reference_results.json",

    "behavior": RESULT_DIR / "model_behavior_comparison.json",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

    # ============================================================
# Load saved experimental results
# ============================================================

data = {
    name: load_json(path)
    for name, path in FILES.items()
}


original = data["original"]
retrained = data["retrained"]

neggrad_100 = data["neggrad_100"]
neggrad_150 = data["neggrad_150"]
neggrad_200 = data["neggrad_200"]
neggrad_500 = data["neggrad_500"]

unlearning_100 = data["unlearning_100"]
unlearning_150 = data["unlearning_150"]
unlearning_200 = data["unlearning_200"]
unlearning_500 = data["unlearning_500"]

retrained_training = data["retrained_training"]
behavior = data["behavior"]


print("All result files loaded successfully.")
print(f"Original model      : {original['model']}")
print(f"Retrained model     : {retrained['model']}")
print(f"Forget samples      : {original['forget_samples']}")
print(f"Retain samples      : {original['retain_samples']}")
print(f"Test samples        : {original['test_samples']}")


# ============================================================
# Main experimental comparison
# ============================================================

rows = [
    (
        "Original",
        original,
        None
    ),
    (
        "NegGrad+ (100)",
        neggrad_100,
        unlearning_100
    ),
    (
        "NegGrad+ (150)",
        neggrad_150,
        unlearning_150
    ),
    (
        "NegGrad+ (200)",
        neggrad_200,
        unlearning_200
    ),
    (
        "NegGrad+ (500)",
        neggrad_500,
        unlearning_500
    ),
    (
        "Retrained",
        retrained,
        None
    ),
]


print("\n" + "=" * 105)
print("FINAL EXPERIMENTAL COMPARISON")
print("=" * 105)

header = (
    f"{'Model':<20}"
    f"{'Forget True':>13}"
    f"{'Forget Corr.':>14}"
    f"{'Retain Acc.':>13}"
    f"{'Test Acc.':>12}"
    f"{'Time (s)':>12}"
)

print(header)
print("-" * 105)

for name, evaluation, timing in rows:

    if timing is not None:
        runtime = timing["unlearning_time_seconds"]
        runtime_text = f"{runtime:.2f}"
    elif name == "Retrained":
        runtime_text = (
            f"{retrained_training['training_time_seconds']:.2f}"
        )
    else:
        runtime_text = "--"

    print(
        f"{name:<20}"
        f"{evaluation['forget_true_label_accuracy']:>12.2f}%"
        f"{evaluation['forget_corrupted_label_accuracy']:>13.2f}%"
        f"{evaluation['retain_accuracy']:>12.2f}%"
        f"{evaluation['clean_test_accuracy']:>11.2f}%"
        f"{runtime_text:>12}"
    )

    # ============================================================
# Behavioral similarity to retrained reference
# ============================================================

print("\n" + "=" * 90)
print("BEHAVIORAL SIMILARITY TO RETRAINED REFERENCE (150-STEP MODEL)")
print("=" * 90)

print(
    f"{'Dataset':<15}"
    f"{'Original Agree.':>18}"
    f"{'Unlearned Agree.':>20}"
    f"{'Original JS':>17}"
    f"{'Unlearned JS':>17}"
)

print("-" * 90)

for split in ["forget", "retain", "test"]:

    original_cmp = behavior[split][
        "original_vs_retrained"
    ]

    unlearned_cmp = behavior[split][
        "unlearned_vs_retrained"
    ]

    print(
        f"{split.capitalize():<15}"
        f"{original_cmp['prediction_agreement_percent']:>17.2f}%"
        f"{unlearned_cmp['prediction_agreement_percent']:>19.2f}%"
        f"{original_cmp['mean_js_divergence']:>17.6f}"
        f"{unlearned_cmp['mean_js_divergence']:>17.6f}"
    )

    # ============================================================
# Computational cost relative to full retraining
# ============================================================

retraining_time = retrained_training[
    "training_time_seconds"
]

print("\n" + "=" * 65)
print("COMPUTATIONAL COST RELATIVE TO RETRAINING")
print("=" * 65)

for name, timing in [
    ("NegGrad+ (100)", unlearning_100),
    ("NegGrad+ (150)", unlearning_150),
    ("NegGrad+ (200)", unlearning_200),
    ("NegGrad+ (500)", unlearning_500),
]:

    unlearning_time = timing[
        "unlearning_time_seconds"
    ]

    speedup = (
        retraining_time / unlearning_time
    )

    print(
        f"{name:<20}"
        f"Time: {unlearning_time:>6.2f} s | "
        f"Speedup vs retraining: {speedup:>6.2f}x"
    )

print(
    f"{'Retraining':<20}"
    f"Time: {retraining_time:>6.2f} s"
)