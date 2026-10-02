import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 0

DATA_DIR = Path("./data")
RESULT_DIR = Path("./results")
MODEL_DIR = Path("./models")

ORIGINAL_MODEL_PATH = (
    MODEL_DIR / "original_corrupted_cnn.pth"
)

RETRAINED_MODEL_PATH = (
    MODEL_DIR / "retrained_reference_cnn.pth"
)

UNLEARNED_MODEL_PATH = (
    MODEL_DIR / "unlearned_neggrad_plus_150steps_cnn.pth"
)

CORRUPTED_INDICES_PATH = (
    RESULT_DIR / "corrupted_indices.npy"
)

OUTPUT_PATH = (
    RESULT_DIR / "model_behavior_comparison.json"
)

# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CIFAR-10
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transform
)

corrupted_indices = np.load(
    CORRUPTED_INDICES_PATH
)

all_indices = np.arange(
    len(train_dataset)
)

retain_indices = np.setdiff1d(
    all_indices,
    corrupted_indices
)

forget_dataset = Subset(
    train_dataset,
    corrupted_indices.tolist()
)

retain_dataset = Subset(
    train_dataset,
    retain_indices.tolist()
)


# ============================================================
# Evaluation loaders
# ============================================================

forget_loader = DataLoader(
    forget_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

retain_loader = DataLoader(
    retain_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("=" * 70)
print("MODEL BEHAVIOR COMPARISON")
print("=" * 70)
print(f"Device           : {device}")
print(f"Forget samples   : {len(forget_dataset)}")
print(f"Retain samples   : {len(retain_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print("=" * 70)

# ============================================================
# Compact CNN
# Same architecture used by all three models
# ============================================================

class CompactCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 8 * 8,
                256
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                10
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # ============================================================
# Model loading
# ============================================================

def load_model(model_path):

    model = CompactCNN().to(device)

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


original_model = load_model(
    ORIGINAL_MODEL_PATH
)

unlearned_model = load_model(
    UNLEARNED_MODEL_PATH
)

retrained_model = load_model(
    RETRAINED_MODEL_PATH
)


print(f"Original model   : {ORIGINAL_MODEL_PATH}")
print(f"Unlearned model  : {UNLEARNED_MODEL_PATH}")
print(f"Retrained model  : {RETRAINED_MODEL_PATH}")

# ============================================================
# Prediction collection
# ============================================================

@torch.no_grad()
def collect_outputs(model, loader):

    all_probs = []
    all_preds = []
    all_targets = []

    for images, targets in loader:

        images = images.to(device)

        logits = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_probs.append(
            probabilities.cpu()
        )

        all_preds.append(
            predictions.cpu()
        )

        all_targets.append(
            targets.cpu()
        )

    return (
        torch.cat(all_probs, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_targets, dim=0)
    )

# ============================================================
# Behavioral comparison metrics
# ============================================================

def prediction_agreement(predictions_a, predictions_b):
    """
    Percentage of samples for which two models produce
    the same predicted class.
    """

    agreement = (
        predictions_a == predictions_b
    ).float().mean().item()

    return agreement * 100.0


def mean_js_divergence(probs_a, probs_b, eps=1e-12):
    """
    Mean Jensen-Shannon divergence between the predictive
    probability distributions of two models.
    """

    probs_a = probs_a.clamp_min(eps)
    probs_b = probs_b.clamp_min(eps)

    mixture = 0.5 * (
        probs_a + probs_b
    )

    kl_a = torch.sum(
        probs_a * torch.log(
            probs_a / mixture
        ),
        dim=1
    )

    kl_b = torch.sum(
        probs_b * torch.log(
            probs_b / mixture
        ),
        dim=1
    )

    js = 0.5 * (
        kl_a + kl_b
    )

    return js.mean().item()

# ============================================================
# Compare models against the Retrained Reference
# ============================================================

def compare_on_split(split_name, loader):

    print(f"\nEvaluating: {split_name}")

    original_probs, original_preds, _ = collect_outputs(
        original_model,
        loader
    )

    unlearned_probs, unlearned_preds, _ = collect_outputs(
        unlearned_model,
        loader
    )

    retrained_probs, retrained_preds, _ = collect_outputs(
        retrained_model,
        loader
    )

    original_agreement = prediction_agreement(
        original_preds,
        retrained_preds
    )

    unlearned_agreement = prediction_agreement(
        unlearned_preds,
        retrained_preds
    )

    original_js = mean_js_divergence(
        original_probs,
        retrained_probs
    )

    unlearned_js = mean_js_divergence(
        unlearned_probs,
        retrained_probs
    )

    return {
        "split": split_name,
        "num_samples": len(loader.dataset),

        "original_vs_retrained": {
            "prediction_agreement_percent": original_agreement,
            "mean_js_divergence": original_js
        },

        "unlearned_vs_retrained": {
            "prediction_agreement_percent": unlearned_agreement,
            "mean_js_divergence": unlearned_js
        }
    }

# ============================================================
# Run comparisons
# ============================================================

results = {
    "reference_model": "retrained_reference_cnn",
    "unlearned_model": "neggrad_plus_150steps",
    "metrics": {
        "prediction_agreement": "higher_is_closer",
        "mean_js_divergence": "lower_is_closer"
    }
}

for split_name, loader in [
    ("forget", forget_loader),
    ("retain", retain_loader),
    ("test", test_loader)
]:
    results[split_name] = compare_on_split(
        split_name,
        loader
    )


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 70)
print("BEHAVIORAL SIMILARITY TO RETRAINED REFERENCE")
print("=" * 70)

for split_name in ["forget", "retain", "test"]:

    split_result = results[split_name]

    original = split_result[
        "original_vs_retrained"
    ]

    unlearned = split_result[
        "unlearned_vs_retrained"
    ]

    print(f"\n{split_name.upper()} SET")

    print(
        f"Original  -> Agreement: "
        f"{original['prediction_agreement_percent']:.2f}% | "
        f"JS: {original['mean_js_divergence']:.6f}"
    )

    print(
        f"Unlearned -> Agreement: "
        f"{unlearned['prediction_agreement_percent']:.2f}% | "
        f"JS: {unlearned['mean_js_divergence']:.6f}"
    )


# ============================================================
# Save results
# ============================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )

print(f"\nResults saved to: {OUTPUT_PATH}")