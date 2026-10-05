# Evaluating Machine Unlearning for Trustworthy AI

Single-seed experimental study of NegGrad+ unlearning on CIFAR-10 using CompactCNN.

## Experimental Setup

- Dataset: CIFAR-10, with 50,000 training and 10,000 test images.
- Forget set: 5,000 training images assigned incorrect labels.
- Retain set: 45,000 training images with correct labels.
- Random seed: 42.
- Architecture: CompactCNN with 1,070,794 parameters.

## Experimental Workflow

1. Create the label-corruption partition.
2. Train the original model on retain and corrupted-label forget data.
3. Train a fresh reference model on retain data only.
4. Apply NegGrad+ to the original checkpoint.
5. Evaluate classification accuracy and behavioral similarity to retraining.

NegGrad+ minimizes:

    L = alpha * L_retain - (1 - alpha) * L_forget

The forget loss uses corrupted labels. The retain loss uses correct labels.
The configuration uses alpha = 0.9, Adam with learning rate 0.0001,
batch size 128, and gradient clipping at 1.0.

Update budgets are 100, 150, 200, and 500 steps.
The 150-step configuration is the selected intermediate operating point.

## Reproduce the Paper Figures

Run from the repository root:

    python src/make_paper_figures.py --validate-only
    python src/make_paper_figures.py

The script reads nine saved JSON files without rerunning experiments.
It saves three figures as PNG and PDF in results/figures/:

- figure_1_accuracy_tradeoff
- figure_2_behavioral_similarity
- figure_3_update_time

The paper_figures_provenance.json file records source data,
SHA-256 hashes, and the calculated runtime speedup.

The figure_1, figure_2, and figure_3 files are the current paper figures.
Other plot files are earlier outputs.

## Results and Limitations

Results are recorded in results/.
Reported measurements correspond to seed 42; repeated-seed uncertainty
has not been estimated.

Accuracy, prediction agreement, and Jensen–Shannon divergence measure
observed behavior. They do not establish complete data removal or a
privacy guarantee.

Retraining time includes per-epoch test evaluation.
The reported speedup compares the recorded retraining and unlearning procedures.

## Repository Notes

- src/train_baseline.py and results/original_results.json belong to the
  earlier clean-baseline experiment.
- The paper's original model is the corrupted-label model.
- Dataset files and trained checkpoints are excluded from Git.

## Status

Experiments are frozen. Paper preparation uses the verified saved results.
Multi-seed evaluation is future work.
