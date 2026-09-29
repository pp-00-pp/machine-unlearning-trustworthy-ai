# Evaluating Machine Unlearning for Trustworthy AI

Experimental study of machine unlearning using the CIFAR-10 image classification dataset.

## Research Objective

This project investigates whether machine unlearning can reduce the influence of selected training samples while preserving the predictive utility of a trained image classification model.

## Dataset

CIFAR-10

## Experimental Framework

The study compares three models:

1. **Original Model** — trained on the complete CIFAR-10 training dataset.
2. **Unlearned Model** — obtained by applying a machine unlearning method to remove the influence of designated training samples.
3. **Retrained Reference Model** — trained from scratch without the designated forget samples.

## Evaluation

The experiments focus on:

- Forgetting effectiveness
- Model utility after unlearning
- Comparison with retraining
- Computational cost

## Status

Research and experimentation in progress.
