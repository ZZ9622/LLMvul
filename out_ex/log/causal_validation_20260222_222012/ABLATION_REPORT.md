# Causal Validation: Ablation Studies Report

**Generated:** 2026-02-23 04:28:30

## Baseline Performance

- **Overall Accuracy:** 100.00%
- **TP Accuracy (Vulnerable):** 100.00%
- **TN Accuracy (Safe):** 100.00%

## Experiment 1: Layer-wise Ablation

| Layer | Type | Overall Acc | TP Acc | TN Acc | Acc Drop | Impact |
|-------|------|-------------|--------|--------|----------|--------|
| 6 | Key | 76.4% | 27.7% | 94.6% | 23.6% | High |
| 7 | Key | 51.1% | 19.3% | 63.1% | 48.9% | High |
| 10 | Key | 75.7% | 19.3% | 96.8% | 24.3% | High |
| 11 | Key | 48.9% | 6.0% | 64.9% | 51.1% | High |
| 1 | Control | 52.1% | 54.2% | 51.4% | 47.9% | High |
| 2 | Control | 51.5% | 15.7% | 64.9% | 48.5% | High |
| 15 | Control | 74.4% | 14.5% | 96.8% | 25.6% | High |
| 16 | Control | 71.1% | 20.5% | 90.1% | 28.9% | High |

## Experiment 2: Attention Head Ablation

| Head | Overall Acc | TP Acc | TN Acc | Acc Drop | TN Impact |
|------|-------------|--------|--------|----------|----------|
| L0H2 | 85.9% | 61.4% | 95.0% | 14.1% | Balanced |
| L5H2 | 85.9% | 61.4% | 95.0% | 14.1% | Balanced |
| L2H2 | 85.9% | 61.4% | 95.0% | 14.1% | Balanced |

## Experiment 3: MLP Neuron Ablation

| Layer | Neurons | Overall Acc | TP Acc | TN Acc | Acc Drop |
|-------|---------|-------------|--------|--------|----------|
| 6 | Top-20 | 85.2% | 68.7% | 91.4% | 14.8% |
| 7 | Top-20 | 82.3% | 50.6% | 94.1% | 17.7% |
| 10 | Top-20 | 85.6% | 65.1% | 93.2% | 14.4% |
| 11 | Top-20 | 86.6% | 63.9% | 95.0% | 13.4% |

## Key Findings

1. **Most Critical Layer:** Layer 11 (accuracy drop: 51.1%)
2. **Most Critical Head:** L0H2 (accuracy drop: 14.1%)

## Interpretation

- **High impact (>20% drop):** Component is critical for decision-making
- **Medium impact (5-20% drop):** Component contributes to performance
- **Low impact (<5% drop):** Component is not essential
- **TN-specific drop:** 'Safety detector' - recognizes safe patterns
- **TP-specific drop:** 'Vulnerability detector' - recognizes unsafe patterns
