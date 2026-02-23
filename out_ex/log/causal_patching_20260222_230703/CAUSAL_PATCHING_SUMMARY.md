# Bidirectional Steering Experiment Summary

**Generated:** 2026-02-23 07:59:22

**Model:** Gemma-2-2B (26 layers, d_model=2304)

## Experiment Design

**Method:** Vector Addition (Steering)

**Formula:** `output = original + coeff × steering_vector`

**Steering Vector:** `Mean_Vuln - Mean_Safe`

**Steering Coefficient:** 8

### Direction 1: Safe → Vulnerable
- Selected 52 TP samples (baseline: predict 'vul')
- Applied: `output = original - 8 × (Mean_Vuln - Mean_Safe)`
- Measured flip rate: % of samples changing 'vul' → 'safe'

### Direction 2: Vulnerable → Safe
- Selected 211 TN samples (baseline: predict 'nonvul')
- Applied: `output = original + 8 × (Mean_Vuln - Mean_Safe)`
- Measured flip rate: % of samples changing 'safe' → 'vul'

## Key Findings

**Direction 1 (S→V) Critical Layer:** Layer 25 (86.5%)

**Direction 2 (V→S) Critical Layer:** Layer 14 (87.7%)

**Layer Agreement:** ✗ Differ (diff=11 layers)

## Stage Analysis

| Stage | S→V Flip Rate | V→S Flip Rate |
|-------|---------------|---------------|
| Early (L0-7) | 13.7% | 48.2% |
| Middle (L8-16) | 48.1% | 78.9% |
| Late (L17-25) | 28.4% | 41.1% |

## Generated Files

- `flip_rate_bidirectional.png`: Comparison of both directions
- `flip_rate_by_stage.png`: Early vs middle vs late layers
- `causal_patching_results.json`: Full detailed results
