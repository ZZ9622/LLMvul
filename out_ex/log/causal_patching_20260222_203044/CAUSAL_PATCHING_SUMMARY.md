# Causal Patching Experiment Summary

**Generated:** 2026-02-22 22:19:12

**Model:** Gemma-2-2B (26 layers, d_model=2304)

## Experiment Design

**Question:** If we replace a vulnerable sample's internal representation with the mean safe representation, will the model change its prediction?

**Method:**
1. Computed mean safe vector from 222 TN samples
2. Selected 51 TP samples (baseline: all predict 'vul')
3. For each layer, patched the residual stream with mean safe vector
4. Measured flip rate: % of samples that changed from 'vul' to 'safe'

## Key Findings

**Critical Decision Layer:** Layer 14 (Flip Rate: 41.2%)

This layer is where the model makes its key vulnerability/safety decision.

**Top-5 Most Effective Layers:**

| Rank | Layer | Flip Rate |
|------|-------|----------|
| 1 | 14 | 41.2% |
| 2 | 16 | 41.2% |
| 3 | 13 | 35.3% |
| 4 | 15 | 35.3% |
| 5 | 12 | 33.3% |

**Stage Analysis:**

| Stage | Layers | Avg Flip Rate |
|-------|--------|---------------|
| Early | 0-7 | 9.8% |
| Middle | 8-16 | 27.5% |
| Late | 17-25 | 22.4% |

## Interpretation

- **Low flip rate (< 20%)**: Layer hasn't formed vulnerability judgment yet
- **High flip rate (> 60%)**: Layer is critical for decision-making
- **Decreasing flip rate**: Decision has been solidified, hard to change

## Generated Files

- `flip_rate_curve.png`: Flip rate per layer
- `flip_heatmap.png`: Which layer flips which sample
- `flip_rate_by_stage.png`: Early vs middle vs late layers
- `causal_patching_results.json`: Full detailed results
