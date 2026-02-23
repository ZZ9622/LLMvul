# Attention Head Importance Analysis Summary

**Generated:** 2026-02-22 20:30:29

**Model:** Gemma-2-2B (26 layers, 8 heads per layer)

## Dataset

- **TP Samples (Vulnerable, correctly detected):** 83
- **TN Samples (Safe, correctly detected):** 100
- **Source:** `tp_tn_samples.jsonl`

## Methodology

1. Extract attention patterns from the last token position (prediction position)
2. Compute attention concentration metrics:
   - **Max attention**: Highest attention weight (focus strength)
   - **Entropy**: Distribution uniformity (lower = more focused)
3. Calculate **Importance Score** = (Max_TP - Max_TN) + 0.5 × (Entropy_TN - Entropy_TP)
4. Rank heads by absolute importance

## Top-10 Most Important Attention Heads

| Rank | Layer | Head | Importance | Max(TP) | Max(TN) | Entropy(TP) | Entropy(TN) | Pattern Detected |
|------|-------|------|------------|---------|---------|-------------|-------------|------------------|
| 1 | 5 | 2 | -0.488989 | 0.3607 | 0.4937 | 2.6754 | 1.9634 | _[See heatmap]_ |
| 2 | 2 | 2 | -0.423475 | 0.4321 | 0.5543 | 2.5531 | 1.9504 | _[See heatmap]_ |
| 3 | 21 | 5 | -0.411157 | 0.2870 | 0.4272 | 2.6340 | 2.0919 | _[See heatmap]_ |
| 4 | 7 | 6 | -0.411024 | 0.0126 | 0.0209 | 3.7469 | 2.9415 | _[See heatmap]_ |
| 5 | 0 | 2 | -0.406797 | 0.0725 | 0.0860 | 4.2057 | 3.4192 | _[See heatmap]_ |
| 6 | 1 | 6 | -0.360858 | 0.0682 | 0.1125 | 4.5767 | 3.9436 | _[See heatmap]_ |
| 7 | 17 | 3 | -0.359640 | 0.2367 | 0.3236 | 3.1200 | 2.5745 | _[See heatmap]_ |
| 8 | 0 | 5 | -0.313146 | 0.3089 | 0.3553 | 3.2023 | 2.6687 | _[See heatmap]_ |
| 9 | 7 | 4 | -0.309863 | 0.0265 | 0.0330 | 4.5976 | 3.9909 | _[See heatmap]_ |
| 10 | 0 | 3 | -0.301709 | 0.1084 | 0.1490 | 4.5245 | 4.0023 | _[See heatmap]_ |
## Generated Files

- **Attention heatmaps:** `attention_rank{1-3}_L{layer}H{head}.png`
- **Importance ranking:** `attention_importance_ranking.png`
- **Layer distribution:** `attention_layer_distribution.png`
- **Raw data:** `attention_head_importance.json`
- **CWE attention heatmap:** `cwe_attention_heatmap.png`
- **CWE top heads:** `cwe_top_heads.png`
- **CWE head overlap:** `cwe_head_overlap.png`
- **CWE analysis data:** `cwe_attention_analysis.json`

## Key Findings

**Most important layers:** {5: 1, 2: 1, 21: 1, 7: 1, 0: 1}

**Heads with positive importance (favor TP):** 16/208
**Heads with negative importance (favor TN):** 192/208

## CWE-Specific Analysis

Analyzed 6 CWE categories with sufficient samples:

| CWE | Samples | Top Attention Head | Importance |
|-----|---------|-------------------|------------|
| CWE-119 | 6 | L21H5 | -0.161047 |
| CWE-125 | 8 | L21H5 | -0.146956 |
| CWE-362 | 5 | L3H5 | 0.317132 |
| CWE-416 | 6 | L5H2 | -0.196165 |
| CWE-703 | 7 | L3H6 | 0.207061 |
| CWE-787 | 18 | L21H5 | -0.144423 |
## Usage

```bash
# View heatmaps
ls /lunarc/nobackup/projects/lu2024-17-13/chun7871/LLMvul/out/plots/attention_analysis_20260222_202911/attention_rank*.png

# View summary charts
ls /lunarc/nobackup/projects/lu2024-17-13/chun7871/LLMvul/out/plots/attention_analysis_20260222_202911/attention_*.png

# View CWE-specific analysis
ls /lunarc/nobackup/projects/lu2024-17-13/chun7871/LLMvul/out/plots/attention_analysis_20260222_202911/cwe_*.png

# View CWE analysis data
cat /lunarc/nobackup/projects/lu2024-17-13/chun7871/LLMvul/out/log/attention_analysis_20260222_202911/cwe_attention_analysis.json
```
