#!/usr/bin/env python3
"""
Attention Head Analysis for Vulnerability Detection
"""

import os
# ── Repository root & output directory (auto-detected) ───────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)           # LLMvul/
OUTPUT_BASE = os.environ.get(
    "LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out")
)
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import AutoTokenizer
from collections import defaultdict, Counter
import gc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit-tracer", "circuit-tracer"))

from circuit_tracer.replacement_model import ReplacementModel

DATA_PATH = os.path.join(ROOT_DIR, "data", "tp_tn_samples.jsonl")

# ── Check for tp_tn_samples.jsonl ────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] {DATA_PATH} not found.")
    print("[INFO]  This file is produced by running: python scripts/prime.py")
    print("[INFO]  Then copy TP/TN results from the output to data/tp_tn_samples.jsonl")
    raise SystemExit(1)

MODEL_PATH = "Chun9622/llmvul-finetuned-gemma"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(OUTPUT_BASE, "log", f"attention_analysis_{ts}")
PLOT_DIR = os.path.join(OUTPUT_BASE, "plots", f"attention_analysis_{ts}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TOP_K_HEADS = 10
VISUALIZE_TOP_N = 3
MAX_SAMPLES_PER_TYPE = 100
MAX_SEQ_LENGTH = 512

log_file = open(os.path.join(LOG_DIR, f"attention_analysis_{ts}.txt"), "w")
sys.stdout = log_file
sys.stderr = log_file

print("="*80)
print("Attention Head Importance Analysis")
print("="*80)
print(f"Timestamp: {ts}")
print(f"Device: {DEVICE}")
print(f"Output: {PLOT_DIR}")
print("="*80)

def patch_model_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_official_model_name
    loading.get_official_model_name = (
        lambda model_name: "google/gemma-2-2b"
        if model_name == MODEL_PATH
        else original(model_name)
    )

def patch_model_config_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_pretrained_model_config
    def patched(model_name, **kwargs):
        if model_name == MODEL_PATH:
            from transformers import AutoConfig
            return AutoConfig.from_pretrained(model_name)
        return original(model_name, **kwargs)
    loading.get_pretrained_model_config = patched

patch_model_loading()
patch_model_config_loading()

print("\n[1/6] Loading Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

rm = ReplacementModel.from_pretrained(
    MODEL_PATH,
    transcoder_set="gemma",
    device=DEVICE,
    dtype=torch.float16
)
rm.eval()

n_layers = rm.cfg.n_layers
n_heads = rm.cfg.n_heads

print(f"  Model: Gemma-2-2B")
print(f"  Layers: {n_layers}")
print(f"  Heads per layer: {n_heads}")
print(f"  Total heads: {n_layers * n_heads}")

print(f"\n[2/6] Loading TP/TN samples from {DATA_PATH}...")

def load_tp_tn_samples(jsonl_path):
    """Load TP and TN samples with CWE information"""
    tp_samples = []
    tn_samples = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                pred_type = data.get('prediction_type', '')
                code = data.get('func', '').strip()
                
                if not code:
                    continue
                
                if len(code) > 2000:
                    code = code[:2000]
                
                prompt = f"Code: {code}\n\nQuestion: Is this code safe or vulnerable?\nAnswer:"
                
                # Extract CWE information
                cwe_list = data.get('cwe', [])
                cwe_category = cwe_list[0] if cwe_list else 'Unknown'
                
                sample = {
                    'idx': data.get('idx', -1),
                    'prompt': prompt,
                    'type': pred_type,
                    'cwe': cwe_category
                }
                
                if pred_type == 'TP':
                    tp_samples.append(sample)
                elif pred_type == 'TN':
                    tn_samples.append(sample)
                    
            except Exception as e:
                continue
    
    return tp_samples, tn_samples

tp_samples, tn_samples = load_tp_tn_samples(DATA_PATH)

print(f"  Loaded {len(tp_samples)} TP samples (Vulnerable, correctly detected)")
print(f"  Loaded {len(tn_samples)} TN samples (Safe, correctly detected)")

if len(tp_samples) > MAX_SAMPLES_PER_TYPE:
    tp_samples = tp_samples[:MAX_SAMPLES_PER_TYPE]
    print(f"  Limited to {MAX_SAMPLES_PER_TYPE} TP samples")

if len(tn_samples) > MAX_SAMPLES_PER_TYPE:
    tn_samples = tn_samples[:MAX_SAMPLES_PER_TYPE]
    print(f"  Limited to {MAX_SAMPLES_PER_TYPE} TN samples")

def self_entropy(attn_vec):
    """Calculate entropy of attention distribution"""
    attn_vec = attn_vec + 1e-10
    attn_vec = attn_vec / np.sum(attn_vec)
    entropy = -np.sum(attn_vec * np.log(attn_vec))
    return entropy

def extract_attention_patterns(samples, sample_type):
    """
    Extract attention patterns from samples
    
    Returns:
        attention_scores: shape (n_samples, n_layers, n_heads, seq_len, seq_len)
        sequence_lengths: actual length of each sample
    """
    print(f"\n[3/6] Extracting attention patterns from {sample_type} samples...")
    
    head_attentions = defaultdict(list)
    valid_samples = 0
    
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(samples)}")
        
        try:
            inputs = tokenizer(
                sample['prompt'], 
                return_tensors='pt', 
                truncation=True, 
                max_length=MAX_SEQ_LENGTH
            ).to(DEVICE)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            seq_len = input_ids.shape[1]
            
            with torch.no_grad():
                _, cache = rm.run_with_cache(input_ids, return_type=None)
            
            for layer in range(n_layers):
                attn_key = f"blocks.{layer}.attn.hook_pattern"
                
                if attn_key in cache:
                    attn_pattern = cache[attn_key].detach().cpu().float()
                    
                    last_token_idx = seq_len - 1
                    
                    for head in range(n_heads):
                        attn_vec = attn_pattern[0, head, last_token_idx, :].numpy()
                        
                        head_attentions[(layer, head)].append({
                            'attention': attn_vec,
                            'seq_len': seq_len,
                            'sample_idx': sample['idx']
                        })
            
            valid_samples += 1
            
            del cache, inputs
            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"  Warning: Failed to process sample {sample.get('idx', i)}: {e}")
            continue
    
    print(f"  Successfully processed {valid_samples}/{len(samples)} samples")
    return head_attentions

tp_attentions = extract_attention_patterns(tp_samples, "TP")
tn_attentions = extract_attention_patterns(tn_samples, "TN")

print(f"\n[4/6] Computing Attention Importance Scores...")

head_importance = []

for layer in range(n_layers):
    for head in range(n_heads):
        key = (layer, head)
        
        if key not in tp_attentions or key not in tn_attentions:
            continue
        
        tp_data = tp_attentions[key]
        tn_data = tn_attentions[key]
        
        tp_max_attn = []
        tn_max_attn = []
        tp_entropy = []
        tn_entropy = []
        
        for d in tp_data:
            attn = d['attention'].copy()
            seq_len = d['seq_len']
            if seq_len > 2:
                attn_masked = attn[1:seq_len-1]
            else:
                attn_masked = attn
            
            if len(attn_masked) > 0:
                tp_max_attn.append(np.max(attn_masked))
                tp_entropy.append(self_entropy(attn_masked))
        
        for d in tn_data:
            attn = d['attention'].copy()
            seq_len = d['seq_len']
            if seq_len > 2:
                attn_masked = attn[1:seq_len-1]
            else:
                attn_masked = attn
            
            if len(attn_masked) > 0:
                tn_max_attn.append(np.max(attn_masked))
                tn_entropy.append(self_entropy(attn_masked))
        
        if not tp_max_attn or not tn_max_attn:
            continue
        
        mean_tp_max = np.mean(tp_max_attn)
        mean_tn_max = np.mean(tn_max_attn)
        
        mean_tp_entropy = np.mean(tp_entropy)
        mean_tn_entropy = np.mean(tn_entropy)
        
        importance_max = mean_tp_max - mean_tn_max
        importance_entropy = mean_tn_entropy - mean_tp_entropy
        
        importance = importance_max + 0.5 * importance_entropy
        
        head_importance.append({
            'layer': int(layer),
            'head': int(head),
            'importance': float(importance),
            'mean_tp_max': float(mean_tp_max),
            'mean_tn_max': float(mean_tn_max),
            'mean_tp_entropy': float(mean_tp_entropy),
            'mean_tn_entropy': float(mean_tn_entropy),
            'tp_count': len(tp_data),
            'tn_count': len(tn_data)
        })

head_importance.sort(key=lambda x: abs(x['importance']), reverse=True)

print(f"\n  Top-{TOP_K_HEADS} Most Important Attention Heads:")
print(f"  {'Rank':<6} {'Layer':<8} {'Head':<8} {'Importance':<15} {'Max(TP)':<15} {'Max(TN)':<15}")
print("  " + "-"*75)

for i, head_info in enumerate(head_importance[:TOP_K_HEADS], 1):
    print(f"  {i:<6} {head_info['layer']:<8} {head_info['head']:<8} "
          f"{head_info['importance']:>14.6f} {head_info['mean_tp_max']:>14.6f} {head_info['mean_tn_max']:>14.6f}")

results_path = os.path.join(LOG_DIR, "attention_head_importance.json")
with open(results_path, 'w') as f:
    json.dump(head_importance, f, indent=2)
print(f"\n  Results saved to: {results_path}")

# Analysis of whether different CWE categories use different attention heads
print(f"\n[4.5/6] Analyzing CWE-specific attention patterns...")

# Group TP samples by CWE category
cwe_groups = defaultdict(list)
for sample in tp_samples:
    cwe = sample.get('cwe', 'Unknown')
    cwe_groups[cwe].append(sample)

# Filter CWE categories with enough samples
MIN_SAMPLES_PER_CWE = 5
cwe_categories = {cwe: samples for cwe, samples in cwe_groups.items() 
                  if len(samples) >= MIN_SAMPLES_PER_CWE}

print(f"  Found {len(cwe_categories)} CWE categories with >= {MIN_SAMPLES_PER_CWE} samples:")
for cwe, samples in sorted(cwe_categories.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"    {cwe}: {len(samples)} samples")

if len(cwe_categories) >= 2:
    # Extract attention patterns for each CWE category
    cwe_attentions = {}
    for cwe, samples in cwe_categories.items():
        print(f"\n  Extracting attention for {cwe} ({len(samples)} samples)...")
        cwe_attentions[cwe] = extract_attention_patterns(samples, cwe)
    
    # Compute head importance for each CWE category
    cwe_head_importance = {}
    TOP_K_PER_CWE = 5
    
    for cwe, cwe_attn in cwe_attentions.items():
        head_scores = []
        
        for layer in range(n_layers):
            for head in range(n_heads):
                key = (layer, head)
                
                if key not in cwe_attn or key not in tn_attentions:
                    continue
                
                cwe_data = cwe_attn[key]
                tn_data = tn_attentions[key]
                
                cwe_max_attn = []
                tn_max_attn = []
                
                for d in cwe_data:
                    attn = d['attention'].copy()
                    seq_len = d['seq_len']
                    if seq_len > 2:
                        attn_masked = attn[1:seq_len-1]
                    else:
                        attn_masked = attn
                    if len(attn_masked) > 0:
                        cwe_max_attn.append(np.max(attn_masked))
                
                for d in tn_data:
                    attn = d['attention'].copy()
                    seq_len = d['seq_len']
                    if seq_len > 2:
                        attn_masked = attn[1:seq_len-1]
                    else:
                        attn_masked = attn
                    if len(attn_masked) > 0:
                        tn_max_attn.append(np.max(attn_masked))
                
                if not cwe_max_attn or not tn_max_attn:
                    continue
                
                mean_cwe_max = np.mean(cwe_max_attn)
                mean_tn_max = np.mean(tn_max_attn)
                importance = mean_cwe_max - mean_tn_max
                
                head_scores.append({
                    'layer': int(layer),
                    'head': int(head),
                    'importance': float(importance),
                    'mean_cwe_max': float(mean_cwe_max),
                    'mean_tn_max': float(mean_tn_max)
                })
        
        head_scores.sort(key=lambda x: abs(x['importance']), reverse=True)
        cwe_head_importance[cwe] = head_scores[:TOP_K_PER_CWE]
    
    # Save CWE-specific results
    cwe_results_path = os.path.join(LOG_DIR, "cwe_attention_analysis.json")
    with open(cwe_results_path, 'w') as f:
        json.dump(cwe_head_importance, f, indent=2)
    print(f"\n  CWE-specific results saved to: {cwe_results_path}")
    
    # Generate CWE comparison heatmap
    print(f"\n  Generating CWE attention head comparison heatmap...")
    
    cwe_list = sorted(cwe_head_importance.keys())
    head_matrix = np.zeros((len(cwe_list), n_layers * n_heads))
    
    for i, cwe in enumerate(cwe_list):
        for head_info in cwe_head_importance[cwe]:
            head_idx = head_info['layer'] * n_heads + head_info['head']
            head_matrix[i, head_idx] = head_info['importance']
    
    # Select top heads across all CWEs for visualization
    head_importance_sum = np.sum(np.abs(head_matrix), axis=0)
    top_head_indices = np.argsort(head_importance_sum)[-30:]  # Top 30 heads
    
    fig, ax = plt.subplots(figsize=(16, max(6, len(cwe_list) * 0.5)))
    
    heatmap_data = head_matrix[:, top_head_indices]
    head_labels = [f"L{idx//n_heads}H{idx%n_heads}" for idx in top_head_indices]
    
    sns.heatmap(heatmap_data, 
                xticklabels=head_labels,
                yticklabels=cwe_list,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Importance Score'},
                ax=ax)
    
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('CWE Category', fontsize=12)
    ax.set_title('CWE-Specific Attention Head Importance Patterns\n'
                 '(Top 30 Most Discriminative Heads)',
                 fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_path = os.path.join(PLOT_DIR, "cwe_attention_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {heatmap_path}")
    
    # Generate top heads per CWE bar chart
    print(f"\n  Generating per-CWE top attention heads chart...")
    
    n_cwes = len(cwe_list)
    fig, axes = plt.subplots(n_cwes, 1, figsize=(12, n_cwes * 3))
    
    if n_cwes == 1:
        axes = [axes]
    
    for i, cwe in enumerate(cwe_list):
        ax = axes[i]
        top_heads = cwe_head_importance[cwe][:TOP_K_PER_CWE]
        
        labels = [f"L{h['layer']}H{h['head']}" for h in top_heads]
        importances = [h['importance'] for h in top_heads]
        colors = ['red' if x > 0 else 'blue' for x in importances]
        
        ax.barh(labels, importances, color=colors, alpha=0.7)
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_title(f'{cwe} (n={len(cwe_groups[cwe])})', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Top-{TOP_K_PER_CWE} Attention Heads per CWE Category',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    per_cwe_path = os.path.join(PLOT_DIR, "cwe_top_heads.png")
    plt.savefig(per_cwe_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {per_cwe_path}")
    
    # Compute head overlap between CWE categories
    print(f"\n  Computing attention head overlap between CWE categories...")
    
    overlap_matrix = np.zeros((len(cwe_list), len(cwe_list)))
    
    for i, cwe1 in enumerate(cwe_list):
        heads1 = {(h['layer'], h['head']) for h in cwe_head_importance[cwe1][:TOP_K_PER_CWE]}
        for j, cwe2 in enumerate(cwe_list):
            heads2 = {(h['layer'], h['head']) for h in cwe_head_importance[cwe2][:TOP_K_PER_CWE]}
            overlap = len(heads1 & heads2)
            overlap_matrix[i, j] = overlap
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(overlap_matrix,
                xticklabels=cwe_list,
                yticklabels=cwe_list,
                annot=True,
                fmt='.0f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Shared Heads'},
                ax=ax)
    
    ax.set_xlabel('CWE Category', fontsize=12)
    ax.set_ylabel('CWE Category', fontsize=12)
    ax.set_title(f'Attention Head Overlap Between CWE Categories\n'
                 f'(Top-{TOP_K_PER_CWE} heads per category)',
                 fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    overlap_path = os.path.join(PLOT_DIR, "cwe_head_overlap.png")
    plt.savefig(overlap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {overlap_path}")
    
    # Statistical summary
    print(f"\n  CWE-Specific Attention Analysis Summary:")
    print(f"  {'CWE':<15} {'Samples':<10} {'Top Head':<15} {'Importance':<12}")
    print("  " + "-"*60)
    
    for cwe in cwe_list:
        top_head = cwe_head_importance[cwe][0]
        print(f"  {cwe:<15} {len(cwe_groups[cwe]):<10} "
              f"L{top_head['layer']}H{top_head['head']:<12} "
              f"{top_head['importance']:>11.6f}")
    
else:
    print(f"\n  Warning: Not enough CWE categories for comparison (need >= 2)")

print(f"\n[5/6] Generating attention heatmaps for Top-{VISUALIZE_TOP_N} heads...")

for rank, head_info in enumerate(head_importance[:VISUALIZE_TOP_N], 1):
    layer = head_info['layer']
    head = head_info['head']
    key = (layer, head)
    
    print(f"\n  [{rank}/{VISUALIZE_TOP_N}] Layer {layer}, Head {head}")
    print(f"      Importance: {head_info['importance']:.6f}")
    
    tp_data = tp_attentions[key]
    tn_data = tn_attentions[key]
    
    def get_middle_attention_strength(d):
        attn = d['attention'].copy()
        seq_len = d['seq_len']
        if seq_len > 2:
            middle_attn = attn[1:seq_len-1]
            return np.sum(middle_attn)
        return np.sum(attn)
    
    tp_sample = max(tp_data, key=get_middle_attention_strength)
    tn_sample = max(tn_data, key=get_middle_attention_strength)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    ax1 = axes[0]
    tp_attn = tp_sample['attention'].copy()
    seq_len_tp = tp_sample['seq_len']
    
    tp_attn_masked = tp_attn.copy()
    tp_attn_masked[0] = 0
    if seq_len_tp > 1:
        tp_attn_masked[seq_len_tp-1] = 0
    
    tp_original_max = np.max(tp_attn[:seq_len_tp])
    tp_masked_max = np.max(tp_attn_masked[1:seq_len_tp-1]) if seq_len_tp > 2 else 0
    tp_sink_weight = tp_attn[0] + (tp_attn[seq_len_tp-1] if seq_len_tp > 1 else 0)
    
    positions = np.arange(seq_len_tp)
    ax1.plot(positions, tp_attn[:seq_len_tp], 'o-', color='darkred', 
             linewidth=2, markersize=4, alpha=0.3, label='Original (with sink)')
    ax1.plot(positions, tp_attn_masked[:seq_len_tp], 'o-', color='red', 
             linewidth=2.5, markersize=5, label='Masked (code only)')
    
    ax1.axvspan(-0.5, 0.5, alpha=0.2, color='gray', label='BOS (masked)')
    if seq_len_tp > 1:
        ax1.axvspan(seq_len_tp-1.5, seq_len_tp-0.5, alpha=0.2, color='gray', label='Self-attn (masked)')
    
    ax1.set_title(f'TP (Vulnerable) - Layer {layer}, Head {head}\n'
                  f'Sample ID: {tp_sample["sample_idx"]} | '
                  f'Sink Weight: {tp_sink_weight:.2%} | Code Max: {tp_masked_max:.4f}',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Token Position', fontsize=11)
    ax1.set_ylabel('Attention Weight', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(-1, seq_len_tp)
    
    ax2 = axes[1]
    tn_attn = tn_sample['attention'].copy()
    seq_len_tn = tn_sample['seq_len']
    
    tn_attn_masked = tn_attn.copy()
    tn_attn_masked[0] = 0
    if seq_len_tn > 1:
        tn_attn_masked[seq_len_tn-1] = 0
    
    tn_original_max = np.max(tn_attn[:seq_len_tn])
    tn_masked_max = np.max(tn_attn_masked[1:seq_len_tn-1]) if seq_len_tn > 2 else 0
    tn_sink_weight = tn_attn[0] + (tn_attn[seq_len_tn-1] if seq_len_tn > 1 else 0)
    
    positions = np.arange(seq_len_tn)
    ax2.plot(positions, tn_attn[:seq_len_tn], 'o-', color='darkblue', 
             linewidth=2, markersize=4, alpha=0.3, label='Original (with sink)')
    ax2.plot(positions, tn_attn_masked[:seq_len_tn], 'o-', color='blue', 
             linewidth=2.5, markersize=5, label='Masked (code only)')
    
    ax2.axvspan(-0.5, 0.5, alpha=0.2, color='gray', label='BOS (masked)')
    if seq_len_tn > 1:
        ax2.axvspan(seq_len_tn-1.5, seq_len_tn-0.5, alpha=0.2, color='gray', label='Self-attn (masked)')
    
    ax2.set_title(f'TN (Safe) - Layer {layer}, Head {head}\n'
                  f'Sample ID: {tn_sample["sample_idx"]} | '
                  f'Sink Weight: {tn_sink_weight:.2%} | Code Max: {tn_masked_max:.4f}',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Token Position', fontsize=11)
    ax2.set_ylabel('Attention Weight', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-1, seq_len_tn)
    
    plt.suptitle(f'Rank #{rank}: Attention Pattern Comparison (Attention Sink Removed)\n'
                 f'Importance Score: {head_info["importance"]:.6f} | '
                 f'TP Code Max: {tp_masked_max:.4f}, TN Code Max: {tn_masked_max:.4f}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, f"attention_rank{rank}_L{layer}H{head}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"      Saved: {save_path}")
    print(f"      TP: Sink={tp_sink_weight:.2%}, Code Max={tp_masked_max:.4f}")
    print(f"      TN: Sink={tn_sink_weight:.2%}, Code Max={tn_masked_max:.4f}")

print(f"\n  Generating summary visualizations...")

fig, ax = plt.subplots(figsize=(12, 8))

top_k_data = head_importance[:TOP_K_HEADS]
labels = [f"L{d['layer']}H{d['head']}" for d in top_k_data]
importances = [d['importance'] for d in top_k_data]
colors = ['red' if x > 0 else 'blue' for x in importances]

ax.barh(labels, importances, color=colors, alpha=0.7)
ax.set_xlabel('Attention Importance Score (Mean_TP - Mean_TN)', fontsize=12)
ax.set_ylabel('Attention Head (Layer.Head)', fontsize=12)
ax.set_title(f'Top-{TOP_K_HEADS} Most Important Attention Heads for Vulnerability Detection',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
bar_path = os.path.join(PLOT_DIR, "attention_importance_ranking.png")
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved: {bar_path}")

fig, ax = plt.subplots(figsize=(14, 6))

layer_importances = defaultdict(list)
for head_info in head_importance:
    layer_importances[head_info['layer']].append(abs(head_info['importance']))

layers = sorted(layer_importances.keys())
avg_importance_per_layer = [np.mean(layer_importances[l]) for l in layers]

ax.plot(layers, avg_importance_per_layer, marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Average Absolute Importance', fontsize=12)
ax.set_title('Layer-wise Attention Importance Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, n_layers, 2))

plt.tight_layout()
layer_path = os.path.join(PLOT_DIR, "attention_layer_distribution.png")
plt.savefig(layer_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved: {layer_path}")

print(f"\n  Generating summary report...")

summary_path = os.path.join(LOG_DIR, "ATTENTION_ANALYSIS_SUMMARY.md")
with open(summary_path, 'w') as f:
    f.write("# Attention Head Importance Analysis Summary\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Model:** Gemma-2-2B ({n_layers} layers, {n_heads} heads per layer)\n\n")
    
    f.write("## Dataset\n\n")
    f.write(f"- **TP Samples (Vulnerable, correctly detected):** {len(tp_samples)}\n")
    f.write(f"- **TN Samples (Safe, correctly detected):** {len(tn_samples)}\n")
    f.write(f"- **Source:** `tp_tn_samples.jsonl`\n\n")
    
    f.write("## Methodology\n\n")
    f.write("1. Extract attention patterns from the last token position (prediction position)\n")
    f.write("2. Compute attention concentration metrics:\n")
    f.write("   - **Max attention**: Highest attention weight (focus strength)\n")
    f.write("   - **Entropy**: Distribution uniformity (lower = more focused)\n")
    f.write("3. Calculate **Importance Score** = (Max_TP - Max_TN) + 0.5 × (Entropy_TN - Entropy_TP)\n")
    f.write("4. Rank heads by absolute importance\n\n")
    
    f.write(f"## Top-{TOP_K_HEADS} Most Important Attention Heads\n\n")
    f.write("| Rank | Layer | Head | Importance | Max(TP) | Max(TN) | Entropy(TP) | Entropy(TN) | Pattern Detected |\n")
    f.write("|------|-------|------|------------|---------|---------|-------------|-------------|------------------|\n")
    
    for i, head_info in enumerate(head_importance[:TOP_K_HEADS], 1):
        f.write(f"| {i} | {head_info['layer']} | {head_info['head']} | "
                f"{head_info['importance']:.6f} | {head_info['mean_tp_max']:.4f} | "
                f"{head_info['mean_tn_max']:.4f} | {head_info.get('mean_tp_entropy', 0):.4f} | "
                f"{head_info.get('mean_tn_entropy', 0):.4f} | _[See heatmap]_ |\n")
    
    f.write("\n## Interpretation\n\n")
    f.write("- **Positive Importance:** Head is more focused (higher max, lower entropy) in TP (vulnerable) cases\n")
    f.write("- **Negative Importance:** Head is more focused in TN (safe) cases\n")
    f.write("- **Higher absolute value:** More discriminative for vulnerability detection\n")
    f.write("- **Max attention:** Measures how strongly the head focuses on key tokens\n")
    f.write("- **Entropy:** Lower entropy = more selective attention pattern\n\n")
    
    f.write("## Generated Files\n\n")
    f.write(f"- **Attention heatmaps:** `attention_rank{{1-{VISUALIZE_TOP_N}}}_L{{layer}}H{{head}}.png`\n")
    f.write(f"- **Importance ranking:** `attention_importance_ranking.png`\n")
    f.write(f"- **Layer distribution:** `attention_layer_distribution.png`\n")
    f.write(f"- **Raw data:** `attention_head_importance.json`\n")
    if len(cwe_categories) >= 2:
        f.write(f"- **CWE attention heatmap:** `cwe_attention_heatmap.png`\n")
        f.write(f"- **CWE top heads:** `cwe_top_heads.png`\n")
        f.write(f"- **CWE head overlap:** `cwe_head_overlap.png`\n")
        f.write(f"- **CWE analysis data:** `cwe_attention_analysis.json`\n")
    f.write("\n")
    
    f.write("## Key Findings\n\n")
    
    top_layers = Counter([h['layer'] for h in head_importance[:5]])
    f.write(f"**Most important layers:** {dict(top_layers)}\n\n")
    
    positive_count = sum(1 for h in head_importance if h['importance'] > 0)
    negative_count = sum(1 for h in head_importance if h['importance'] < 0)
    f.write(f"**Heads with positive importance (favor TP):** {positive_count}/{len(head_importance)}\n")
    f.write(f"**Heads with negative importance (favor TN):** {negative_count}/{len(head_importance)}\n\n")
    
    if len(cwe_categories) >= 2:
        f.write("## CWE-Specific Analysis\n\n")
        f.write(f"Analyzed {len(cwe_categories)} CWE categories with sufficient samples:\n\n")
        f.write("| CWE | Samples | Top Attention Head | Importance |\n")
        f.write("|-----|---------|-------------------|------------|\n")
        
        for cwe in sorted(cwe_head_importance.keys()):
            top_head = cwe_head_importance[cwe][0]
            f.write(f"| {cwe} | {len(cwe_groups[cwe])} | "
                   f"L{top_head['layer']}H{top_head['head']} | "
                   f"{top_head['importance']:.6f} |\n")
        
        f.write("\n### Key Observations\n\n")
        f.write("- Different CWE categories show varying attention head preferences\n")
        f.write("- The heatmap reveals specialized attention patterns for vulnerability types\n")
        f.write("- Head overlap analysis shows which CWEs share similar detection mechanisms\n")
        f.write("- Lower overlap suggests distinct attention mechanisms for different vulnerability classes\n\n")
    
    f.write("## Usage\n\n")
    f.write("```bash\n")
    f.write(f"# View heatmaps\n")
    f.write(f"ls {PLOT_DIR}/attention_rank*.png\n\n")
    f.write(f"# View summary charts\n")
    f.write(f"ls {PLOT_DIR}/attention_*.png\n\n")
    if len(cwe_categories) >= 2:
        f.write(f"# View CWE-specific analysis\n")
        f.write(f"ls {PLOT_DIR}/cwe_*.png\n\n")
        f.write(f"# View CWE analysis data\n")
        f.write(f"cat {LOG_DIR}/cwe_attention_analysis.json\n")
    f.write("```\n")

print(f"      Saved: {summary_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated files:")
print(f"  Plots: {PLOT_DIR}")
print(f"  Logs: {LOG_DIR}")
print(f"\nKey outputs:")
print(f"  - {VISUALIZE_TOP_N} attention heatmap comparisons")
print(f"  - Top-{TOP_K_HEADS} importance ranking chart")
print(f"  - Layer-wise distribution chart")
if len(cwe_categories) >= 2:
    print(f"  - CWE-specific attention analysis ({len(cwe_categories)} categories)")
    print(f"  - CWE attention head heatmap")
    print(f"  - CWE head overlap matrix")
print(f"  - Summary report with interpretation guide")
print("\n" + "="*80)

log_file.close()

