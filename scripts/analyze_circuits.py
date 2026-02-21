#!/usr/bin/env python3

import json
import os
import sys
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)
OUTPUT_BASE = os.environ.get("LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out"))

def load_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def find_peak_layers(category_avg_l0):
    peak_info = {}
    
    for category, l0_dict in category_avg_l0.items():
        if not l0_dict:
            continue
        layers_values = [(int(k), v) for k, v in l0_dict.items()]
        layers_values.sort(key=lambda x: x[1], reverse=True)
        
        peak_layer, peak_value = layers_values[0]
        
        # Find top 3 layers
        top3_layers = [(layer, val) for layer, val in layers_values[:3]]
        
        peak_info[category] = {
            'peak_layer': peak_layer,
            'peak_value': peak_value,
            'top3_layers': top3_layers
        }
    
    return peak_info

def compute_layer_variance(category_avg_l0, num_layers=26):
    layer_variance = {}
    
    for layer_idx in range(num_layers):
        layer_str = str(layer_idx)
        values = []
        
        for category, l0_dict in category_avg_l0.items():
            if layer_str in l0_dict:
                values.append(l0_dict[layer_str])
        
        if len(values) > 1:
            variance = np.var(values)
            mean = np.mean(values)
            cv = variance / mean if mean > 0 else 0 
            layer_variance[layer_idx] = {
                'variance': variance,
                'mean': mean,
                'cv': cv,
                'n_categories': len(values)
            }
    
    return layer_variance

def identify_specialized_layers(layer_variance, threshold_percentile=75):

    variances = [info['variance'] for info in layer_variance.values()]
    if not variances:
        return {}, 0.0
    threshold = np.percentile(variances, threshold_percentile)
    
    specialized = {layer: info for layer, info in layer_variance.items() 
                   if info['variance'] >= threshold}
    
    return specialized, threshold

def compare_categories_pairwise(vul_samples, category1, category2):

    cat1_samples = [s for s in vul_samples if s.get('cwe', '') and 
                    categorize_cwe(s.get('cwe', '')) == category1]
    cat2_samples = [s for s in vul_samples if s.get('cwe', '') and 
                    categorize_cwe(s.get('cwe', '')) == category2]
    
    if not cat1_samples or not cat2_samples:
        return None

    all_layers = set()
    for s in cat1_samples + cat2_samples:
        all_layers.update(s.get('l0_per_layer', {}).keys())
    
    differences = {}
    for layer in sorted(all_layers, key=int):
        cat1_values = [s['l0_per_layer'].get(layer, 0) for s in cat1_samples 
                       if layer in s.get('l0_per_layer', {})]
        cat2_values = [s['l0_per_layer'].get(layer, 0) for s in cat2_samples 
                       if layer in s.get('l0_per_layer', {})]
        
        if cat1_values and cat2_values:
            mean1, mean2 = np.mean(cat1_values), np.mean(cat2_values)
            diff = abs(mean1 - mean2)
            relative_diff = diff / max(mean1, mean2) if max(mean1, mean2) > 0 else 0
            
            differences[layer] = {
                'cat1_mean': mean1,
                'cat2_mean': mean2,
                'abs_diff': diff,
                'relative_diff': relative_diff,
                'cat1_n': len(cat1_values),
                'cat2_n': len(cat2_values)
            }
    
    return differences

def categorize_cwe(cwe_str):
    """Same categorization as in prime3.py"""
    memory_cwes = ['CWE-119', 'CWE-120', 'CWE-121', 'CWE-122', 'CWE-125', 'CWE-126', 
                   'CWE-127', 'CWE-131', 'CWE-190', 'CWE-416', 'CWE-476', 'CWE-787',
                   'CWE-415']
    injection_cwes = ['CWE-89', 'CWE-78', 'CWE-79', 'CWE-94', 'CWE-95', 'CWE-77',
                     'CWE-772']
    logic_cwes = ['CWE-191', 'CWE-369', 'CWE-400', 'CWE-617', 'CWE-835',
                 'CWE-401', 'CWE-703', 'CWE-20']
    crypto_cwes = ['CWE-327', 'CWE-328', 'CWE-329', 'CWE-330', 'CWE-347']
    auth_cwes = ['CWE-287', 'CWE-306', 'CWE-862', 'CWE-863',
                'CWE-264', 'CWE-269', 'CWE-275', 'CWE-284']
    concurrency_cwes = ['CWE-362', 'CWE-366', 'CWE-367', 'CWE-820']
    info_leak_cwes = ['CWE-200', 'CWE-203', 'CWE-212', 'CWE-532']
    
    if any(cwe in cwe_str for cwe in memory_cwes):
        return "Memory Safety"
    elif any(cwe in cwe_str for cwe in injection_cwes):
        return "Injection"
    elif any(cwe in cwe_str for cwe in logic_cwes):
        return "Logic/Input Validation"
    elif any(cwe in cwe_str for cwe in crypto_cwes):
        return "Cryptography"
    elif any(cwe in cwe_str for cwe in auth_cwes):
        return "Auth/Access Control"
    elif any(cwe in cwe_str for cwe in concurrency_cwes):
        return "Concurrency"
    elif any(cwe in cwe_str for cwe in info_leak_cwes):
        return "Information Disclosure"
    else:
        return "Other"

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_circuits.py <path_to_out.json>")
        print("\nExample:")
        print("  python analyze_circuits.py ../log/20250906_122429/out.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    print(f"[INFO] Loading results from: {json_path}")
    
    data = load_results(json_path)

    vuln_categories = data.get('vulnerability_categories', {})
    category_counts = vuln_categories.get('category_counts', {})
    category_avg_l0 = vuln_categories.get('category_avg_l0', {})
    vul_samples = data.get('vul_samples', [])

    # ── Build category_avg_l0 from vul_samples + primevul JSONL if missing ──
    if not category_avg_l0 and vul_samples:
        vul_jsonl = os.path.join(ROOT_DIR, "data", "primevul236.jsonl")
        idx_to_cwe = {}
        if os.path.exists(vul_jsonl):
            with open(vul_jsonl, encoding='utf-8') as _f:
                for line in _f:
                    rec = json.loads(line)
                    idx_to_cwe[str(rec.get('idx', ''))] = rec.get('cwe', '')
        cat_l0_sums  = defaultdict(lambda: defaultdict(float))
        cat_counts_map = defaultdict(int)
        for s in vul_samples:
            cwe  = idx_to_cwe.get(str(s.get('idx', '')), '')
            cat  = categorize_cwe(cwe)
            l0   = s.get('l0_per_layer', {})
            if not l0:
                continue
            for layer, val in l0.items():
                cat_l0_sums[cat][str(layer)] += float(val)
            cat_counts_map[cat] += 1
        # Average
        for cat in cat_l0_sums:
            n = max(cat_counts_map[cat], 1)
            category_avg_l0[cat] = {k: v / n for k, v in cat_l0_sums[cat].items()}
        category_counts = dict(cat_counts_map)
        print(f"[INFO] Built category_avg_l0 for {len(category_avg_l0)} categories from {len(vul_samples)} vul_samples")

    # ── Output directory for plots ───────────────────────────────────────────
    from datetime import datetime as _dt
    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    PLOT_DIR = os.path.join(OUTPUT_BASE, "plots", f"analyze_circuits_{_ts}")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("VULNERABILITY-SPECIFIC CIRCUIT ANALYSIS")
    print("="*80)
    
    print("\n[1] PEAK ACTIVATION LAYERS PER CATEGORY")
    print("-" * 80)
    peak_info = find_peak_layers(category_avg_l0)
    
    for category in sorted(peak_info.keys()):
        info = peak_info[category]
        count = category_counts.get(category, 0)
        print(f"\n{category} (n={count}):")
        print(f"  Peak: Layer {info['peak_layer']} (L0={info['peak_value']:.2f})")
        print(f"  Top 3 layers: ", end="")
        top3_str = ", ".join([f"L{layer}({val:.1f})" for layer, val in info['top3_layers']])
        print(top3_str)
    
    print("\n\n[2] LAYER SPECIALIZATION ANALYSIS")
    print("-" * 80)
    layer_variance = compute_layer_variance(category_avg_l0)
    specialized, threshold = identify_specialized_layers(layer_variance, threshold_percentile=75)
    
    print(f"\nHighly specialized layers (variance > {threshold:.2f}):")
    for layer in sorted(specialized.keys()):
        info = specialized[layer]
        print(f"  Layer {layer:2d}: variance={info['variance']:6.2f}, "
              f"mean={info['mean']:5.2f}, CV={info['cv']:.3f}")

    early_layers = [l for l in specialized.keys() if l < 9]
    middle_layers = [l for l in specialized.keys() if 9 <= l < 17]
    late_layers = [l for l in specialized.keys() if l >= 17]
    
    print(f"\nSpecialized layer distribution:")
    print(f"  Early layers (0-8):   {len(early_layers)} specialized")
    print(f"  Middle layers (9-16): {len(middle_layers)} specialized")
    print(f"  Late layers (17-25):  {len(late_layers)} specialized")
    
    # 3. Pairwise category comparison
    print("\n\n[3] PAIRWISE CATEGORY COMPARISONS")
    print("-" * 80)

    if 'Memory Safety' in category_avg_l0 and 'Injection' in category_avg_l0:
        print("\nMemory Safety vs Injection:")
        diffs = compare_categories_pairwise(vul_samples, 'Memory Safety', 'Injection')
        if diffs:
            # Find top 5 layers with largest differences
            sorted_diffs = sorted(diffs.items(), key=lambda x: x[1]['abs_diff'], reverse=True)[:5]
            for layer, info in sorted_diffs:
                print(f"  Layer {layer:2s}: Memory={info['cat1_mean']:5.2f}, "
                      f"Injection={info['cat2_mean']:5.2f}, "
                      f"diff={info['abs_diff']:5.2f} ({info['relative_diff']*100:.1f}%)")

    if 'Memory Safety' in category_avg_l0 and 'Logic/Input Validation' in category_avg_l0:
        print("\nMemory Safety vs Logic/Input Validation:")
        diffs = compare_categories_pairwise(vul_samples, 'Memory Safety', 'Logic/Input Validation')
        if diffs:
            sorted_diffs = sorted(diffs.items(), key=lambda x: x[1]['abs_diff'], reverse=True)[:5]
            for layer, info in sorted_diffs:
                print(f"  Layer {layer:2s}: Memory={info['cat1_mean']:5.2f}, "
                      f"Logic={info['cat2_mean']:5.2f}, "
                      f"diff={info['abs_diff']:5.2f} ({info['relative_diff']*100:.1f}%)")

    print("\n\n[4] PUBLICATION SUMMARY")
    print("-" * 80)
    
    total_vul = sum(category_counts.values())
    print(f"\nDataset: {total_vul} vulnerable samples across {len(category_counts)} categories")
    
    print("\nKey findings:")

    early_peaks = {cat: info['peak_layer'] for cat, info in peak_info.items() 
                   if info['peak_layer'] < 9}
    late_peaks = {cat: info['peak_layer'] for cat, info in peak_info.items() 
                  if info['peak_layer'] >= 17}
    
    if early_peaks:
        early_cat = min(early_peaks.items(), key=lambda x: x[1])
        print(f"- {early_cat[0]} shows earliest peak activation (Layer {early_cat[1]}), "
              f"suggesting syntactic pattern matching")
    
    if late_peaks:
        late_cat = max(late_peaks.items(), key=lambda x: x[1])
        print(f"- {late_cat[0]} shows latest peak activation (Layer {late_cat[1]}), "
              f"suggesting semantic reasoning requirements")
    
    total_l0_per_cat = {cat: sum(l0_dict.values()) for cat, l0_dict in category_avg_l0.items()}
    if total_l0_per_cat:
        max_cat = max(total_l0_per_cat.items(), key=lambda x: x[1])
        print(f"- {max_cat[0]} has highest total L0 ({max_cat[1]:.1f}), "
              f"indicating most distributed circuit usage")
    
    # ── Generate and save plots ─────────────────────────────────────────────
    if category_avg_l0:
        # Plot 1: avg L0 per layer per category
        fig, ax = plt.subplots(figsize=(14, 6))
        n_layers = 26
        layers = list(range(n_layers))
        for cat, l0_dict in sorted(category_avg_l0.items()):
            vals = [l0_dict.get(str(l), 0) for l in layers]
            ax.plot(layers, vals, marker='o', markersize=3, label=f"{cat} (n={category_counts.get(cat,0)})")
        ax.set_xlabel('Layer')
        ax.set_ylabel('Avg L0 (attributed tokens)')
        ax.set_title('Average L0 Attribution per Layer by Vulnerability Category')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        p1 = os.path.join(PLOT_DIR, 'category_avg_l0_per_layer.png')
        fig.tight_layout()
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {p1}")

        # Plot 2: total L0 per category (bar chart)
        total_l0 = {cat: sum(l0_dict.values()) for cat, l0_dict in category_avg_l0.items()}
        cats_sorted = sorted(total_l0, key=lambda c: -total_l0[c])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(cats_sorted)), [total_l0[c] for c in cats_sorted], color='steelblue')
        ax.set_xticks(range(len(cats_sorted)))
        ax.set_xticklabels(cats_sorted, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Total avg L0')
        ax.set_title('Total Circuit Usage (avg L0) per Vulnerability Category')
        ax.grid(True, axis='y', alpha=0.3)
        p2 = os.path.join(PLOT_DIR, 'category_total_l0.png')
        fig.tight_layout()
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {p2}")
    else:
        print("[WARN] No category_avg_l0 data – plots skipped.")

    print("\n" + "="*80)
    print("[COMPLETE] Circuit analysis finished")
    print(f"[PLOTS]    {PLOT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
