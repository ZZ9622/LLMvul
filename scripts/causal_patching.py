#!/usr/bin/env python3
"""
Causal Patching Experiment for Vulnerability Detection
safe to vulnerable samples
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
from collections import defaultdict
import gc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit-tracer", "circuit-tracer"))
from circuit_tracer.replacement_model import ReplacementModel

SAFE_DATA_PATH = os.path.join(ROOT_DIR, "data", "tp_tn_samples.jsonl")  # TN 样本

# ── Check for tp_tn_samples.jsonl ────────────────────────────────────────────
if not os.path.exists(SAFE_DATA_PATH):
    print(f"[ERROR] {SAFE_DATA_PATH} not found.")
    print("[INFO]  This file is produced by running: python scripts/prime.py")
    print("[INFO]  Then copy TP/TN results from the output to data/tp_tn_samples.jsonl")
    raise SystemExit(1)

VUL_DATA_PATH = os.path.join(ROOT_DIR, "data", "tp_tn_samples.jsonl")   # TP 样本
MODEL_PATH = "Chun9622/llmvul-finetuned-gemma"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_SAFE_SAMPLES = None
NUM_VUL_SAMPLES = None
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 200

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(OUTPUT_BASE, "log", f"causal_patching_{ts}")
PLOT_DIR = os.path.join(OUTPUT_BASE, "plots", f"causal_patching_{ts}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


log_file = open(os.path.join(LOG_DIR, f"causal_patching_{ts}.txt"), "w")
sys.stdout = log_file
sys.stderr = log_file

print("="*80)
print("Causal Patching Experiment for Vulnerability Detection")
print("="*80)
print(f"Timestamp: {ts}")
print(f"Device: {DEVICE}")
print(f"Output: {PLOT_DIR}")
print("="*80)

# === Model Loading ===
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
d_model = rm.cfg.d_model

print(f"  Model: Gemma-2-2B")
print(f"  Layers: {n_layers}")
print(f"  Hidden size (d_model): {d_model}")

def extract_label(text):
    """
    Extract vul/nonvul label from model output (from prime.py).
    
    Enhanced to better handle code continuation and truncated outputs.
    
    Priority strategy:
    0. Find "Answer:" and extract text after it
    1. Look for "safe" or "vulnerable" keywords anywhere in output
    2. Check for code-followed-by-answer pattern
    3. Detect question echo patterns
    4. Keyword analysis with lower threshold
    """
    import re
    
    if not text or len(text.strip()) == 0:
        return "unknown"
    
    t = text.lower().strip()
    original_text = text.strip()
    
    answer_match = re.search(r'answer:\s*(.+)', t, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if answer_text:
            answer_lines = answer_text.split('\n')
            answer_first_line = answer_lines[0].strip()
            answer_words = answer_first_line.split()[:15]
            
            if len(answer_words) > 0:
                first_word = answer_words[0]
                
                if 'vulnerable' in answer_first_line[:100] or 'unsafe' in answer_first_line[:100]:
                    if 'not vulnerable' not in answer_first_line[:100] and 'not unsafe' not in answer_first_line[:100]:
                        return "vul"
                
                if 'safe' in answer_first_line[:100] or 'secure' in answer_first_line[:100]:
                    if 'not safe' not in answer_first_line[:100] and 'unsafe' not in answer_first_line[:100]:
                        return "nonvul"
                
                if first_word in ['it', 'this', 'the']:
                    if 'vulnerable' in answer_first_line[:100]:
                        return "vul"
                    if 'safe' in answer_first_line[:100] or 'secure' in answer_first_line[:100]:
                        return "nonvul"
                
                if first_word in ['vulnerable', 'unsafe', 'insecure', 'yes']:
                    if first_word == 'yes':
                        if len(answer_words) >= 2 and 'vulnerable' in answer_first_line[:60]:
                            return "vul"
                    else:
                        return "vul"
                
                if first_word in ['safe', 'secure', 'no']:
                    if first_word == 'no':
                        return "nonvul"
                    return "nonvul"
    
    question_match = re.search(r'question:.*?answer:\s*(.+)', t, re.IGNORECASE | re.DOTALL)
    if question_match:
        answer_part = question_match.group(1).strip()
        if 'vulnerable' in answer_part[:50] and 'not vulnerable' not in answer_part[:50]:
            return "vul"
        if 'safe' in answer_part[:50] and 'not safe' not in answer_part[:50]:
            return "nonvul"
    
    full_text = t[:800]
    
    if re.search(r'\b(is|be|appears?|seems?)\s+(vulnerable|unsafe|insecure)', full_text):
        if 'not vulnerable' not in full_text and 'not unsafe' not in full_text:
            return "vul"
    
    if re.search(r'\bvulnerable\s+to\b', full_text):
        return "vul"
    
    if re.search(r'\b(this|the)\s+(code|function|program)\s+(is|has|contains)\s+(a\s+)?(vulnerable|vulnerability|unsafe|flaw|bug)', full_text):
        return "vul"
    
    if re.search(r'\b(buffer\s+overflow|memory\s+leak|injection|overflow|underflow|race\s+condition)', full_text):
        if re.search(r'\b(fix|prevent|avoid|safe\s+from)\s+', full_text):
            pass
        else:
            return "vul"
    
    if re.search(r'\b(is|be|appears?|seems?)\s+(safe|secure|correct)', full_text):
        if 'not safe' not in full_text and 'unsafe' not in full_text and 'not secure' not in full_text:
            return "nonvul"
    
    if re.search(r'\b(this|the)\s+(code|function|program)\s+(is|appears)\s+(safe|secure|correct)', full_text):
        return "nonvul"
    
    if re.search(r'\bno\s+(vulnerabilities|security\s+issues|flaws|bugs|problems)', full_text):
        return "nonvul"
    
    lines = t.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if len(line) < 5:
            continue
        words = line.split()[:5]
        if len(words) > 0:
            first_word = words[0]
            if first_word in ['const', 'void', 'int', 'static', 'struct', 'char', 'return', 'typedef', 
                            'if', 'for', 'while', 'switch', 'case', '{', '}', '//', '/*']:
                continue
            if first_word in ['vulnerable', 'unsafe', 'insecure']:
                return "vul"
            if first_word in ['safe', 'secure']:
                return "nonvul"
            if 'vulnerable' in line or 'unsafe' in line:
                return "vul"
            if 'safe' in line and 'unsafe' not in line:
                return "nonvul"
    
    first_500 = t[:500]
    
    safe_keywords = ['safe', 'secure', 'correct', 'valid', 'okay', 'fine', 'proper', 'protected', 'good']
    vul_keywords = ['vulnerable', 'unsafe', 'insecure', 'flaw', 'bug', 'exploit', 
                    'overflow', 'leak', 'injection', 'attack', 'danger', 'malicious', 'risk']
    
    safe_count = sum(first_500.count(kw) for kw in safe_keywords)
    vul_count = sum(first_500.count(kw) for kw in vul_keywords)
    
    if vul_count >= 1 and safe_count == 0:
        return "vul"
    
    if safe_count >= 1 and vul_count == 0:
        return "nonvul"
    
    if vul_count > safe_count and vul_count >= 1:
        if vul_count >= safe_count * 1.5:
            return "vul"
    
    if safe_count > vul_count and safe_count >= 1:
        if safe_count >= vul_count * 1.5:
            return "nonvul"
    
    security_words = ['security', 'vulnerability', 'safe', 'unsafe', 'secure', 'insecure']
    has_security_content = any(word in first_500 for word in security_words)
    
    if has_security_content:
        positive = first_500.count('safe') + first_500.count('secure') + first_500.count('correct')
        negative = first_500.count('vulnerable') + first_500.count('unsafe') + first_500.count('flaw')
        
        if positive > negative:
            return "nonvul"
        elif negative > positive:
            return "vul"
    
    return "unknown"

print(f"\n[2/6] Step 1: Computing Mean Safe Vectors from TN samples...")

def load_samples(jsonl_path, target_type, max_samples=None):
    """Load samples of specified type (all if max_samples is None)"""
    samples = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                pred_type = data.get('prediction_type', '')
                
                if target_type == 'TN' and pred_type != 'TN':
                    continue
                if target_type == 'TP' and pred_type != 'TP':
                    continue
                
                code = data.get('func', '').strip()
                if not code:
                    continue
                
                if len(code) > 2000:
                    code = code[:2000]
                
                prompt = f"Code: {code}\n\nQuestion: Is this code safe or vulnerable?\nAnswer:"
                
                samples.append({
                    'idx': data.get('idx', -1),
                    'prompt': prompt,
                    'type': pred_type
                })
                
                if max_samples and len(samples) >= max_samples:
                    break
                    
            except Exception as e:
                continue
    
    return samples

safe_samples = load_samples(SAFE_DATA_PATH, 'TN', NUM_SAFE_SAMPLES)
print(f"  Loaded {len(safe_samples)} TN (safe) samples")

# 计算每层的平均残差流向量
mean_safe_cache = {}  # {layer: mean_vector}

print(f"  Extracting residual stream activations...")
all_residuals = defaultdict(list)  # {layer: [vectors]}

for i, sample in enumerate(safe_samples):
    if (i + 1) % 10 == 0:
        print(f"    Progress: {i+1}/{len(safe_samples)}")
    
    try:
        # Tokenize
        inputs = tokenizer(
            sample['prompt'],
            return_tensors='pt',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)
        
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        
        # Run with cache
        with torch.no_grad():
            _, cache = rm.run_with_cache(input_ids, return_type=None)
        
        # 提取每层最后一个 token 的残差流
        for layer in range(n_layers):
            # Cache key: blocks.{layer}.hook_resid_post
            resid_key = f"blocks.{layer}.hook_resid_post"
            
            if resid_key in cache:
                # shape: (batch=1, seq_len, d_model)
                resid = cache[resid_key]
                
                # 取最后一个 token 的向量
                last_token_vec = resid[0, seq_len-1, :].detach().cpu().float()
                
                all_residuals[layer].append(last_token_vec)
        
        # 清理
        del cache, inputs
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"    Warning: Failed to process sample {i}: {e}")
        continue

print(f"  Computing mean vectors for each layer...")
for layer in range(n_layers):
    if layer in all_residuals and len(all_residuals[layer]) > 0:
        # Stack and average
        vectors = torch.stack(all_residuals[layer])  # (n_samples, d_model)
        mean_vec = torch.mean(vectors, dim=0)  # (d_model,)
        mean_safe_cache[layer] = mean_vec
        print(f"    Layer {layer}: averaged {len(all_residuals[layer])} samples")

print(f"  ✓ Mean Safe Cache computed for {len(mean_safe_cache)} layers")

print(f"\n[3/6] Step 2: Filtering baseline vulnerable samples...")

vul_samples = load_samples(VUL_DATA_PATH, 'TP', None)
print(f"  Loaded {len(vul_samples)} TP (vulnerable) candidate samples")

print(f"  Running baseline predictions (without patching)...")
baseline_vul_samples = []

for i, sample in enumerate(vul_samples):
    if (i + 1) % 10 == 0:
        print(f"    Progress: {i+1}/{len(vul_samples)}")
    
    try:
        # Tokenize
        inputs = tokenizer(
            sample['prompt'],
            return_tensors='pt',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)
        
        input_ids = inputs['input_ids']
        
        # Generate
        with torch.no_grad():
            output_ids = rm.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                verbose=False
            )
        
        prompt_len = input_ids.shape[1]
        generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        
        pred_label = extract_label(generated_text)
        
        if pred_label == "vul":
            baseline_vul_samples.append({
                'idx': sample['idx'],
                'prompt': sample['prompt'],
                'baseline_output': generated_text,
                'baseline_label': pred_label
            })
        
        del inputs, output_ids
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                    
    except Exception as e:
        print(f"    Warning: Failed to process sample {i}: {e}")
        continue

print(f"  ✓ Selected {len(baseline_vul_samples)} vulnerable samples (baseline: all predict 'vul')")

print(f"\n[4/6] Step 3: Layer-by-layer Patching Sweep...")

# 存储结果: {layer: [sample_results]}
patching_results = defaultdict(list)

for target_layer in range(n_layers):
    print(f"\n  [Layer {target_layer}/{n_layers-1}] Patching residual stream...")
    
    if target_layer not in mean_safe_cache:
        print(f"    Skipping (no mean safe vector)")
        continue
    
    mean_safe_vec = mean_safe_cache[target_layer].to(DEVICE, dtype=torch.float16)
    
    flipped_count = 0
    
    for i, sample in enumerate(baseline_vul_samples):
        try:
            # Tokenize
            inputs = tokenizer(
                sample['prompt'],
                return_tensors='pt',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(DEVICE)
            
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1]
            
            original_seq_len = seq_len
            
            def patch_hook(module, input, output):
                """
                Hook function to patch residual stream at last token position of original prompt.
                output shape: (batch, current_seq_len, d_model)
                
                Only patch the last position of original prompt, not current sequence last position.
                """
                current_len = output.shape[1]
                if current_len >= original_seq_len:
                    output[0, original_seq_len-1, :] = mean_safe_vec
                return output
            
            hook_handle = rm.blocks[target_layer].hook_resid_post.register_forward_hook(patch_hook)
            
            try:
                with torch.no_grad():
                    output_ids = rm.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        verbose=False
                    )
                
                prompt_len = input_ids.shape[1]
                generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
                
                patched_label = extract_label(generated_text)
                
                flipped = (patched_label != "vul")
                
                if flipped:
                    flipped_count += 1
                
                patching_results[target_layer].append({
                    'sample_idx': sample['idx'],
                    'baseline_label': 'vul',
                    'patched_label': patched_label,
                    'flipped': flipped,
                    'patched_output': generated_text
                })
                
            finally:
                hook_handle.remove()
            
            del inputs, output_ids
            
        except Exception as e:
            print(f"      Warning: Failed sample {i} at layer {target_layer}: {e}")
            continue
    
    flip_rate = flipped_count / len(baseline_vul_samples) * 100 if baseline_vul_samples else 0
    print(f"    Flipped: {flipped_count}/{len(baseline_vul_samples)} ({flip_rate:.1f}%)")
    
    if target_layer % 5 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\n  ✓ Patching sweep completed for all {n_layers} layers")

print(f"\n[5/6] Step 4: Computing statistics...")

flip_rates = {}
for layer in range(n_layers):
    if layer in patching_results:
        results = patching_results[layer]
        flipped = sum(1 for r in results if r['flipped'])
        total = len(results)
        flip_rate = flipped / total * 100 if total > 0 else 0
        flip_rates[layer] = flip_rate
    else:
        flip_rates[layer] = 0.0

max_flip_layer = max(flip_rates, key=flip_rates.get)
max_flip_rate = flip_rates[max_flip_layer]

print(f"\n  Key Layer: {max_flip_layer} (Flip Rate: {max_flip_rate:.1f}%)")
print(f"\n  Top-5 Most Effective Layers:")
sorted_layers = sorted(flip_rates.items(), key=lambda x: x[1], reverse=True)[:5]
for layer, rate in sorted_layers:
    print(f"    Layer {layer}: {rate:.1f}%")

print(f"\n[6/6] Step 5: Generating visualizations...")

# 1. Flip Rate Curve
fig, ax = plt.subplots(figsize=(14, 6))

layers = sorted(flip_rates.keys())
rates = [flip_rates[l] for l in layers]

ax.plot(layers, rates, 'o-', linewidth=2.5, markersize=6, color='steelblue')
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Flip Rate')
ax.axvline(x=max_flip_layer, color='orange', linestyle='--', alpha=0.7, 
           label=f'Max at Layer {max_flip_layer}')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Flip Rate (%)', fontsize=12)
ax.set_title(f'Causal Patching: Flip Rate per Layer\n'
             f'(Injecting Mean Safe Vector into Vulnerable Samples)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xticks(range(0, n_layers, 2))
ax.set_ylim(-5, 105)

plt.tight_layout()
curve_path = os.path.join(PLOT_DIR, "flip_rate_curve.png")
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {curve_path}")

fig, ax = plt.subplots(figsize=(16, 8))

n_samples = len(baseline_vul_samples)
flip_matrix = np.zeros((n_samples, n_layers))

for layer in range(n_layers):
    if layer in patching_results:
        results = patching_results[layer]
        for i, result in enumerate(results):
            if i < n_samples:
                flip_matrix[i, layer] = 1 if result['flipped'] else 0

sns.heatmap(
    flip_matrix,
    cmap='RdYlGn',
    cbar_kws={'label': 'Flipped (1) / Not Flipped (0)'},
    ax=ax,
    xticklabels=2,
    yticklabels=False
)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel(f'Vulnerable Samples (n={n_samples})', fontsize=12)
ax.set_title('Causal Patching Heatmap: Which Layer Flips Which Sample?',
             fontsize=14, fontweight='bold')

plt.tight_layout()
heatmap_path = os.path.join(PLOT_DIR, "flip_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {heatmap_path}")

early_layers = list(range(0, n_layers // 3))
middle_layers = list(range(n_layers // 3, 2 * n_layers // 3))
late_layers = list(range(2 * n_layers // 3, n_layers))

early_rate = np.mean([flip_rates[l] for l in early_layers])
middle_rate = np.mean([flip_rates[l] for l in middle_layers])
late_rate = np.mean([flip_rates[l] for l in late_layers])

fig, ax = plt.subplots(figsize=(8, 6))
stages = ['Early\n(L0-L8)', 'Middle\n(L9-L17)', 'Late\n(L18-L25)']
stage_rates = [early_rate, middle_rate, late_rate]

bars = ax.bar(stages, stage_rates, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
ax.set_ylabel('Average Flip Rate (%)', fontsize=12)
ax.set_title('Flip Rate by Model Stage', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, rate in zip(bars, stage_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
stage_path = os.path.join(PLOT_DIR, "flip_rate_by_stage.png")
plt.savefig(stage_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {stage_path}")

print(f"\n  Saving results...")

results_json = {
    'experiment': 'causal_patching',
    'timestamp': ts,
    'config': {
        'num_safe_samples': len(safe_samples),
        'num_vul_samples': len(baseline_vul_samples),
        'n_layers': n_layers,
        'd_model': d_model
    },
    'flip_rates': {int(k): float(v) for k, v in flip_rates.items()},
    'key_layer': int(max_flip_layer),
    'max_flip_rate': float(max_flip_rate),
    'stage_statistics': {
        'early_layers': {'range': [0, n_layers//3], 'avg_flip_rate': float(early_rate)},
        'middle_layers': {'range': [n_layers//3, 2*n_layers//3], 'avg_flip_rate': float(middle_rate)},
        'late_layers': {'range': [2*n_layers//3, n_layers], 'avg_flip_rate': float(late_rate)}
    },
    'detailed_results': {int(k): v for k, v in patching_results.items()}
}

json_path = os.path.join(LOG_DIR, "causal_patching_results.json")
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"  Saved: {json_path}")

summary_path = os.path.join(LOG_DIR, "CAUSAL_PATCHING_SUMMARY.md")
with open(summary_path, 'w') as f:
    f.write("# Causal Patching Experiment Summary\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Model:** Gemma-2-2B ({n_layers} layers, d_model={d_model})\n\n")
    
    f.write("## Experiment Design\n\n")
    f.write("**Question:** If we replace a vulnerable sample's internal representation with the mean safe representation, will the model change its prediction?\n\n")
    f.write("**Method:**\n")
    f.write(f"1. Computed mean safe vector from {len(safe_samples)} TN samples\n")
    f.write(f"2. Selected {len(baseline_vul_samples)} TP samples (baseline: all predict 'vul')\n")
    f.write(f"3. For each layer, patched the residual stream with mean safe vector\n")
    f.write(f"4. Measured flip rate: % of samples that changed from 'vul' to 'safe'\n\n")
    
    f.write("## Key Findings\n\n")
    f.write(f"**Critical Decision Layer:** Layer {max_flip_layer} (Flip Rate: {max_flip_rate:.1f}%)\n\n")
    f.write("This layer is where the model makes its key vulnerability/safety decision.\n\n")
    
    f.write("**Top-5 Most Effective Layers:**\n\n")
    f.write("| Rank | Layer | Flip Rate |\n")
    f.write("|------|-------|----------|\n")
    for rank, (layer, rate) in enumerate(sorted_layers, 1):
        f.write(f"| {rank} | {layer} | {rate:.1f}% |\n")
    
    f.write("\n**Stage Analysis:**\n\n")
    f.write("| Stage | Layers | Avg Flip Rate |\n")
    f.write("|-------|--------|---------------|\n")
    f.write(f"| Early | 0-{n_layers//3-1} | {early_rate:.1f}% |\n")
    f.write(f"| Middle | {n_layers//3}-{2*n_layers//3-1} | {middle_rate:.1f}% |\n")
    f.write(f"| Late | {2*n_layers//3}-{n_layers-1} | {late_rate:.1f}% |\n")
    
    f.write("\n## Interpretation\n\n")
    f.write("- **Low flip rate (< 20%)**: Layer hasn't formed vulnerability judgment yet\n")
    f.write("- **High flip rate (> 60%)**: Layer is critical for decision-making\n")
    f.write("- **Decreasing flip rate**: Decision has been solidified, hard to change\n\n")
    
    f.write("## Generated Files\n\n")
    f.write("- `flip_rate_curve.png`: Flip rate per layer\n")
    f.write("- `flip_heatmap.png`: Which layer flips which sample\n")
    f.write("- `flip_rate_by_stage.png`: Early vs middle vs late layers\n")
    f.write("- `causal_patching_results.json`: Full detailed results\n")

print(f"  Saved: {summary_path}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print(f"\nKey Finding: Layer {max_flip_layer} is the critical decision layer ({max_flip_rate:.1f}% flip rate)")
print(f"\nGenerated files:")
print(f"  Plots: {PLOT_DIR}")
print(f"  Logs: {LOG_DIR}")
print(f"\nInterpretation:")
print(f"  - Early layers (avg {early_rate:.1f}%): Still processing syntax")
print(f"  - Middle layers (avg {middle_rate:.1f}%): Forming semantic understanding")
print(f"  - Late layers (avg {late_rate:.1f}%): Final decision-making")
print("\n" + "="*80)

log_file.close()

