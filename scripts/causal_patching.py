#!/usr/bin/env python3
"""
Bidirectional Steering Experiment for Vulnerability Detection

Two-way steering intervention using vector addition:
1. Safe → Vulnerable: Steer vulnerable samples toward safe direction
2. Vulnerable → Safe: Steer safe samples toward vulnerable direction

Method: output = original + coeff * steering_vector
where steering_vector = Mean_Vuln - Mean_Safe

This validates which layers are truly critical for decision-making in both directions.
"""

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)
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

SAFE_DATA_PATH = os.path.join(ROOT_DIR, "data", "tp_tn_samples.jsonl")

# please use the tp_tn_samples.jsonl which is from the dataset used in prime.py, to ensure the prompt format and samples are consistent with the main pipeline.

VUL_DATA_PATH = os.path.join(ROOT_DIR, "data", "tp_tn_samples.jsonl")
MODEL_PATH = "Chun9622/llmvul-finetuned-gemma"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_SAFE_SAMPLES = None
NUM_VUL_SAMPLES = None
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 200
STEERING_COEFF = 8  # Coefficient for steering vector

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(OUTPUT_BASE, "log", f"causal_patching_{ts}")
PLOT_DIR = os.path.join(OUTPUT_BASE, "plots", f"causal_patching_{ts}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


log_file = open(os.path.join(LOG_DIR, f"causal_patching_{ts}.txt"), "w")
sys.stdout = log_file
sys.stderr = log_file

print("="*80)
print("Bidirectional Steering Experiment (Vector Addition)")
print("="*80)
print(f"Method: output = original + coeff × (Mean_Vuln - Mean_Safe)")
print(f"Steering Coefficient: {STEERING_COEFF}")
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

print(f"\nLoading Model & Tokenizer...")
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

print(f"  Layers: {n_layers}")
print(f"  Hidden size (d_model): {d_model}")

def extract_label(text):
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

print(f"\nComputing Mean Safe Vectors from TN samples...")

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
mean_safe_cache = {}  
print(f"  Extracting residual stream activations...")
all_residuals = defaultdict(list)  

for i, sample in enumerate(safe_samples):
    if (i + 1) % 10 == 0:
        print(f"    Progress: {i+1}/{len(safe_samples)}")
    
    try:
        inputs = tokenizer(
            sample['prompt'],
            return_tensors='pt',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)
        
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            _, cache = rm.run_with_cache(input_ids, return_type=None)
        
        for layer in range(n_layers):
            resid_key = f"blocks.{layer}.hook_resid_post"
            
            if resid_key in cache:
                resid = cache[resid_key]
                last_token_vec = resid[0, seq_len-1, :].detach().cpu().float()
                
                all_residuals[layer].append(last_token_vec)
        
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
        vectors = torch.stack(all_residuals[layer])  
        mean_vec = torch.mean(vectors, dim=0)  
        mean_safe_cache[layer] = mean_vec
        print(f"    Layer {layer}: averaged {len(all_residuals[layer])} samples")

print(f"  Mean Safe Cache computed for {len(mean_safe_cache)} layers")

print(f"\nComputing Mean Vulnerable Vectors from TP samples...")

vul_samples_all = load_samples(VUL_DATA_PATH, 'TP', NUM_VUL_SAMPLES)
print(f"  Loaded {len(vul_samples_all)} TP (vulnerable) samples")
mean_vul_cache = {}
print(f"  Extracting residual stream activations...")
all_residuals_vul = defaultdict(list)

for i, sample in enumerate(vul_samples_all):
    if (i + 1) % 10 == 0:
        print(f"    Progress: {i+1}/{len(vul_samples_all)}")
    
    try:
        inputs = tokenizer(
            sample['prompt'],
            return_tensors='pt',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)
        
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            _, cache = rm.run_with_cache(input_ids, return_type=None)
        
        for layer in range(n_layers):
            resid_key = f"blocks.{layer}.hook_resid_post"
            if resid_key in cache:
                resid = cache[resid_key]
                last_token_vec = resid[0, seq_len-1, :].detach().cpu().float()
                all_residuals_vul[layer].append(last_token_vec)
        
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
    if layer in all_residuals_vul and len(all_residuals_vul[layer]) > 0:
        vectors = torch.stack(all_residuals_vul[layer])
        mean_vec = torch.mean(vectors, dim=0)
        mean_vul_cache[layer] = mean_vec
        print(f"    Layer {layer}: averaged {len(all_residuals_vul[layer])} samples")

print(f"  Mean Vulnerable Cache computed for {len(mean_vul_cache)} layers")

print(f"\nFiltering baseline vulnerable samples (predict 'vul')...")

def filter_baseline_samples(samples, target_label):
    """Filter samples where model predicts target_label correctly"""
    print(f"  Running baseline predictions (target label: {target_label})...")
    baseline_samples = []
    
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(samples)}")
        
        try:
            inputs = tokenizer(
                sample['prompt'],
                return_tensors='pt',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(DEVICE)
            
            input_ids = inputs['input_ids']
            
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
            
            if pred_label == target_label:
                baseline_samples.append({
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
    
    return baseline_samples

baseline_vul_samples = filter_baseline_samples(vul_samples_all, 'vul')
print(f"  Selected {len(baseline_vul_samples)} vulnerable samples (baseline: all predict 'vul')")

print(f"\nFiltering baseline safe samples (predict 'nonvul')...")
baseline_safe_samples = filter_baseline_samples(safe_samples, 'nonvul')
print(f"  Selected {len(baseline_safe_samples)} safe samples (baseline: all predict 'nonvul')")

print(f"\nComputing Steering Vectors...")
print(f"  Steering Vector = Mean_Vuln - Mean_Safe")
print(f"  Steering Coefficient = {STEERING_COEFF}")

steering_cache = {}
for layer in range(n_layers):
    if layer in mean_vul_cache and layer in mean_safe_cache:
        vul_vec = mean_vul_cache[layer]
        safe_vec = mean_safe_cache[layer]
        steering_vec = vul_vec - safe_vec
        steering_cache[layer] = steering_vec
        print(f"    Layer {layer}: ||steering|| = {torch.norm(steering_vec).item():.4f}")
    else:
        print(f"    Layer {layer}: skipping (missing vector)")

print(f"  Computed steering vectors for {len(steering_cache)} layers")

def steering_and_test(baseline_samples, steering_vector_cache, original_label, direction_name, steer_sign):
    """Run steering sweep for one direction.

    steer_sign: -1 to steer toward safe (flip vul→safe), +1 to steer toward vul (flip safe→vul)
    """
    print(f"\n[{direction_name}] Layer-by-layer Steering Sweep (coeff={STEERING_COEFF}, sign={steer_sign:+d})...")
    
    patching_results = defaultdict(list)
    
    for target_layer in range(n_layers):
        print(f"\n  [Layer {target_layer}/{n_layers-1}] Applying steering vector...")
        
        if target_layer not in steering_vector_cache:
            print(f"    Skipping (no steering vector)")
            continue
        
        steering_vec = steering_vector_cache[target_layer].to(DEVICE, dtype=torch.float16)
        steering_vec = steer_sign * STEERING_COEFF * steering_vec
        
        flipped_count = 0
        
        for i, sample in enumerate(baseline_samples):
            try:
                inputs = tokenizer(
                    sample['prompt'],
                    return_tensors='pt',
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH
                ).to(DEVICE)
                
                input_ids = inputs['input_ids']
                seq_len = input_ids.shape[1]
                original_seq_len = seq_len
                
                def steering_hook(module, input, output):
                    """Steering: output = original + coeff * steering_vector"""
                    current_len = output.shape[1]
                    if current_len >= original_seq_len:
                        output[0, original_seq_len-1, :] = output[0, original_seq_len-1, :] + steering_vec
                    return output
                
                hook_handle = rm.blocks[target_layer].hook_resid_post.register_forward_hook(steering_hook)
                
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
                    flipped = (patched_label != original_label)
                    
                    if flipped:
                        flipped_count += 1
                    
                    patching_results[target_layer].append({
                        'sample_idx': sample['idx'],
                        'baseline_label': original_label,
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
        
        flip_rate = flipped_count / len(baseline_samples) * 100 if baseline_samples else 0
        print(f"    Flipped: {flipped_count}/{len(baseline_samples)} ({flip_rate:.1f}%)")
        
        if target_layer % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return patching_results

print(f"\nDirection 1 – Safe → Vulnerable")
print("  (Steering vulnerable samples toward safe: output = orig - coeff * steering)")
safe_to_vul_results = steering_and_test(baseline_vul_samples, steering_cache, 'vul', 'S→V', steer_sign=-1)

print(f"\nDirection 2 – Vulnerable → Safe")
print("  (Steering safe samples toward vulnerable: output = orig + coeff * steering)")
vul_to_safe_results = steering_and_test(baseline_safe_samples, steering_cache, 'nonvul', 'V→S', steer_sign=+1)

print(f"\nComputing statistics and generating visualizations...")

flip_rates_s2v = {}
flip_rates_v2s = {}

for layer in range(n_layers):
    if layer in safe_to_vul_results:
        results = safe_to_vul_results[layer]
        flipped = sum(1 for r in results if r['flipped'])
        total = len(results)
        flip_rates_s2v[layer] = flipped / total * 100 if total > 0 else 0.0
    else:
        flip_rates_s2v[layer] = 0.0
    
    if layer in vul_to_safe_results:
        results = vul_to_safe_results[layer]
        flipped = sum(1 for r in results if r['flipped'])
        total = len(results)
        flip_rates_v2s[layer] = flipped / total * 100 if total > 0 else 0.0
    else:
        flip_rates_v2s[layer] = 0.0

max_flip_layer_s2v = max(flip_rates_s2v, key=flip_rates_s2v.get)
max_flip_rate_s2v = flip_rates_s2v[max_flip_layer_s2v]

max_flip_layer_v2s = max(flip_rates_v2s, key=flip_rates_v2s.get)
max_flip_rate_v2s = flip_rates_v2s[max_flip_layer_v2s]

print(f"\n  Direction 1 (Safe → Vulnerable):")
print(f"    Key Layer: {max_flip_layer_s2v} (Flip Rate: {max_flip_rate_s2v:.1f}%)")
print(f"\n  Direction 2 (Vulnerable → Safe):")
print(f"    Key Layer: {max_flip_layer_v2s} (Flip Rate: {max_flip_rate_v2s:.1f}%)")

print(f"\n  Generating visualizations...")

# plot directional flip rate curve
fig, ax = plt.subplots(figsize=(14, 6))

layers = sorted(flip_rates_s2v.keys())
rates_s2v = [flip_rates_s2v[l] for l in layers]
rates_v2s = [flip_rates_v2s[l] for l in layers]

ax.plot(layers, rates_s2v, 'o-', linewidth=2.5, markersize=6, color='steelblue',
        label='Safe → Vulnerable (steer vul toward safe)')
ax.plot(layers, rates_v2s, 's-', linewidth=2.5, markersize=6, color='coral',
        label='Vulnerable → Safe (steer safe toward vul)')

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=max_flip_layer_s2v, color='steelblue', linestyle='--', alpha=0.5,
           label=f'S→V max at L{max_flip_layer_s2v}')
ax.axvline(x=max_flip_layer_v2s, color='coral', linestyle='--', alpha=0.5,
           label=f'V→S max at L{max_flip_layer_v2s}')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Flip Rate (%)', fontsize=12)
ax.set_title(f'Bidirectional Steering: Flip Rate per Layer (coeff={STEERING_COEFF})',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xticks(range(0, n_layers, 2))
ax.set_ylim(-5, 105)

plt.tight_layout()
curve_path = os.path.join(PLOT_DIR, "flip_rate_bidirectional.png")
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {curve_path}")

# plot flip rate by stage (bidirectional grouped bar)
early_layers = list(range(0, n_layers // 3))
middle_layers = list(range(n_layers // 3, 2 * n_layers // 3))
late_layers = list(range(2 * n_layers // 3, n_layers))

early_rate_s2v  = np.mean([flip_rates_s2v[l] for l in early_layers])
middle_rate_s2v = np.mean([flip_rates_s2v[l] for l in middle_layers])
late_rate_s2v   = np.mean([flip_rates_s2v[l] for l in late_layers])

early_rate_v2s  = np.mean([flip_rates_v2s[l] for l in early_layers])
middle_rate_v2s = np.mean([flip_rates_v2s[l] for l in middle_layers])
late_rate_v2s   = np.mean([flip_rates_v2s[l] for l in late_layers])

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.35

bars1 = ax.bar(x - width/2, [early_rate_s2v, middle_rate_s2v, late_rate_s2v],
               width, label='Safe → Vulnerable', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, [early_rate_v2s, middle_rate_v2s, late_rate_v2s],
               width, label='Vulnerable → Safe', color='coral', alpha=0.8)

ax.set_ylabel('Average Flip Rate (%)', fontsize=12)
ax.set_title(f'Flip Rate by Model Stage (Bidirectional Steering, coeff={STEERING_COEFF})',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Early\n(L0-L8)', 'Middle\n(L9-L17)', 'Late\n(L18-L25)'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
stage_path = os.path.join(PLOT_DIR, "flip_rate_by_stage.png")
plt.savefig(stage_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {stage_path}")

print(f"\n  Saving results...")

results_json = {
    'experiment': 'bidirectional_steering',
    'method': 'vector_addition',
    'formula': 'output = original + coeff * (Mean_Vuln - Mean_Safe)',
    'timestamp': ts,
    'config': {
        'steering_coefficient': STEERING_COEFF,
        'num_safe_samples': len(safe_samples),
        'num_vul_samples': len(vul_samples_all),
        'num_baseline_safe': len(baseline_safe_samples),
        'num_baseline_vul': len(baseline_vul_samples),
        'n_layers': n_layers,
        'd_model': d_model
    },
    'direction_1_safe_to_vul': {
        'flip_rates': {int(k): float(v) for k, v in flip_rates_s2v.items()},
        'key_layer': int(max_flip_layer_s2v),
        'max_flip_rate': float(max_flip_rate_s2v),
        'stage_statistics': {
            'early': float(early_rate_s2v),
            'middle': float(middle_rate_s2v),
            'late': float(late_rate_s2v)
        },
        'detailed_results': {int(k): v for k, v in safe_to_vul_results.items()}
    },
    'direction_2_vul_to_safe': {
        'flip_rates': {int(k): float(v) for k, v in flip_rates_v2s.items()},
        'key_layer': int(max_flip_layer_v2s),
        'max_flip_rate': float(max_flip_rate_v2s),
        'stage_statistics': {
            'early': float(early_rate_v2s),
            'middle': float(middle_rate_v2s),
            'late': float(late_rate_v2s)
        },
        'detailed_results': {int(k): v for k, v in vul_to_safe_results.items()}
    }
}

json_path = os.path.join(LOG_DIR, "causal_patching_results.json")
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"  Saved: {json_path}")

summary_path = os.path.join(LOG_DIR, "CAUSAL_PATCHING_SUMMARY.md")
with open(summary_path, 'w') as f:
    f.write("# Bidirectional Steering Experiment Summary\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Model:** Gemma-2-2B ({n_layers} layers, d_model={d_model})\n\n")

    f.write("## Experiment Design\n\n")
    f.write("**Method:** Vector Addition (Steering)\n\n")
    f.write(f"**Formula:** `output = original + coeff × steering_vector`\n\n")
    f.write(f"**Steering Vector:** `Mean_Vuln - Mean_Safe`\n\n")
    f.write(f"**Steering Coefficient:** {STEERING_COEFF}\n\n")

    f.write("### Direction 1: Safe → Vulnerable\n")
    f.write(f"- Selected {len(baseline_vul_samples)} TP samples (baseline: predict 'vul')\n")
    f.write(f"- Applied: `output = original - {STEERING_COEFF} × (Mean_Vuln - Mean_Safe)`\n")
    f.write("- Measured flip rate: % of samples changing 'vul' → 'safe'\n\n")

    f.write("### Direction 2: Vulnerable → Safe\n")
    f.write(f"- Selected {len(baseline_safe_samples)} TN samples (baseline: predict 'nonvul')\n")
    f.write(f"- Applied: `output = original + {STEERING_COEFF} × (Mean_Vuln - Mean_Safe)`\n")
    f.write("- Measured flip rate: % of samples changing 'safe' → 'vul'\n\n")

    f.write("## Key Findings\n\n")
    f.write(f"**Direction 1 (S→V) Critical Layer:** Layer {max_flip_layer_s2v} ({max_flip_rate_s2v:.1f}%)\n\n")
    f.write(f"**Direction 2 (V→S) Critical Layer:** Layer {max_flip_layer_v2s} ({max_flip_rate_v2s:.1f}%)\n\n")

    agreement = "✓ Agree" if abs(max_flip_layer_s2v - max_flip_layer_v2s) <= 2 else "✗ Differ"
    f.write(f"**Layer Agreement:** {agreement} (diff={abs(max_flip_layer_s2v - max_flip_layer_v2s)} layers)\n\n")

    f.write("## Stage Analysis\n\n")
    f.write("| Stage | S→V Flip Rate | V→S Flip Rate |\n")
    f.write("|-------|---------------|---------------|\n")
    f.write(f"| Early (L0-{n_layers//3-1}) | {early_rate_s2v:.1f}% | {early_rate_v2s:.1f}% |\n")
    f.write(f"| Middle (L{n_layers//3}-{2*n_layers//3-1}) | {middle_rate_s2v:.1f}% | {middle_rate_v2s:.1f}% |\n")
    f.write(f"| Late (L{2*n_layers//3}-{n_layers-1}) | {late_rate_s2v:.1f}% | {late_rate_v2s:.1f}% |\n\n")

    f.write("## Generated Files\n\n")
    f.write("- `flip_rate_bidirectional.png`: Comparison of both directions\n")
    f.write("- `flip_rate_by_stage.png`: Early vs middle vs late layers\n")
    f.write("- `causal_patching_results.json`: Full detailed results\n")

print(f"  Saved: {summary_path}")

print("\n" + "="*80)
print("BIDIRECTIONAL STEERING EXPERIMENT COMPLETE")
print("="*80)
print(f"\nMethod: Vector Addition Steering")
print(f"Formula: output = original + {STEERING_COEFF} × (Mean_Vuln - Mean_Safe)")
print(f"\nDirection 1 (Safe → Vulnerable):")
print(f"  Critical Layer: {max_flip_layer_s2v} ({max_flip_rate_s2v:.1f}% flip rate)")
print(f"  Stage Rates: Early={early_rate_s2v:.1f}%, Middle={middle_rate_s2v:.1f}%, Late={late_rate_s2v:.1f}%")
print(f"\nDirection 2 (Vulnerable → Safe):")
print(f"  Critical Layer: {max_flip_layer_v2s} ({max_flip_rate_v2s:.1f}% flip rate)")
print(f"  Stage Rates: Early={early_rate_v2s:.1f}%, Middle={middle_rate_v2s:.1f}%, Late={late_rate_v2s:.1f}%")
print(f"\nLayer Agreement: {abs(max_flip_layer_s2v - max_flip_layer_v2s)} layers difference")
print(f"\nGenerated files:")
print(f"  Plots: {PLOT_DIR}")
print(f"  Logs: {LOG_DIR}")
print("\n" + "="*80)

log_file.close()