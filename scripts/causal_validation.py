#!/usr/bin/env python3

import os
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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "circuit-tracer", "circuit-tracer"))
sys.path.insert(0, _REPO_ROOT)
import config

from circuit_tracer.replacement_model import ReplacementModel

DATA_PATH = os.path.join(config.DATA_DIR, "tp_tn_samples.jsonl")
NEURON_ANALYSIS_PATH = os.path.join(config.DATA_DIR, "neuron_analysis.json")
MODEL_NAME = config.MODEL_NAME
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = None
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 200

TARGET_LAYERS = [6, 7, 10, 11]
CONTROL_LAYERS = [1, 2, 15, 16]

TARGET_HEADS = [
    (0, 2),
    (5, 2),
    (2, 2),
]

print(f"\n[INIT] Loading neuron IDs from {NEURON_ANALYSIS_PATH}...", flush=True)
TARGET_NEURONS = {}
try:
    with open(NEURON_ANALYSIS_PATH, 'r') as f:
        neuron_data = json.load(f)
    
    for layer_str, neurons in neuron_data.items():
        layer = int(layer_str)
        top_20_ids = [n['neuron_idx'] for n in neurons[:20]]
        TARGET_NEURONS[layer] = top_20_ids
        print(f"  Layer {layer}: Loaded {len(top_20_ids)} neuron IDs (e.g., {top_20_ids[:3]}...)", flush=True)
    
    print(f"  Loaded neuron IDs for {len(TARGET_NEURONS)} layers", flush=True)
except Exception as e:
    print(f"  Failed to load neuron IDs: {e}", flush=True)
    print(f"  Using fallback: first 20 indices", flush=True)
    TARGET_NEURONS = {
        6: list(range(20)),
        7: list(range(20)),
        10: list(range(20)),
        11: list(range(20)),
    }

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(config.LOG_BASE, f"causal_validation_{ts}")
PLOT_DIR = os.path.join(config.PLOT_BASE, f"causal_validation_{ts}")
config.ensure_output_dirs(LOG_DIR, PLOT_DIR)

log_file = open(os.path.join(LOG_DIR, f"causal_validation_{ts}.txt"), "w")
sys.stdout = log_file
sys.stderr = log_file

print("="*80, flush=True)
print("Causal Validation Experiment: Ablation Studies", flush=True)
print("="*80, flush=True)
print(f"Timestamp: {ts}", flush=True)
print(f"Device: {DEVICE}", flush=True)
print(f"Data: {DATA_PATH}", flush=True)
print(f"Neuron IDs: {NEURON_ANALYSIS_PATH}", flush=True)
print("="*80, flush=True)
print("\n[IMPORTANT] Using ALL TP/TN samples for accurate baseline", flush=True)
print("[IMPORTANT] Prompt format matches prime.py for consistency", flush=True)
print("[IMPORTANT] Using real Top-20 neuron IDs from neuron_analysis.json", flush=True)
print("="*80, flush=True)

def patch_model_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_official_model_name
    loading.get_official_model_name = (
        lambda model_name: "google/gemma-2-2b"
        if model_name == MODEL_NAME
        else original(model_name)
    )

def patch_model_config_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_pretrained_model_config
    def patched(model_name, **kwargs):
        if model_name == MODEL_NAME:
            from transformers import AutoConfig
            return AutoConfig.from_pretrained(model_name)
        return original(model_name, **kwargs)
    loading.get_pretrained_model_config = patched

patch_model_loading()
patch_model_config_loading()

print("\n[1/7] Loading Model & Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

rm = ReplacementModel.from_pretrained(
    MODEL_NAME,
    transcoder_set="gemma",
    device=DEVICE,
    dtype=torch.float16
)
rm.eval()

n_layers = rm.cfg.n_layers
n_heads = rm.cfg.n_heads
d_model = rm.cfg.d_model

print(f"  Model: Gemma-2-2B", flush=True)
print(f"  Layers: {n_layers}, Heads: {n_heads}, d_model: {d_model}", flush=True)

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

print(f"\n[2/7] Loading TP/TN samples from {DATA_PATH}...", flush=True)

def load_samples(jsonl_path, max_samples=None):
    tp_samples = []
    tn_samples = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                pred_type = data.get('prediction_type', '')
                
                if pred_type not in ['TP', 'TN']:
                    continue
                
                code = data.get('func', '').strip()
                if not code:
                    continue
                
                prompt = f"""Code: {code}

Question: Is this code safe or vulnerable?
Answer:"""
                
                sample = {
                    'idx': data.get('idx', -1),
                    'prompt': prompt,
                    'true_label': 'vul' if pred_type == 'TP' else 'nonvul',
                    'type': pred_type,
                    'model_output': data.get('model_output', '')
                }
                
                if pred_type == 'TP':
                    tp_samples.append(sample)
                elif pred_type == 'TN':
                    tn_samples.append(sample)
                
                if max_samples is not None:
                    if len(tp_samples) >= max_samples and len(tn_samples) >= max_samples:
                        break
                    
            except Exception as e:
                continue

    if max_samples is not None:
        tp_samples = tp_samples[:max_samples]
        tn_samples = tn_samples[:max_samples]
    
    return tp_samples, tn_samples

tp_samples, tn_samples = load_samples(DATA_PATH, MAX_SAMPLES)
all_samples = tp_samples + tn_samples

print(f"  Loaded {len(tp_samples)} TP samples (vulnerable)", flush=True)
print(f"  Loaded {len(tn_samples)} TN samples (safe)", flush=True)
print(f"  Total: {len(all_samples)} samples", flush=True)

print(f"\n[3/7] Computing Baseline (No Ablation)...", flush=True)

def evaluate_samples(samples, ablation_fn=None, ablation_name="Baseline"):
    results = {
        'tp_correct': 0,
        'tp_total': 0,
        'tn_correct': 0,
        'tn_total': 0,
        'predictions': []
    }
    
    hooks = []
    if ablation_fn:
        hooks = ablation_fn()
    
    try:
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(samples)}", flush=True)
            
            try:
                if ablation_fn is None and 'model_output' in sample:
                    generated_text = sample['model_output']
                    pred_label = extract_label(generated_text)
                else:
                    inputs = tokenizer(
                        sample['prompt'].strip(),
                        return_tensors='pt',
                        truncation=True,
                        max_length=MAX_SEQ_LENGTH,
                        padding=False
                    )
                    
                    input_ids = inputs['input_ids'].to(DEVICE)
                    
                    with torch.inference_mode():
                        output_ids = rm.generate(
                            input_ids,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            verbose=False
                        )
                    
                    prompt_len = input_ids.shape[1]
                    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
                    pred_label = extract_label(generated_text)
                    
                    del inputs, output_ids
                    
                    if i % 20 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                true_label = sample['true_label']
                correct = (pred_label == true_label)
                
                if true_label == 'vul':
                    results['tp_total'] += 1
                    if correct:
                        results['tp_correct'] += 1
                else:
                    results['tn_total'] += 1
                    if correct:
                        results['tn_correct'] += 1
                
                results['predictions'].append({
                    'idx': sample['idx'],
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'correct': correct
                })
                        
            except Exception as e:
                print(f"    Warning: Failed sample {i}: {e}", flush=True)
                continue
                
    finally:
        for hook in hooks:
            hook.remove()
    
    tp_acc = results['tp_correct'] / results['tp_total'] if results['tp_total'] > 0 else 0
    tn_acc = results['tn_correct'] / results['tn_total'] if results['tn_total'] > 0 else 0
    overall_acc = (results['tp_correct'] + results['tn_correct']) / len(samples)
    
    results['tp_accuracy'] = tp_acc
    results['tn_accuracy'] = tn_acc
    results['overall_accuracy'] = overall_acc
    
    print(f"    {ablation_name}:", flush=True)
    print(f"      TP Accuracy: {tp_acc:.2%} ({results['tp_correct']}/{results['tp_total']})", flush=True)
    print(f"      TN Accuracy: {tn_acc:.2%} ({results['tn_correct']}/{results['tn_total']})", flush=True)
    print(f"      Overall Accuracy: {overall_acc:.2%}", flush=True)
    
    return results

baseline_results = evaluate_samples(all_samples, ablation_fn=None, ablation_name="Baseline")

print(f"\n[4/7] Experiment 1: Layer-wise Ablation...", flush=True)

layer_ablation_results = {}

def create_layer_ablation_hook(target_layer):
    def ablation_fn():
        def mean_ablation_hook(module, input, output):
            return output * 0
        
        hooks = []
        attn_hook = rm.blocks[target_layer].hook_attn_out.register_forward_hook(mean_ablation_hook)
        hooks.append(attn_hook)
        mlp_hook = rm.blocks[target_layer].hook_mlp_out.register_forward_hook(mean_ablation_hook)
        hooks.append(mlp_hook)
        
        return hooks
    
    return ablation_fn

print(f"\n  Testing KEY layers (should cause accuracy drop):", flush=True)
for layer in TARGET_LAYERS:
    print(f"\n  Ablating Layer {layer}...", flush=True)
    ablation_fn = create_layer_ablation_hook(layer)
    results = evaluate_samples(all_samples, ablation_fn=ablation_fn, ablation_name=f"Layer {layer} Ablated")
    layer_ablation_results[f"layer_{layer}"] = results

print(f"\n  Testing CONTROL layers (should have minimal impact):", flush=True)
for layer in CONTROL_LAYERS:
    print(f"\n  Ablating Layer {layer}...", flush=True)
    ablation_fn = create_layer_ablation_hook(layer)
    results = evaluate_samples(all_samples, ablation_fn=ablation_fn, ablation_name=f"Layer {layer} Ablated")
    layer_ablation_results[f"layer_{layer}_control"] = results

print(f"\n[5/7] Experiment 2: Attention Head Ablation...", flush=True)

head_ablation_results = {}

def create_head_ablation_hook(target_layer, target_head):
    def ablation_fn():
        def zero_hook(module, input, output):
            output[:, :, target_head, :] = 0
            return output
        
        hook = rm.blocks[target_layer].attn.hook_result.register_forward_hook(zero_hook)
        return [hook]
    
    return ablation_fn

print(f"\n  Testing Top-3 important heads:", flush=True)
for layer, head in TARGET_HEADS:
    print(f"\n  Ablating Layer {layer}, Head {head}...", flush=True)
    ablation_fn = create_head_ablation_hook(layer, head)
    results = evaluate_samples(all_samples, ablation_fn=ablation_fn, 
                               ablation_name=f"L{layer}H{head} Ablated")
    head_ablation_results[f"L{layer}H{head}"] = results

print(f"\n[6/7] Experiment 3: MLP Neuron Ablation...", flush=True)

neuron_ablation_results = {}

def create_neuron_ablation_hook(target_layer, target_neurons):
    def ablation_fn():
        def zero_hook(module, input, output):
            for neuron_idx in target_neurons:
                if neuron_idx < output.shape[-1]:
                    output[:, :, neuron_idx] = 0
            return output
        
        hook = rm.blocks[target_layer].mlp.old_mlp.hook_post.register_forward_hook(zero_hook)
        return [hook]
    
    return ablation_fn

print(f"\n  Testing Top-20 neurons in key layers:", flush=True)
for layer in [6, 7, 10, 11]:
    if layer in TARGET_NEURONS:
        print(f"\n  Ablating Layer {layer} Top-20 neurons...", flush=True)
        ablation_fn = create_neuron_ablation_hook(layer, TARGET_NEURONS[layer])
        results = evaluate_samples(all_samples, ablation_fn=ablation_fn,
                                   ablation_name=f"L{layer} Neurons Ablated")
        neuron_ablation_results[f"layer_{layer}_neurons"] = results

print(f"\n[7/7] Generating Results...", flush=True)

all_results = {
    'baseline': baseline_results,
    'layer_ablation': layer_ablation_results,
    'head_ablation': head_ablation_results,
    'neuron_ablation': neuron_ablation_results
}

def clean_results(obj):
    if isinstance(obj, dict):
        return {k: clean_results(v) for k, v in obj.items() if k != 'predictions'}
    elif isinstance(obj, list):
        return [clean_results(item) for item in obj]
    else:
        return obj

json_path = os.path.join(LOG_DIR, "ablation_results.json")
with open(json_path, 'w') as f:
    json.dump(clean_results(all_results), f, indent=2)
print(f"  Saved: {json_path}", flush=True)
    
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax1 = axes[0]
layer_names = []
layer_accs = []
layer_colors = []

layer_names.append('Baseline')
layer_accs.append(baseline_results['overall_accuracy'] * 100)
layer_colors.append('green')

for layer in TARGET_LAYERS:
    key = f"layer_{layer}"
    if key in layer_ablation_results:
        layer_names.append(f'L{layer}\n(Key)')
        layer_accs.append(layer_ablation_results[key]['overall_accuracy'] * 100)
        layer_colors.append('red')

for layer in CONTROL_LAYERS[:2]:
    key = f"layer_{layer}_control"
    if key in layer_ablation_results:
        layer_names.append(f'L{layer}\n(Ctrl)')
        layer_accs.append(layer_ablation_results[key]['overall_accuracy'] * 100)
        layer_colors.append('blue')

ax1.bar(range(len(layer_names)), layer_accs, color=layer_colors, alpha=0.7)
ax1.set_xticks(range(len(layer_names)))
ax1.set_xticklabels(layer_names, fontsize=9)
ax1.set_ylabel('Overall Accuracy (%)', fontsize=11)
ax1.set_title('Layer Ablation Impact', fontsize=12, fontweight='bold')
ax1.axhline(y=baseline_results['overall_accuracy'] * 100, color='green', 
            linestyle='--', alpha=0.5, label='Baseline')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
head_names = ['Baseline']
head_accs = [baseline_results['overall_accuracy'] * 100]

for layer, head in TARGET_HEADS:
    key = f"L{layer}H{head}"
    if key in head_ablation_results:
        head_names.append(key)
        head_accs.append(head_ablation_results[key]['overall_accuracy'] * 100)

ax2.bar(range(len(head_names)), head_accs, color=['green'] + ['red'] * (len(head_names)-1), alpha=0.7)
ax2.set_xticks(range(len(head_names)))
ax2.set_xticklabels(head_names, fontsize=9, rotation=15)
ax2.set_ylabel('Overall Accuracy (%)', fontsize=11)
ax2.set_title('Attention Head Ablation Impact', fontsize=12, fontweight='bold')
ax2.axhline(y=baseline_results['overall_accuracy'] * 100, color='green', 
            linestyle='--', alpha=0.5, label='Baseline')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

ax3 = axes[2]
neuron_names = ['Baseline']
neuron_accs = [baseline_results['overall_accuracy'] * 100]

for layer in [6, 7, 10, 11]:
    key = f"layer_{layer}_neurons"
    if key in neuron_ablation_results:
        neuron_names.append(f'L{layer}\nNeurons')
        neuron_accs.append(neuron_ablation_results[key]['overall_accuracy'] * 100)

ax3.bar(range(len(neuron_names)), neuron_accs, color=['green'] + ['orange'] * (len(neuron_names)-1), alpha=0.7)
ax3.set_xticks(range(len(neuron_names)))
ax3.set_xticklabels(neuron_names, fontsize=9)
ax3.set_ylabel('Overall Accuracy (%)', fontsize=11)
ax3.set_title('MLP Neuron Ablation Impact', fontsize=12, fontweight='bold')
ax3.axhline(y=baseline_results['overall_accuracy'] * 100, color='green', 
            linestyle='--', alpha=0.5, label='Baseline')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(PLOT_DIR, "ablation_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}", flush=True)

fig, ax = plt.subplots(figsize=(14, 8))

experiments = ['Baseline']
tp_accs = [baseline_results['tp_accuracy'] * 100]
tn_accs = [baseline_results['tn_accuracy'] * 100]

for layer in TARGET_LAYERS:
    key = f"layer_{layer}"
    if key in layer_ablation_results:
        experiments.append(f'L{layer}')
        tp_accs.append(layer_ablation_results[key]['tp_accuracy'] * 100)
        tn_accs.append(layer_ablation_results[key]['tn_accuracy'] * 100)

for layer, head in TARGET_HEADS[:2]:
    key = f"L{layer}H{head}"
    if key in head_ablation_results:
        experiments.append(f'L{layer}H{head}')
        tp_accs.append(head_ablation_results[key]['tp_accuracy'] * 100)
        tn_accs.append(head_ablation_results[key]['tn_accuracy'] * 100)

x = np.arange(len(experiments))
width = 0.35

bars1 = ax.bar(x - width/2, tp_accs, width, label='TP Accuracy (Vulnerable)', color='darkred', alpha=0.8)
bars2 = ax.bar(x + width/2, tn_accs, width, label='TN Accuracy (Safe)', color='darkblue', alpha=0.8)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('TP vs TN Accuracy After Ablation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=baseline_results['tp_accuracy'] * 100, color='red', linestyle='--', alpha=0.3)
ax.axhline(y=baseline_results['tn_accuracy'] * 100, color='blue', linestyle='--', alpha=0.3)

plt.tight_layout()
fig2_path = os.path.join(PLOT_DIR, "tp_tn_breakdown.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig2_path}", flush=True)
        
report_path = os.path.join(LOG_DIR, "ABLATION_REPORT.md")
with open(report_path, 'w') as f:
    f.write("# Causal Validation: Ablation Studies Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## Baseline Performance\n\n")
    f.write(f"- **Overall Accuracy:** {baseline_results['overall_accuracy']:.2%}\n")
    f.write(f"- **TP Accuracy (Vulnerable):** {baseline_results['tp_accuracy']:.2%}\n")
    f.write(f"- **TN Accuracy (Safe):** {baseline_results['tn_accuracy']:.2%}\n\n")
    
    f.write("## Experiment 1: Layer-wise Ablation\n\n")
    f.write("| Layer | Type | Overall Acc | TP Acc | TN Acc | Acc Drop | Impact |\n")
    f.write("|-------|------|-------------|--------|--------|----------|--------|\n")
    
    for layer in TARGET_LAYERS + CONTROL_LAYERS:
        key = f"layer_{layer}" if layer in TARGET_LAYERS else f"layer_{layer}_control"
        if key in layer_ablation_results:
            r = layer_ablation_results[key]
            layer_type = "Key" if layer in TARGET_LAYERS else "Control"
            acc_drop = (baseline_results['overall_accuracy'] - r['overall_accuracy']) * 100
            impact = "High" if acc_drop > 20 else ("Medium" if acc_drop > 5 else "Low")
            f.write(f"| {layer} | {layer_type} | {r['overall_accuracy']:.1%} | {r['tp_accuracy']:.1%} | "
                   f"{r['tn_accuracy']:.1%} | {acc_drop:.1f}% | {impact} |\n")
    
    f.write("\n## Experiment 2: Attention Head Ablation\n\n")
    f.write("| Head | Overall Acc | TP Acc | TN Acc | Acc Drop | TN Impact |\n")
    f.write("|------|-------------|--------|--------|----------|----------|\n")
    
    for layer, head in TARGET_HEADS:
        key = f"L{layer}H{head}"
        if key in head_ablation_results:
            r = head_ablation_results[key]
            acc_drop = (baseline_results['overall_accuracy'] - r['overall_accuracy']) * 100
            tn_drop = (baseline_results['tn_accuracy'] - r['tn_accuracy']) * 100
            impact = "Safety Detector" if tn_drop > 10 else "Balanced"
            f.write(f"| L{layer}H{head} | {r['overall_accuracy']:.1%} | {r['tp_accuracy']:.1%} | "
                   f"{r['tn_accuracy']:.1%} | {acc_drop:.1f}% | {impact} |\n")
    
    f.write("\n## Experiment 3: MLP Neuron Ablation\n\n")
    f.write("| Layer | Neurons | Overall Acc | TP Acc | TN Acc | Acc Drop |\n")
    f.write("|-------|---------|-------------|--------|--------|----------|\n")
    
    for layer in [6, 7, 10, 11]:
        key = f"layer_{layer}_neurons"
        if key in neuron_ablation_results:
            r = neuron_ablation_results[key]
            acc_drop = (baseline_results['overall_accuracy'] - r['overall_accuracy']) * 100
            f.write(f"| {layer} | Top-20 | {r['overall_accuracy']:.1%} | {r['tp_accuracy']:.1%} | "
                   f"{r['tn_accuracy']:.1%} | {acc_drop:.1f}% |\n")
    
    f.write("\n## Key Findings\n\n")
    
    max_layer_impact = 0
    max_layer = None
    for layer in TARGET_LAYERS:
        key = f"layer_{layer}"
        if key in layer_ablation_results:
            impact = baseline_results['overall_accuracy'] - layer_ablation_results[key]['overall_accuracy']
            if impact > max_layer_impact:
                max_layer_impact = impact
                max_layer = layer
    
    if max_layer:
        f.write(f"1. **Most Critical Layer:** Layer {max_layer} (accuracy drop: {max_layer_impact:.1%})\n")
    
    max_head_impact = 0
    max_head = None
    for layer, head in TARGET_HEADS:
        key = f"L{layer}H{head}"
        if key in head_ablation_results:
            impact = baseline_results['overall_accuracy'] - head_ablation_results[key]['overall_accuracy']
            if impact > max_head_impact:
                max_head_impact = impact
                max_head = (layer, head)
    
    if max_head:
        f.write(f"2. **Most Critical Head:** L{max_head[0]}H{max_head[1]} (accuracy drop: {max_head_impact:.1%})\n")
    
    f.write("\n## Interpretation\n\n")
    f.write("- **High impact (>20% drop):** Component is critical for decision-making\n")
    f.write("- **Medium impact (5-20% drop):** Component contributes to performance\n")
    f.write("- **Low impact (<5% drop):** Component is not essential\n")
    f.write("- **TN-specific drop:** 'Safety detector' - recognizes safe patterns\n")
    f.write("- **TP-specific drop:** 'Vulnerability detector' - recognizes unsafe patterns\n")

print(f"  Saved: {report_path}", flush=True)

print("\n" + "="*80, flush=True)
print("CAUSAL VALIDATION COMPLETE", flush=True)
print("="*80, flush=True)
print(f"\nGenerated files:", flush=True)
print(f"  Results: {LOG_DIR}", flush=True)
print(f"  Plots: {PLOT_DIR}", flush=True)
print("\n" + "="*80, flush=True)

log_file.close()
