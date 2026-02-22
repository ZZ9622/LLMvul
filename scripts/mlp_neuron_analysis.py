#!/usr/bin/env python3
"""
MLP Neuron Activation Analysis for Vulnerability Detection
"""

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)
OUTPUT_BASE = os.environ.get(
    "LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out")
)

import sys
import json
import re
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer
import warnings
import gc
import threading
from contextlib import nullcontext

sys.path.insert(0, os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer"))
from circuit_tracer.replacement_model import ReplacementModel

DATA_PATH  = os.path.join(ROOT_DIR, "data", "tp_tn_samples.jsonl")
MODEL_PATH = "Chun9622/llmvul-finetuned-gemma"

DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "8"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS",  "10"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))
LOG_EVERY       = int(os.getenv("LOG_EVERY", "10"))

TARGET_LAYERS = [6, 7, 10, 11]   # Layers for MLP analysis
TOP_K_NEURONS = 20                # Number of top neurons to visualise

ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR  = os.path.join(OUTPUT_BASE, "log",   f"mlp_neuron_{ts}")
PLOT_DIR = os.path.join(OUTPUT_BASE, "plots", f"mlp_neuron_{ts}")
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

neuron_data_path = os.path.join(LOG_DIR, "neuron_analysis.json")

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

_log_fh = open(os.path.join(LOG_DIR, f"log_{ts}.txt"), "w")
sys.stdout = Tee(sys.stdout, _log_fh)
sys.stderr = sys.stdout

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

print("="*80)
print("MLP Neuron Activation Analysis for Vulnerability Detection")
print("="*80)
print(f"Timestamp : {ts}")
print(f"Device    : {DEVICE}")
print(f"Output    : {PLOT_DIR}")
print(f"Layers    : {TARGET_LAYERS}")
print(f"Top-K     : {TOP_K_NEURONS}")
print("="*80)

class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        self._writers = 0
        self._write_waiters = 0

    def acquire_read(self):
        self._read_ready.acquire()
        try:
            while self._writers > 0 or self._write_waiters > 0:
                self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        self._read_ready.acquire()
        self._write_waiters += 1
        try:
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._write_waiters -= 1
            self._writers += 1
        finally:
            self._read_ready.release()

    def release_write(self):
        self._read_ready.acquire()
        try:
            self._writers -= 1
            self._read_ready.notify_all()
        finally:
            self._read_ready.release()

model_lock = ReadWriteLock()

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

print("\nLoading tokenizer...")
print("\n[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

print("[INFO] Loading ReplacementModel...")
rm = ReplacementModel.from_pretrained(
    MODEL_PATH,
    transcoder_set="gemma",
    device=DEVICE,
    dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
)
rm.eval()
print("[INFO] Model loaded.")

if torch.cuda.is_available():
    pred_stream = torch.cuda.Stream()
else:
    pred_stream = None

neuron_storage = {
    l: {"vul": [], "nonvul": [], "vul_indices": [], "nonvul_indices": []}
    for l in TARGET_LAYERS
}
neuron_storage_lock = threading.Lock()

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def extract_label(text):
    if not text:
        return "unknown"
    t = text.lower().strip()
    if 'vulnerable' in t or 'unsafe' in t:
        return "vul"
    if 'safe' in t or 'secure' in t:
        return "nonvul"
    return "unknown"

def capture_activations_batch(input_ids, attention_mask, batch_indices, sample_type="vul"):
    try:
        with torch.inference_mode():
            _, cache = rm.run_with_cache(input_ids, return_type=None)

        seq_lens = attention_mask.sum(dim=1)

        with neuron_storage_lock:
            for layer_idx in TARGET_LAYERS:
                cache_key = f"blocks.{layer_idx}.mlp.hook_post"

                if cache_key not in cache:
                    for alt_key in [
                        f"blocks.{layer_idx}.mlp.old_mlp.hook_post",
                        f"blocks.{layer_idx}.hook_mlp_out",
                    ]:
                        if alt_key in cache:
                            cache_key = alt_key
                            break

                if cache_key in cache:
                    batch_acts = cache[cache_key].detach().cpu().to(torch.float16)

                    if len(neuron_storage[layer_idx][sample_type]) == 0:
                        print(f"[INFO] Layer {layer_idx}: capturing from '{cache_key}', shape={batch_acts.shape}")

                    for i in range(len(batch_indices)):
                        if i >= batch_acts.shape[0]:
                            break
                        last_token_idx = min(int(seq_lens[i]) - 1, batch_acts.shape[1] - 1)
                        last_token_idx = max(0, last_token_idx)
                        vec = batch_acts[i, last_token_idx, :].float().numpy()
                        neuron_storage[layer_idx][sample_type].append(vec)
                        neuron_storage[layer_idx][f"{sample_type}_indices"].append(batch_indices[i])
                else:
                    if len(neuron_storage[layer_idx][sample_type]) == 0:
                        mlp_keys = [k for k in cache.keys() if f"blocks.{layer_idx}.mlp" in k]
                        print(f"[WARN] '{cache_key}' not found for layer {layer_idx}. Available: {mlp_keys}")

    except Exception as e:
        print(f"[ERROR] Activation capture failed: {e}")
        import traceback; traceback.print_exc()

def run_prediction_batch(samples, sample_type="vul"):
    if not samples:
        return []

    prompts  = [s["prompt"].strip() for s in samples]
    indices  = [s["idx"] for s in samples]

    stream_context = torch.cuda.stream(pred_stream) if pred_stream else nullcontext()

    with stream_context:
        enc = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=MAX_INPUT_TOKENS, padding=True
        )
        enc = {k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
        attn      = enc.get("attention_mask")

        prompt_lens = attn.sum(dim=1).cpu().tolist() if attn is not None else [len(ids) for ids in input_ids]

        model_lock.acquire_read()
        try:
            capture_activations_batch(input_ids, attn, indices, sample_type=sample_type)
            output_ids = rm.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                verbose=False
            )
        finally:
            model_lock.release_read()

        gen_slices = [output_ids[i][int(prompt_lens[i]):] for i in range(len(samples))]
        decoded    = tokenizer.batch_decode(gen_slices, skip_special_tokens=True)

        results = []
        for i in range(len(samples)):
            out_text = decoded[i].strip() or tokenizer.decode(output_ids[i], skip_special_tokens=True).strip()
            results.append((extract_label(out_text), out_text, prompts[i]))

    if pred_stream:
        pred_stream.synchronize()
    return results

def process_samples(samples, tag):
    sample_type_key = "vul" if tag.startswith("VUL") else "nonvul"
    total = len(samples)
    print(f"\n[{tag}] Processing {total} samples (type='{sample_type_key}')...")

    idx = 0
    while idx < total:
        batch   = samples[idx: idx + PRED_BATCH_SIZE]
        run_prediction_batch(batch, sample_type=sample_type_key)

        done = idx + len(batch)
        if done % LOG_EVERY == 0 or done == total:
            print(f"  [{tag}] {done}/{total}")

        idx += len(batch)
        if idx % (PRED_BATCH_SIZE * 5) == 0:
            gc.collect()

    print(f"  [{tag}] Capture summary:")
    for layer in TARGET_LAYERS:
        print(f"    Layer {layer}: vul={len(neuron_storage[layer]['vul'])}, "
              f"nonvul={len(neuron_storage[layer]['nonvul'])}")

    clear_gpu_cache()

def load_and_split_prompts(jsonl_path):
    print(f"\n[INFO] Reading from {jsonl_path}")
    vul_samples, nonvul_samples = [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                code = data.get("func", "").strip()
                if not code:
                    continue

                pred_type = data.get("prediction_type", "")
                if pred_type == "TP":
                    label = "vul"
                elif pred_type == "TN":
                    label = "nonvul"
                else:
                    target = data.get("target")
                    if target == 1:
                        label = "vul"
                    elif target == 0:
                        label = "nonvul"
                    else:
                        continue

                prompt = f"Code: {code}\n\nQuestion: Is this code safe or vulnerable?\nAnswer:"
                sample = {"idx": data.get("idx", -1), "true_label": label, "prompt": prompt}

                if label == "vul":
                    vul_samples.append(sample)
                else:
                    nonvul_samples.append(sample)

            except Exception:
                pass

    return vul_samples, nonvul_samples

def analyze_neuron_specialization():
    print("\n[INFO] Starting neuron specialisation analysis...")
    analysis_results = {}

    for layer in TARGET_LAYERS:
        print(f"\n[LAYER {layer}]")
        print(f"  VUL samples   : {len(neuron_storage[layer]['vul'])}")
        print(f"  NONVUL samples: {len(neuron_storage[layer]['nonvul'])}")

        if not neuron_storage[layer]["vul"]:
            print(f"  [WARN] No VUL data – skipping"); continue
        if not neuron_storage[layer]["nonvul"]:
            print(f"  [WARN] No NONVUL data – skipping"); continue

        vul_acts    = np.array(neuron_storage[layer]["vul"])
        nonvul_acts = np.array(neuron_storage[layer]["nonvul"])

        mean_vul    = np.mean(vul_acts,    axis=0)
        mean_nonvul = np.mean(nonvul_acts, axis=0)
        diff        = mean_vul - mean_nonvul

        top_indices = np.argsort(diff)[-TOP_K_NEURONS:][::-1]

        viz_limit    = 50
        limit_vul    = min(viz_limit, vul_acts.shape[0])
        limit_nonvul = min(viz_limit, nonvul_acts.shape[0])

        vul_subset    = vul_acts[:limit_vul,    :]
        nonvul_subset = nonvul_acts[:limit_nonvul, :]

        print(f"  Heatmap: {limit_vul} VUL + {limit_nonvul} SAFE samples")

        layer_data     = []
        heatmap_matrix = []

        for neuron_idx in top_indices:
            layer_data.append({
                "neuron_idx"  : int(neuron_idx),
                "selectivity" : float(diff[neuron_idx]),
                "avg_vul"     : float(mean_vul[neuron_idx]),
                "avg_nonvul"  : float(mean_nonvul[neuron_idx])
            })
            combined_row = np.concatenate([
                vul_subset[:, neuron_idx],
                nonvul_subset[:, neuron_idx]
            ])
            heatmap_matrix.append(combined_row)

        analysis_results[layer] = layer_data

        # heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        heatmap_data = np.array(heatmap_matrix)

        sns.heatmap(
            heatmap_data,
            cmap="RdYlBu_r",
            cbar_kws={'label': 'MLP Activation Value'},
            ax=ax,
            vmin=np.percentile(heatmap_data, 5),
            vmax=np.percentile(heatmap_data, 95)
        )

        ax.axvline(x=limit_vul, color='white', linestyle='--', linewidth=3)
        ax.text(limit_vul / 2,              -1, 'Vulnerable (TP)',
                ha='center', fontsize=14, fontweight='bold', color='black')
        ax.text(limit_vul + limit_nonvul / 2, -1, 'Safe (TN)',
                ha='center', fontsize=14, fontweight='bold', color='black')

        ax.set_title(
            f"MLP Neuron Activation Patterns in Layer {layer}\n"
            f"Top-{TOP_K_NEURONS} Vulnerability-Selective Neurons",
            fontsize=16, fontweight='bold', pad=60
        )
        ax.set_xlabel("Samples (Left: Vulnerable, Right: Safe)", fontsize=12)
        ax.set_ylabel("Neuron Index", fontsize=12)
        ax.set_yticks(np.arange(len(top_indices)) + 0.5)
        ax.set_yticklabels([f"N{i}" for i in top_indices], rotation=0, fontsize=9)

        heatmap_path = os.path.join(PLOT_DIR, f"layer_{layer}_mlp_neuron_heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {heatmap_path}")

        # selectivity bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        neuron_labels      = [f"N{i}" for i in top_indices]
        selectivity_values = [float(diff[i]) for i in top_indices]
        colors             = ['red' if v > 0 else 'blue' for v in selectivity_values]

        ax.barh(neuron_labels, selectivity_values, color=colors, alpha=0.7)
        ax.set_xlabel('Selectivity (Mean_VUL − Mean_NONVUL)', fontsize=12)
        ax.set_ylabel('Neuron Index', fontsize=12)
        ax.set_title(
            f'Layer {layer}: Top-{TOP_K_NEURONS} Vulnerability-Selective Neurons',
            fontsize=14, fontweight='bold'
        )
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        bar_path = os.path.join(PLOT_DIR, f"layer_{layer}_neuron_selectivity.png")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {bar_path}")

    return analysis_results

vul_prompts, nonvul_prompts = load_and_split_prompts(DATA_PATH)
print(f"[INFO] VUL (TP) samples  : {len(vul_prompts)}")
print(f"[INFO] NONVUL (TN) samples: {len(nonvul_prompts)}")

if vul_prompts:
    process_samples(vul_prompts,    tag="VUL_TP")
if nonvul_prompts:
    process_samples(nonvul_prompts, tag="NONVUL_TN")

data = analyze_neuron_specialization()

with open(neuron_data_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"\n[SAVED] {neuron_data_path}")

print("\n" + "="*80)
print("MLP NEURON ANALYSIS COMPLETE")
print("="*80)
print(f"Plots : {PLOT_DIR}")
print(f"Log   : {LOG_DIR}")
print("="*80)
