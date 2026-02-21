#!/usr/bin/env python3

import torch
import numpy as np
import json
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from transformers import AutoTokenizer
from collections import defaultdict
import sys
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)          
OUTPUT_BASE = os.environ.get(
    "LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out")
)

MODEL_PATH = "Chun9622/llmvul-finetuned-gemma" 
DATA_PATH = os.path.join(ROOT_DIR, "data", "tp_tn_samples.jsonl")

# please use tp_tn_samples.jsonl from the dataset prepared by running prime.py, which contains a balanced set of true positive (vulnerable) and true negative (safe) samples with their vulnerability types. This file is essential for the analysis in this script.

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit-tracer", "circuit-tracer"))
from circuit_tracer.replacement_model import ReplacementModel

def load_data(path, label_type, limit=None, return_vuln_type=False):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            if d.get('prediction_type') == label_type:
                code = d.get('func', '').strip()
                
                cwe_list = d.get('cwe', [])
                vuln_type = cwe_list[0] if cwe_list else 'unknown'
                if code:
                    prompt = f"Code: {code}\n\nQuestion: Is this code safe or vulnerable?\nAnswer:"
                    if return_vuln_type:
                        samples.append((prompt, vuln_type))
                    else:
                        samples.append(prompt)
            if limit and len(samples) >= limit: break
    return samples

def get_l2_norms(model, tokenizer, samples, vuln_types=None):
    layer_norms = defaultdict(list)
    type_layer_norms = defaultdict(lambda: defaultdict(list))
    
    for i, item in enumerate(samples):
        if isinstance(item, tuple):
            prompt, vuln_type = item
        else:
            prompt = item
            vuln_type = None
            
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(inputs['input_ids'], return_type=None)

        for layer in range(model.cfg.n_layers):
            act = cache[f"blocks.{layer}.hook_resid_post"]
            vec = act[0, -1, :]
            norm = torch.norm(vec, p=2).item()
            layer_norms[layer].append(norm)

            if vuln_type and vuln_types:
                type_layer_norms[vuln_type][layer].append(norm)

        del cache
        torch.cuda.empty_cache()

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def bootstrap_ci(x, y, n_boot=1000, ci=95):
    diffs = []
    x = np.array(x)
    y = np.array(y)
    for _ in range(n_boot):
        x_sample = np.random.choice(x, len(x), replace=True)
        y_sample = np.random.choice(y, len(y), replace=True)
        diffs.append(np.mean(x_sample) - np.mean(y_sample))
    
    lower = np.percentile(diffs, (100-ci)/2)
    upper = np.percentile(diffs, 100 - (100-ci)/2)
    return lower, upper

import transformer_lens.loading_from_pretrained as loading
loading.get_official_model_name = lambda x: "google/gemma-2-2b" if x == MODEL_PATH else x
rm = ReplacementModel.from_pretrained(MODEL_PATH, transcoder_set="gemma", device=DEVICE, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

vul_prompts = load_data(DATA_PATH, 'TP', NUM_SAMPLES, return_vuln_type=True)
safe_prompts = load_data(DATA_PATH, 'TN', NUM_SAMPLES, return_vuln_type=False)

vul_norms, vul_type_norms = get_l2_norms(rm, tokenizer, vul_prompts, vuln_types=True)
safe_norms, _ = get_l2_norms(rm, tokenizer, safe_prompts, vuln_types=False)

n_layers = rm.cfg.n_layers
p_values = []
cohens_ds = []
ci_lowers = []
ci_uppers = []

for layer in range(n_layers):
    v_data = vul_norms[layer]
    s_data = safe_norms[layer]
    
    t_stat, p_val = stats.ttest_ind(v_data, s_data, equal_var=False)
    d = cohens_d(v_data, s_data)
    lower, upper = bootstrap_ci(v_data, s_data)
    
    p_values.append(p_val)
    cohens_ds.append(d)
    ci_lowers.append(lower)
    ci_uppers.append(upper)

bonferroni_thresh = 0.05 / n_layers
_, fdr_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("OVERALL_STATISTICS")
print(f"Layer,P_Value,Bonferroni_Sig,FDR_Sig,Cohen_d,CI_Lower,CI_Upper")
for layer in range(n_layers):
    bonf_sig = 1 if p_values[layer] < bonferroni_thresh else 0
    fdr_sig = 1 if fdr_corrected[layer] else 0
    print(f"{layer},{p_values[layer]:.6e},{bonf_sig},{fdr_sig},{cohens_ds[layer]:.4f},{ci_lowers[layer]:.4f},{ci_uppers[layer]:.4f}")

print("\nPER_VULNERABILITY_TYPE")
vuln_types = sorted(vul_type_norms.keys())
for vtype in vuln_types:
    print(f"\nVulnerability_Type: {vtype}")
    print(f"Layer,P_Value,Cohen_d,CI_Lower,CI_Upper")
    
    for layer in range(n_layers):
        if layer in vul_type_norms[vtype] and len(vul_type_norms[vtype][layer]) > 1:
            vt_data = vul_type_norms[vtype][layer]
            s_data = safe_norms[layer]
            
            if len(vt_data) > 1 and len(s_data) > 1:
                t_stat, p_val = stats.ttest_ind(vt_data, s_data, equal_var=False)
                d = cohens_d(vt_data, s_data)
                lower, upper = bootstrap_ci(vt_data, s_data, n_boot=500)
                print(f"{layer},{p_val:.6e},{d:.4f},{lower:.4f},{upper:.4f}")

