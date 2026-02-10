# test.py
#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------
# Paths and logging
# --------------------------
LOG_DIR = "/home/chun7871/lu2025-17-14/chun7871/log"
ANALYSIS_DIR = "/home/chun7871/lu2025-17-14/chun7871/analysis"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOG_DIR, f"run_{ts}.log")
analysis_file = os.path.join(ANALYSIS_DIR, f"analysis_{ts}.json")

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

log_fp = open(log_file, "w")
sys.stdout = Tee(sys.stdout, log_fp)
sys.stderr = Tee(sys.stderr, log_fp)

# --------------------------
# Import circuit-tracer
# --------------------------
sys.path.insert(0, "/home/chun7871/lu2025-17-14/chun7871/circuit-tracer/circuit-tracer")

from transformers import AutoTokenizer
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute import attribute

# --------------------------
# Compatibility patches (KEEPED)
# --------------------------
def patch_model_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_official_model_name
    loading.get_official_model_name = (
        lambda model_name: "google/gemma-2-2b"
        if model_name == "/home/chun7871/gemma"
        else original(model_name)
    )
    return original

def patch_model_config_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_pretrained_model_config
    def patched(model_name, **kwargs):
        if model_name == "/home/chun7871/gemma":
            from transformers import AutoConfig
            return AutoConfig.from_pretrained(model_name, local_files_only=True)
        return original(model_name, **kwargs)
    loading.get_pretrained_model_config = patched
    return original

# --------------------------
# Utilities
# --------------------------
def safe_tensor_to_python(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().numpy().tolist()
    return x

def extract_feature_info(feature):
    info = {}
    try:
        vals = []
        for i in range(min(4, len(feature))):
            v = feature[i]
            vv = safe_tensor_to_python(v)
            vals.append(vv)
        if len(vals) >= 2:
            info['layer'] = int(vals[0]) if isinstance(vals[0], (int, float)) else vals[0]
            info['position'] = int(vals[1]) if isinstance(vals[1], (int, float)) else vals[1]
    except Exception:
        pass
    return info

def tally_l0_per_layer(graph):
    l0_per_layer = {}
    unique_positions = set()

    if hasattr(graph, 'active_features'):
        for f in graph.active_features:
            info = extract_feature_info(f)
            layer = info.get('layer', None)
            pos = info.get('position', None)
            if isinstance(layer, int):
                l0_per_layer[layer] = l0_per_layer.get(layer, 0) + 1
            if isinstance(pos, int):
                unique_positions.add(pos)

    denom = max(len(unique_positions), 1)
    l0_avg_per_layer = [
        {"layer": L, "avg_L0_per_pos": l0 / denom, "total_L0": l0}
        for L, l0 in l0_per_layer.items()
    ]
    l0_avg_per_layer.sort(key=lambda x: x["layer"])
    return l0_avg_per_layer

def plot_l0_per_layer(l0_list, out_path):
    if not l0_list:
        return None
    layers = [d["layer"] for d in l0_list]
    avg_l0 = [d["avg_L0_per_pos"] for d in l0_list]
    plt.figure(figsize=(10, 4))
    plt.plot(layers, avg_l0, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Avg L0 per position")
    plt.title("Average L0 (active latents) per layer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# --------------------------
# Main
# --------------------------
def main():
    MODEL_PATH = "/home/chun7871/gemma"
    PROMPT = "France's capital city is called"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply patches (keep)
    patch_model_loading()
    patch_model_config_loading()

    # ReplacementModel with transcoder
    replacement_model = ReplacementModel.from_pretrained(
        MODEL_PATH,
        transcoder_set="gemma",
        device=device,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    )

    # Run attribution
    graph = attribute(
        prompt=PROMPT,
        model=replacement_model,
        max_n_logits=5,
        batch_size=512 if device.startswith("cuda") else 128,
        max_feature_nodes=2000 if device.startswith("cuda") else 500,
        verbose=True
    )

    # Compute L0 metrics
    l0_avg_per_layer = tally_l0_per_layer(graph)

    # Save plot
    l0_img = plot_l0_per_layer(
        l0_avg_per_layer,
        os.path.join(ANALYSIS_DIR, f"l0_per_layer_{ts}.png")
    )

    # Save JSON
    analysis = {
        "timestamp": ts,
        "prompt": PROMPT,
        "l0_avg_per_layer": l0_avg_per_layer,
        "paths": {
            "log_file": log_file,
            "json_report": analysis_file,
            "l0_plot": l0_img,
        },
    }
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Print to log
    print(f"[OK] Analysis JSON saved to: {analysis_file}")
    print(f"[OK] L0 plot saved to: {l0_img}")
    print("\nPer-layer L0:")
    for d in l0_avg_per_layer:
        print(f"  Layer {d['layer']:02d} | avg_L0_per_pos={d['avg_L0_per_pos']:.2f} | total_L0={d['total_L0']}")

if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()