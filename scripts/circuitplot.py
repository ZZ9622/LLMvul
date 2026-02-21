#!/usr/bin/env python3
import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)           
OUTPUT_BASE = os.environ.get(
    "LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out")
)
_CT_PATH = os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer")
if _CT_PATH not in sys.path:
    sys.path.insert(0, _CT_PATH)
import json
import torch
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
from transformers import AutoTokenizer
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute import attribute
import warnings

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from visualize_custom import visualize_graph
    print("[INFO] Successfully imported visualize_graph from visualize_custom.")
except ImportError as e:
    print(f"[WARN] Could not import visualize_graph: {e}")
    visualize_graph = None

TARGET_IDS = [196316, 90797, 205736, 220195]

VUL_PATH = os.path.join(ROOT_DIR, "data", "primevul236.jsonl")
NONVUL_PATH = os.path.join(ROOT_DIR, "data", "primenonvul236.jsonl")
MODEL_PATH = "Chun9622/llmvul-finetuned-gemma"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(OUTPUT_BASE, "log", ts)
PLOT_DIR = os.path.join(OUTPUT_BASE, "plots", ts)
CIRCUIT_DIR = f"{PLOT_DIR}/circuits"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CIRCUIT_DIR, exist_ok=True)

MAX_FEATURE_NODES = 5000
EDGE_THRESHOLD = 0.01
NODE_THRESHOLD = 0.2
SHOW_TOP_K_EDGES = 300

sys.stdout = open(os.path.join(LOG_DIR, f"log_{ts}.txt"), "w")
sys.stderr = sys.stdout

print(f"[INFO] Target IDs: {TARGET_IDS}")
print("[INFO] Loading Model & Tokenizer...")

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
rm = ReplacementModel.from_pretrained(
    MODEL_PATH,
    transcoder_set="gemma",
    device=DEVICE,
    torch_dtype=torch.float16
)
rm.eval()
print("[INFO] Model Loaded.")

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_label(text):
    t = text.lower()
    if 'vulnerable' in t and 'not vulnerable' not in t: return "vul"
    if 'safe' in t and 'not safe' not in t: return "nonvul"
    return "unknown"

def load_and_filter_prompts(jsonl_path, target_ids):
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                idx = obj.get("idx", -1)
                
                if idx in target_ids:
                    code = obj.get("func", "").strip()
                    prompt = f"Code: {code}\n\nQuestion: Is this code safe or vulnerable?\nAnswer:"
                    samples.append({
                        "idx": idx,
                        "true_label": "vul" if obj.get("target") == 1 else "nonvul",
                        "prompt": prompt
                    })
            except:
                pass
    return samples

print("[INFO] Loading and filtering data...")
vul_samples = load_and_filter_prompts(VUL_PATH, TARGET_IDS)
nonvul_samples = load_and_filter_prompts(NONVUL_PATH, TARGET_IDS)
all_targets = vul_samples + nonvul_samples

print(f"[INFO] Found {len(all_targets)} samples matching your IDs.")

for i, sample in enumerate(all_targets):
    idx = sample['idx']
    tag = sample['true_label'].upper()
    print(f"\n[{i+1}/{len(all_targets)}] Processing Sample {idx} ({tag})...")
    
    inputs = tokenizer(sample['prompt'], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = rm.generate(inputs.input_ids, max_new_tokens=50, do_sample=False)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred_label = extract_label(output_text)
    
    print(f"   True: {sample['true_label']} | Pred: {pred_label}")
    
    print("   Running attribution (finding the circuit)...")
    clear_gpu_cache()
    try:
        with torch.enable_grad():
            g = attribute(
                prompt=sample['prompt'],
                model=rm,
                max_n_logits=3,
                batch_size=1,
                max_feature_nodes=MAX_FEATURE_NODES,
                verbose=False
            )
        
        if visualize_graph:
            filename = f"circuit_{tag}_{idx}.pdf"
            save_path = os.path.join(CIRCUIT_DIR, filename)
            
            print(f"   Drawing graph (Threshold: Edge={EDGE_THRESHOLD}, Node={NODE_THRESHOLD})...")
            
            result = visualize_graph(
                g,
                save_path,
                node_threshold=NODE_THRESHOLD,
                edge_threshold=EDGE_THRESHOLD,
                max_nodes_per_layer=15,
                show_top_k_edges=SHOW_TOP_K_EDGES,
                save_json=True,
                tokenizer=tokenizer
            )
            
            if result.get("success"):
                print(f"   [SUCCESS] Graph saved to {save_path}")
            else:
                print(f"   [FAIL] Visualization returned false.")
        else:
            print(f"   [WARN] Visualization skipped (library not found).")
        
    except Exception as e:
        print(f"   [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()

print("\n[DONE] All tasks completed.")
print(f"[INFO] Please check {CIRCUIT_DIR} for your PDF files.")