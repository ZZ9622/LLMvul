#!/usr/bin/env python3
"""
CWE-Based Circuit Visualization
Adapted from cwe_circuits.py – uses relative paths and HuggingFace model ID.
"""
import os
import sys
import json
import gc
from collections import defaultdict
import torch
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
from transformers import AutoTokenizer
import warnings

# ── Repository root & output directory ────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)
OUTPUT_BASE = os.environ.get("LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out"))

# ── circuit-tracer sys.path ────────────────────────────────────────────────────
_CT_PATH = os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer")
if _CT_PATH not in sys.path:
    sys.path.insert(0, _CT_PATH)

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute import attribute

try:
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from visualize_custom import visualize_graph
    print("[INFO] Successfully imported visualize_graph.")
except ImportError as e:
    print(f"[ERROR] Could not import visualize_graph: {e}")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_CWES = [
    "CWE-787",
    "CWE-476",
    "CWE-125",
    "CWE-416",
    "CWE-119",
    "CWE-190",
]

CWE_DESCRIPTIONS = {
    "CWE-787": "Buffer Overflow (Out-of-bounds Write)",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-125": "Out-of-bounds Read",
    "CWE-416": "Use After Free",
    "CWE-119": "Memory Buffer Operations",
    "CWE-190": "Integer Overflow",
    "CWE-703": "Improper Check/Handling",
    "CWE-369": "Divide By Zero",
    "CWE-362": "Race Condition",
    "CWE-200": "Information Exposure",
}

VUL_PATH    = os.path.join(ROOT_DIR, "data", "primevul236.jsonl")
MODEL_PATH  = "Chun9622/llmvul-finetuned-gemma"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_FEATURE_NODES   = 5000
NODE_THRESHOLD      = 0.70
EDGE_THRESHOLD      = 0.85
SHOW_TOP_K_EDGES    = 150
MAX_NODES_PER_LAYER = 20

ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR     = os.path.join(OUTPUT_BASE, "log",   f"cwe_circuits_{ts}")
CIRCUIT_DIR = os.path.join(OUTPUT_BASE, "plots", f"cwe_circuits_{ts}", "circuits")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CIRCUIT_DIR, exist_ok=True)

# Redirect stdout/stderr to log file
_log_file   = open(os.path.join(LOG_DIR, f"cwe_analysis_{ts}.txt"), "w")
sys.stdout  = _log_file
sys.stderr  = _log_file

print("=" * 80)
print("CWE-Based Circuit Visualization Analysis")
print("=" * 80)
print(f"Target CWEs : {TARGET_CWES}")
print(f"Output Dir  : {CIRCUIT_DIR}")
print("=" * 80)

# ── Model patches ─────────────────────────────────────────────────────────────
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

print("\n[INFO] Loading Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

rm = ReplacementModel.from_pretrained(
    MODEL_PATH,
    transcoder_set="gemma",
    device=DEVICE,
    dtype=torch.float16,
)
rm.eval()
print("[INFO] Model Loaded Successfully.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def extract_label(text):
    t = text.lower()
    if 'vulnerable' in t and 'not vulnerable' not in t:
        return "vul"
    if 'safe' in t and 'not safe' not in t:
        return "nonvul"
    return "unknown"

def load_samples_by_cwe(jsonl_path):
    """Organise samples by CWE type."""
    cwe_samples = defaultdict(list)
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                obj  = json.loads(line)
                idx  = obj.get("idx", -1)
                cwes = obj.get("cwe", [])
                code = obj.get("func", "").strip()
                if not code or not cwes:
                    continue
                if isinstance(cwes, str):
                    cwes = [cwes]
                for cwe in cwes:
                    if cwe in TARGET_CWES:
                        if len(code) > 1000:
                            code = code[:1000] + "\n// ... (truncated)"
                        prompt = (
                            f"Code: {code}\n\n"
                            f"Question: Is this code safe or vulnerable?\nAnswer:"
                        )
                        cwe_samples[cwe].append({
                            "idx": idx,
                            "cwe": cwe,
                            "prompt": prompt,
                            "code_length": len(code),
                        })
            except Exception:
                pass
    return cwe_samples

def select_representative_sample(samples):
    """Prefer code length 200-600 chars for clearer visualisation."""
    if not samples:
        return None
    sorted_samples = sorted(samples, key=lambda x: abs(x['code_length'] - 400))
    for s in sorted_samples:
        if 200 <= s['code_length'] <= 600:
            return s
    return sorted_samples[0]

def generate_circuit_for_cwe(cwe, sample):
    """Run attribution + save circuit for one CWE sample."""
    idx    = sample['idx']
    prompt = sample['prompt']

    print(f"\n{'='*80}")
    print(f"Processing CWE : {cwe}")
    print(f"Description    : {CWE_DESCRIPTIONS.get(cwe, 'Unknown')}")
    print(f"Sample ID      : {idx}  |  Code length: {sample['code_length']} chars")
    print(f"{'='*80}")

    print("[STEP 1/3] Running model prediction...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        output      = rm.generate(inputs.input_ids, max_new_tokens=50, do_sample=False)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred_label  = extract_label(output_text)
    print(f"  Prediction: {pred_label}")

    print("[STEP 2/3] Running attribution analysis...")
    clear_gpu_cache()
    try:
        with torch.enable_grad():
            g = attribute(
                prompt=prompt,
                model=rm,
                max_n_logits=3,
                batch_size=1,
                max_feature_nodes=MAX_FEATURE_NODES,
                verbose=False,
            )
        n_active = len(g.active_features) if hasattr(g, 'active_features') else 0
        print(f"  Active features found: {n_active}")

        print("[STEP 3/3] Generating circuit visualisation...")
        cwe_clean  = cwe.replace("CWE-", "")
        desc_clean = (
            CWE_DESCRIPTIONS.get(cwe, "Unknown")
            .replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        )
        filename  = f"circuit_{cwe_clean}_{desc_clean}_sample{idx}"
        save_path = os.path.join(CIRCUIT_DIR, filename + ".pdf")

        result = visualize_graph(
            g,
            save_path,
            node_threshold=NODE_THRESHOLD,
            edge_threshold=EDGE_THRESHOLD,
            max_nodes_per_layer=MAX_NODES_PER_LAYER,
            show_top_k_edges=SHOW_TOP_K_EDGES,
            save_json=True,
            tokenizer=tokenizer,
        )

        if result.get("success"):
            print(f"  ✓ Circuit saved: {filename}.pdf/.txt/.json")
            if "active_nodes" in result:
                print(f"  → Active nodes: {result['active_nodes']}")
            if "active_edges" in result:
                print(f"  → Active edges: {result['active_edges']}")
            return {
                "cwe": cwe, "idx": idx, "success": True,
                "files": {
                    "pdf":  save_path,
                    "txt":  save_path.replace(".pdf", ".txt"),
                    "json": save_path.replace(".pdf", ".json"),
                },
                "stats": result,
            }
        else:
            print("  ✗ Visualisation returned non-success")
            return {"cwe": cwe, "idx": idx, "success": False}

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"cwe": cwe, "idx": idx, "success": False, "error": str(e)}

# ── Phase 1: Load samples ──────────────────────────────────────────────────────
print("\n[PHASE 1] Loading and organising samples by CWE...")
cwe_samples = load_samples_by_cwe(VUL_PATH)

print(f"\n[INFO] Found samples for {len(cwe_samples)} target CWEs:")
for cwe in TARGET_CWES:
    count = len(cwe_samples.get(cwe, []))
    print(f"  - {cwe} ({CWE_DESCRIPTIONS.get(cwe, 'Unknown')}): {count} samples")

# ── Phase 2: Select representative samples ────────────────────────────────────
print("\n[PHASE 2] Selecting representative samples...")
selected_samples = {}
for cwe in TARGET_CWES:
    if cwe in cwe_samples and cwe_samples[cwe]:
        sample = select_representative_sample(cwe_samples[cwe])
        if sample:
            selected_samples[cwe] = sample
            print(f"  ✓ {cwe}: Sample {sample['idx']} (length={sample['code_length']})")
        else:
            print(f"  ✗ {cwe}: No suitable sample found")
    else:
        print(f"  ✗ {cwe}: No samples in dataset")

print(f"\n[INFO] Selected {len(selected_samples)} samples for visualisation")

# ── Phase 3: Generate circuits ────────────────────────────────────────────────
print("\n[PHASE 3] Generating circuit visualisations...")
results = []
for i, (cwe, sample) in enumerate(selected_samples.items(), 1):
    print(f"\n[{i}/{len(selected_samples)}] Processing {cwe}...")
    result = generate_circuit_for_cwe(cwe, sample)
    results.append(result)
    clear_gpu_cache()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

successful = [r for r in results if r.get("success")]
failed     = [r for r in results if not r.get("success")]

print(f"\nSuccessfully generated: {len(successful)}/{len(results)} circuits")

if successful:
    print("\n✓ Generated Circuit Diagrams:")
    for r in successful:
        cwe  = r['cwe']
        desc = CWE_DESCRIPTIONS.get(cwe, 'Unknown')
        print(f"  - {cwe} ({desc})")
        print(f"    PDF: {os.path.basename(r['files']['pdf'])}")
        print(f"    Sample ID: {r['idx']}")

if failed:
    print(f"\n✗ Failed: {len(failed)} circuits")
    for r in failed:
        print(f"  - {r['cwe']}: {r.get('error', 'Unknown error')}")

print(f"\n[INFO] All output files in: {CIRCUIT_DIR}")
print(f"[INFO] Log: {os.path.join(LOG_DIR, 'cwe_analysis_' + ts + '.txt')}")

# ── Write Markdown summary ────────────────────────────────────────────────────
summary_path = os.path.join(LOG_DIR, "CWE_CIRCUITS_SUMMARY.md")
with open(summary_path, 'w') as f:
    f.write("# CWE-Based Circuit Visualisation Summary\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Output Directory:** `{CIRCUIT_DIR}`\n\n")

    f.write("## Generated Circuits\n\n")
    f.write(f"Successfully generated {len(successful)} out of {len(results)} circuit diagrams.\n\n")

    for r in successful:
        cwe  = r['cwe']
        desc = CWE_DESCRIPTIONS.get(cwe, 'Unknown')
        f.write(f"### {cwe}: {desc}\n\n")
        f.write(f"- **Sample ID:** {r['idx']}\n")
        f.write(f"- **PDF:** `{os.path.basename(r['files']['pdf'])}`\n")
        f.write(f"- **Statistics:** `{os.path.basename(r['files']['txt'])}`\n")
        f.write(f"- **JSON Data:** `{os.path.basename(r['files']['json'])}`\n")
        if 'stats' in r:
            stats = r['stats']
            if 'active_nodes' in stats:
                f.write(f"- **Active Nodes:** {stats['active_nodes']}\n")
            if 'active_edges' in stats:
                f.write(f"- **Active Edges:** {stats['active_edges']}\n")
        f.write("\n")

    f.write("## Circuit Interpretation\n\n")
    f.write("Each circuit diagram shows:\n\n")
    f.write("1. **Input Tokens** (Green circles): Key tokens from the vulnerable code\n")
    f.write("2. **Feature Nodes** (Blue circles): Activated features in different layers\n")
    f.write("   - Layer 0-5: Low-level syntax features\n")
    f.write("   - Layer 6-15: Mid-level semantic features\n")
    f.write("   - Layer 16-25: High-level reasoning features\n")
    f.write("3. **Output Logits** (Red squares): Final prediction\n")
    f.write("4. **Edges**: Information flow (blue=positive, red=negative weight)\n\n")

    f.write("## Usage\n\n")
    f.write("```bash\n")
    f.write(f"# View all generated PDFs\n")
    f.write(f"ls {CIRCUIT_DIR}/*.pdf\n\n")
    f.write(f"# View interactive circuits (JSON) via web server\n")
    f.write(f"cd {os.path.join(ROOT_DIR, 'circuit-tracer', 'circuit-tracer')}\n")
    f.write(f"python -m circuit_tracer start-server --graph_file_dir {CIRCUIT_DIR} --port 8041\n")
    f.write("```\n")

print(f"\n[INFO] Summary saved: {summary_path}")
print("\n[DONE] All processing complete!")

_log_file.close()