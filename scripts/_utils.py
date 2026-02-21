"""
Shared utilities for LLMvul scripts.
Provides ROOT_DIR, data loading with HuggingFace fallback,
circuit-tracer sys.path setup, model loading patch, and output directory helpers.
"""

import os
import sys
import json
from datetime import datetime

# ── Repository root (LLMvul/) ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

# ── circuit-tracer import path ──────────────────────────────────────────────
CT_PATH = os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer")
if CT_PATH not in sys.path:
    sys.path.insert(0, CT_PATH)

# ── Model / Dataset identifiers ─────────────────────────────────────────────
MODEL_HF_ID = "Chun9622/llmvul-finetuned-gemma"
DATASET_HF_ID = "Chun9622/LLMvul"

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_BASE = os.environ.get(
    "LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out")
)


def make_output_dirs(subdir: str = ""):
    """Return (log_dir, plot_dir) timestamped under OUTPUT_BASE/subdir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(OUTPUT_BASE, subdir, ts) if subdir else os.path.join(OUTPUT_BASE, ts)
    log_dir = os.path.join(prefix, "log")
    plot_dir = os.path.join(prefix, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return log_dir, plot_dir, ts


# ── Data helpers ────────────────────────────────────────────────────────────

def _local_data(filename: str) -> str | None:
    """Return path if local data file exists, else None."""
    p = os.path.join(ROOT_DIR, "data", filename)
    return p if os.path.exists(p) else None


def _hf_to_jsonl(hf_records, jsonl_path: str):
    """Write HuggingFace dataset records to a JSONL file."""
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in hf_records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")


def ensure_primevul_jsonl() -> tuple[str, str]:
    """
    Return (vul_jsonl_path, nonvul_jsonl_path) for the PrimeVul dataset.
    Downloads from HuggingFace ``Chun9622/LLMvul`` if local copies are absent.
    """
    vul_local = _local_data("primevul236.jsonl")
    nonvul_local = _local_data("primenonvul236.jsonl")
    if vul_local and nonvul_local:
        return vul_local, nonvul_local

    print("[INFO] Local data not found – downloading from HuggingFace …")
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(DATASET_HF_ID)

    data_dir = os.path.join(ROOT_DIR, "data")
    vul_path = os.path.join(data_dir, "primevul236.jsonl")
    nonvul_path = os.path.join(data_dir, "primenonvul236.jsonl")

    # Try named splits first, fall back to filtering a single split
    if "vulnerable" in ds and "non_vulnerable" in ds:
        _hf_to_jsonl(ds["vulnerable"], vul_path)
        _hf_to_jsonl(ds["non_vulnerable"], nonvul_path)
    elif "train" in ds:
        split = ds["train"]
        _hf_to_jsonl((r for r in split if r.get("target") == 1), vul_path)
        _hf_to_jsonl((r for r in split if r.get("target") == 0), nonvul_path)
    else:
        split = ds[list(ds.keys())[0]]
        _hf_to_jsonl((r for r in split if r.get("target") == 1), vul_path)
        _hf_to_jsonl((r for r in split if r.get("target") == 0), nonvul_path)

    print(f"[INFO] Data saved: {vul_path}, {nonvul_path}")
    return vul_path, nonvul_path


def ensure_tp_tn_jsonl() -> str | None:
    """
    Return path to tp_tn_samples.jsonl if it exists locally.
    This file is produced by prime.py (or can be placed in data/ manually).
    Returns None if not found (caller should handle gracefully).
    """
    p = _local_data("tp_tn_samples.jsonl")
    if p is None:
        print(
            "[WARN] data/tp_tn_samples.jsonl not found. "
            "Run scripts/prime.py first and copy TP/TN results to data/."
        )
    return p


def ensure_neuron_analysis_json() -> str | None:
    """Return path to neuron_analysis.json or None."""
    p = _local_data("neuron_analysis.json")
    if p is None:
        print(
            "[WARN] data/neuron_analysis.json not found. "
            "Falling back to default neuron indices."
        )
    return p


# ── TransformerLens patch for fine-tuned Gemma ──────────────────────────────

def apply_model_patches(model_path: str = MODEL_HF_ID):
    """
    Patch transformer_lens so it recognises our fine-tuned Gemma as
    'google/gemma-2-2b' (its base architecture).
    Must be called BEFORE loading ReplacementModel.
    """
    try:
        import transformer_lens.loading_from_pretrained as loading  # type: ignore

        _orig_name = loading.get_official_model_name
        loading.get_official_model_name = (
            lambda mn: "google/gemma-2-2b" if mn == model_path else _orig_name(mn)
        )

        _orig_cfg = loading.get_pretrained_model_config

        def _patched_cfg(mn, **kw):
            if mn == model_path:
                from transformers import AutoConfig  # type: ignore
                return AutoConfig.from_pretrained(mn)
            return _orig_cfg(mn, **kw)

        loading.get_pretrained_model_config = _patched_cfg
    except Exception as e:
        print(f"[WARN] Could not apply model patches: {e}")
