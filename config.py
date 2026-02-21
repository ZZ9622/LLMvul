# -*- coding: utf-8 -*-
"""
LLMvul config: paths, model and dataset identifiers.
All paths are relative to repo root. Model and dataset are loaded from HuggingFace.
"""
import os

# Repo root (directory containing this config.py)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# HuggingFace model and dataset (downloaded at runtime)
MODEL_NAME = "Chun9622/llmvul-finetuned-gemma"
DATASET_NAME = "Chun9622/LLMvul"

# Base output dir; can override with env LLMVUL_OUTPUT_DIR
_output_base = os.environ.get("LLMVUL_OUTPUT_DIR", REPO_ROOT)
DATA_DIR = os.path.join(REPO_ROOT, "data")
LOG_BASE = os.path.join(_output_base, "log")
PLOT_BASE = os.path.join(_output_base, "plots")

# circuit-tracer (optional): clone into repo and pip install -e circuit-tracer/circuit-tracer
CIRCUIT_TRACER_PATH = os.path.join(REPO_ROOT, "circuit-tracer", "circuit-tracer")


def ensure_output_dirs(log_dir, plot_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return log_dir, plot_dir


def get_dataset_from_huggingface(split="train", vul_limit=None, nonvul_limit=None, total_limit=None):
    """
    Load dataset from HuggingFace (Chun9622/LLMvul) and return vul/nonvul sample lists.
    Each sample: {"idx", "true_label", "prompt"}.
    total_limit: stop after this many samples total (vul+nonvul). Useful for demo.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install: pip install datasets")
    ds = load_dataset(DATASET_NAME, split=split)
    vul_prompts = []
    nonvul_prompts = []
    for i, row in enumerate(ds):
        if total_limit and len(vul_prompts) + len(nonvul_prompts) >= total_limit:
            break
        func = (row.get("func") or "").strip()
        if not func:
            continue
        target = row.get("target", -1)
        if target not in (0, 1):
            continue
        true_label = "vul" if target == 1 else "nonvul"
        prompt = f"Code: {func}\n\nQuestion: Is this code safe or vulnerable?\nAnswer:"
        idx = row.get("idx", i)
        sample = {"idx": idx, "true_label": true_label, "prompt": prompt}
        if target == 1:
            vul_prompts.append(sample)
            if vul_limit and len(vul_prompts) >= vul_limit:
                if nonvul_limit and len(nonvul_prompts) >= nonvul_limit:
                    break
        else:
            nonvul_prompts.append(sample)
            if nonvul_limit and len(nonvul_prompts) >= nonvul_limit:
                if vul_limit and len(vul_prompts) >= vul_limit:
                    break
    return vul_prompts, nonvul_prompts
