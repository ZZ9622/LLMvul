#!/usr/bin/env python3
"""
LLMvul Quick Demo – for reviewers
===================================
Runs inference on the **first 5 samples** from the HuggingFace dataset
``Chun9622/LLMvul`` using the fine-tuned model ``Chun9622/llmvul-finetuned-gemma``
and prints a concise result table together with classification metrics.

Usage (from the repo root):
    python demo/demo.py

Output files are written to  demo/output/<timestamp>/.
Model and dataset are downloaded automatically on first run.

Optional environment variables:
    LLMVUL_DEMO_N      Number of samples per class to test (default: 5)
    LLMVUL_OUTPUT_DIR  Override output directory root
"""

import os
import sys
import json
import time
from datetime import datetime

# ── Repo root ──────────────────────────────────────────────────────────────────
_DEMO_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_DEMO_DIR)
OUTPUT_BASE = os.environ.get("LLMVUL_OUTPUT_DIR", os.path.join(_DEMO_DIR, "output"))

# ── circuit-tracer (optional – needed only for attribution) ───────────────────
_CT = os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer")
if os.path.isdir(_CT) and _CT not in sys.path:
    sys.path.insert(0, _CT)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "Chun9622/llmvul-finetuned-gemma"
DATASET_ID  = "Chun9622/LLMvul"
N_SAMPLES   = int(os.environ.get("LLMVUL_DEMO_N", "5"))
MAX_NEW_TOK = 200
MAX_INPUT   = 512

# ─────────────────────────────────────────────────────────────────────────────
ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(OUTPUT_BASE, ts)
os.makedirs(OUT_DIR, exist_ok=True)

_log_path  = os.path.join(OUT_DIR, "demo.log")
_log_fh    = open(_log_path, "w", encoding="utf-8")


def log(msg: str = ""):
    print(msg, flush=True)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


# ─────────────────────────────────────────────────────────────────────────────
def extract_label(text: str) -> str:
    """Classify model output as 'vul', 'nonvul', or 'unknown'."""
    import re
    if not text:
        return "unknown"
    t = text.lower()

    m = re.search(r'answer:\s*(.{1,120})', t, re.DOTALL)
    if m:
        a = m.group(1).strip()
        if ('vulnerable' in a or 'unsafe' in a) and 'not vulnerable' not in a:
            return "vul"
        if ('safe' in a or 'secure' in a) and 'unsafe' not in a:
            return "nonvul"

    snippet = t[:500]
    vul_kw   = ['vulnerable','unsafe','insecure','flaw','overflow','injection','exploit']
    safe_kw  = ['safe','secure','correct','no vulnerability','no vulnerabilities']
    vc = sum(snippet.count(k) for k in vul_kw)
    sc = sum(snippet.count(k) for k in safe_kw)
    if vc > sc:
        return "vul"
    if sc > vc:
        return "nonvul"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
def load_samples(n_per_class: int):
    """Load first n_per_class vulnerable + n_per_class non-vulnerable samples."""
    log(f"[1/3] Loading dataset ({DATASET_ID}) …")
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(DATASET_ID)

    # Resolve splits
    if "vulnerable" in ds and "non_vulnerable" in ds:
        vul_split   = list(ds["vulnerable"])
        nonvul_split = list(ds["non_vulnerable"])
    elif "train" in ds:
        all_recs = list(ds["train"])
        vul_split    = [r for r in all_recs if r.get("target") == 1]
        nonvul_split = [r for r in all_recs if r.get("target") == 0]
    else:
        key = list(ds.keys())[0]
        all_recs = list(ds[key])
        vul_split    = [r for r in all_recs if r.get("target") == 1]
        nonvul_split = [r for r in all_recs if r.get("target") == 0]

    vul_samples   = vul_split[:n_per_class]
    nonvul_samples = nonvul_split[:n_per_class]

    log(f"  Vulnerable samples : {len(vul_samples)}")
    log(f"  Non-vulnerable     : {len(nonvul_samples)}")

    def make_entry(rec, true_label):
        code = str(rec.get("func", "")).strip()
        prompt = (
            f"Code: {code}\n\n"
            "Question: Is this code safe or vulnerable?\n"
            "Answer:"
        )
        return {
            "idx":        rec.get("idx", -1),
            "true_label": true_label,
            "prompt":     prompt,
            "cwe":        rec.get("cwe", []),
        }

    samples  = [make_entry(r, "vul")    for r in vul_samples]
    samples += [make_entry(r, "nonvul") for r in nonvul_samples]
    return samples


# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load tokenizer + model. Falls back to CPU if no GPU."""
    import torch
    from transformers import AutoTokenizer  # type: ignore

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"\n[2/3] Loading model ({MODEL_ID})  [device={DEVICE}] …")
    log(      "      (First run downloads ~5 GB – subsequent runs use cache)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try circuit-tracer ReplacementModel first; fall back to plain AutoModelForCausalLM
    try:
        import transformer_lens.loading_from_pretrained as _tl  # type: ignore
        _orig = _tl.get_official_model_name
        _tl.get_official_model_name = (
            lambda mn: "google/gemma-2-2b" if mn == MODEL_ID else _orig(mn)
        )
        from circuit_tracer.replacement_model import ReplacementModel  # type: ignore
        dtype = torch.float16 if DEVICE != "cpu" else torch.float32
        model = ReplacementModel.from_pretrained(
            MODEL_ID, transcoder_set="gemma", device=DEVICE,
            torch_dtype=dtype
        )
        model.eval()
        log("  Using circuit-tracer ReplacementModel (attribution-capable).")
        return tokenizer, model, DEVICE, "ct"
    except Exception as e:
        log(f"  circuit-tracer not available ({e}); using standard AutoModel.")

    from transformers import AutoModelForCausalLM  # type: ignore
    dtype = torch.float16 if DEVICE != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, DEVICE, "hf"


# ─────────────────────────────────────────────────────────────────────────────
def run_inference(samples, tokenizer, model, device, model_type):
    """Run batch inference and return list of result dicts."""
    import torch

    log(f"\n[3/3] Running inference on {len(samples)} samples …\n")
    results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        enc = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_INPUT
        )
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            if model_type == "ct":
                out_ids = model.generate(
                    input_ids, max_new_tokens=MAX_NEW_TOK,
                    do_sample=False, verbose=False
                )
            else:
                out_ids = model.generate(
                    input_ids, max_new_tokens=MAX_NEW_TOK,
                    do_sample=False
                )

        gen_ids  = out_ids[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred     = extract_label(gen_text)

        results.append({
            "idx":         sample["idx"],
            "true_label":  sample["true_label"],
            "pred_label":  pred,
            "model_output": gen_text,
            "cwe":         sample["cwe"],
            "correct":     pred == sample["true_label"],
        })

        status = "✓" if pred == sample["true_label"] else "✗"
        elapsed = time.time() - t0
        log(
            f"  [{i+1:02d}/{len(samples)}] idx={sample['idx']:<8} "
            f"true={sample['true_label']:<8} pred={pred:<8}  {status}  "
            f"({elapsed:.1f}s)  output: {gen_text[:60].replace(chr(10),' ')}…"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(results):
    tp = sum(1 for r in results if r["true_label"] == "vul"    and r["pred_label"] == "vul")
    fp = sum(1 for r in results if r["true_label"] == "nonvul" and r["pred_label"] == "vul")
    tn = sum(1 for r in results if r["true_label"] == "nonvul" and r["pred_label"] == "nonvul")
    fn = sum(1 for r in results if r["true_label"] == "vul"    and r["pred_label"] == "nonvul")
    unk = sum(1 for r in results if r["pred_label"] == "unknown")

    total      = len(results)
    classified = tp + fp + tn + fn
    acc        = (tp + tn) / classified if classified else 0.0
    prec       = tp / (tp + fp)         if (tp + fp)  else 0.0
    rec        = tp / (tp + fn)         if (tp + fn)  else 0.0
    f1         = 2*prec*rec/(prec+rec)  if (prec+rec) else 0.0

    return dict(tp=tp, fp=fp, tn=tn, fn=fn, unknown=unk,
                total=total, classified=classified,
                accuracy=acc, precision=prec, recall=rec, f1=f1)


# ─────────────────────────────────────────────────────────────────────────────
def print_table(results, metrics):
    sep  = "─" * 85
    log(f"\n{'═'*85}")
    log("  DEMO RESULTS")
    log(f"{'═'*85}")
    log(f"  {'#':<4} {'idx':<10} {'True':<10} {'Pred':<10} {'OK?':<5} {'Model output (first 45 chars)'}")
    log(sep)
    for i, r in enumerate(results, 1):
        ok  = "✓" if r["correct"] else "✗"
        out = r["model_output"][:45].replace("\n", " ")
        log(f"  {i:<4} {str(r['idx']):<10} {r['true_label']:<10} "
            f"{r['pred_label']:<10} {ok:<5} {out}")
    log(sep)
    log(f"\n  Confusion matrix  (on {metrics['classified']} classified, "
        f"{metrics['unknown']} unknown):")
    log(f"    TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    log(f"\n  Accuracy : {metrics['accuracy']:.3f}")
    log(f"  Precision: {metrics['precision']:.3f}")
    log(f"  Recall   : {metrics['recall']:.3f}")
    log(f"  F1 score : {metrics['f1']:.3f}")
    log(f"\n{'═'*85}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("  LLMvul Quick Demo")
    log(f"  Model  : {MODEL_ID}")
    log(f"  Dataset: {DATASET_ID}")
    log(f"  Samples: {N_SAMPLES} vulnerable + {N_SAMPLES} non-vulnerable")
    log(f"  Output : {OUT_DIR}")
    log("=" * 70)

    samples                  = load_samples(N_SAMPLES)
    tokenizer, model, device, mtype = load_model()
    results                  = run_inference(samples, tokenizer, model, device, mtype)
    metrics                  = compute_metrics(results)
    print_table(results, metrics)

    # ── Save results JSON ──────────────────────────────────────────────────────
    out_json = {
        "model":    MODEL_ID,
        "dataset":  DATASET_ID,
        "n_samples_per_class": N_SAMPLES,
        "timestamp": ts,
        "metrics":  metrics,
        "results":  results,
    }
    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(out_json, fh, indent=2, ensure_ascii=False)
    log(f"\n  Results saved: {json_path}")
    log(f"  Log saved    : {_log_path}")
    _log_fh.close()


if __name__ == "__main__":
    main()
