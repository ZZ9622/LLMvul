#!/usr/bin/env python3
"""
Demo for reviewers: run on first 5 samples from the HuggingFace dataset and output all test results quickly.
Usage: from repo root, run:
    python demo/demo.py
   or
    python -m demo.demo

Requires: pip install -r requirements.txt, and optionally circuit-tracer for full attribution.
Model and dataset are downloaded from HuggingFace (Chun9622/llmvul-finetuned-gemma, Chun9622/LLMvul).
"""
import os
import sys
import json
import subprocess

# Ensure we run from repo root
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEMO_DIR)
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

OUTPUT_DIR = os.path.join(DEMO_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("LLMvul Demo – first 5 samples, full pipeline (prediction + L0 attribution)")
    print("=" * 60)
    env = os.environ.copy()
    env["DEMO_LIMIT"] = "5"
    env["LLMVUL_OUTPUT_DIR"] = os.path.join(REPO_ROOT, "demo", "output")
    script = os.path.join(REPO_ROOT, "scripts", "prime.py")
    if not os.path.isfile(script):
        print(f"Error: {script} not found. Run from repo root.")
        sys.exit(1)
    print(f"Output directory: {env['LLMVUL_OUTPUT_DIR']}")
    print("Running prime.py (model and dataset will be downloaded from HuggingFace on first run)...")
    print()
    ret = subprocess.run([sys.executable, script], env=env, cwd=REPO_ROOT)
    if ret.returncode != 0:
        print("Prime script exited with code", ret.returncode)
        sys.exit(ret.returncode)
    # Find latest log dir under demo/output
    log_base = os.path.join(env["LLMVUL_OUTPUT_DIR"], "log")
    if os.path.isdir(log_base):
        subdirs = sorted([d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))], reverse=True)
        if subdirs:
            latest = os.path.join(log_base, subdirs[0])
            out_json = os.path.join(latest, "out.json")
            pred_json = os.path.join(latest, "all_predictions.json")
            print()
            print("=" * 60)
            print("Demo results")
            print("=" * 60)
            print(f"Log/plot directory: {latest}")
            if os.path.isfile(out_json):
                with open(out_json) as f:
                    data = json.load(f)
                m = data.get("classification_metrics", {})
                if m:
                    print(f"Total samples: {m.get('total_samples', 'N/A')}")
                    print(f"Accuracy (non-unknown): {m.get('accuracy', 0):.4f}")
                    print(f"Precision: {m.get('precision', 0):.4f}")
                    print(f"Recall: {m.get('recall', 0):.4f}")
                    print(f"F1: {m.get('f1_score', 0):.4f}")
            if os.path.isfile(pred_json):
                print(f"Predictions: {pred_json}")
            print("Done.")

if __name__ == "__main__":
    main()
