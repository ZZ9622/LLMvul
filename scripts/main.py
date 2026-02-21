#!/usr/bin/env python3
"""
Entry point: run scripts by keyword.
Usage: python scripts/main.py <keyword> [args...]
       or from repo root: python -m scripts.main <keyword> [args...]
"""
import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_MAP = {
    "prime": "prime.py",
    "attention": "attention_analysis.py",
    "causal_patching": "causal_patching.py",
    "causal_validation": "causal_validation.py",
    "advanced_statistical": "advanced_statistical.py",
    "analyze_circuits": "analyze_circuits.py",
    "circuitplot": "circuitplot.py",
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/main.py <keyword> [args...]")
        print("Keywords:", ", ".join(SCRIPT_MAP))
        sys.exit(1)
    keyword = sys.argv[1].lower()
    if keyword not in SCRIPT_MAP:
        print(f"Unknown keyword: {keyword}. Choose from: {', '.join(SCRIPT_MAP)}")
        sys.exit(1)
    script_name = SCRIPT_MAP[keyword]
    script_path = os.path.join(REPO_ROOT, "scripts", script_name)
    if not os.path.isfile(script_path):
        print(f"Script not found: {script_path}")
        sys.exit(1)
    os.chdir(REPO_ROOT)
    code = subprocess.run([sys.executable, script_path] + sys.argv[2:])
    sys.exit(code.returncode)

if __name__ == "__main__":
    main()
