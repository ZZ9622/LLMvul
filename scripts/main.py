#!/usr/bin/env python3
"""
LLMvul – unified entry point.

Usage:
    python scripts/main.py <keyword> [extra args...]

Keywords:
    prime               Main pipeline: vulnerability prediction + L0 attribution
    attention           Attention head importance analysis
    causal_patching     Causal patching experiment (safe→vulnerable)
    causal_validation   Causal validation / ablation studies
    advanced_statistical Advanced statistical analysis (L2 norms, effect sizes)
    analyze_circuits    Post-hoc circuit analysis (pass a prime output JSON path)
    circuitplot         Circuit attribution & visualization for specific samples

Examples:
    python scripts/main.py prime
    python scripts/main.py attention
    python scripts/main.py analyze_circuits ./out/log/20250101_120000/out.json
    python scripts/main.py circuitplot
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

KEYWORD_MAP = {
    "prime":                "prime.py",
    "attention":            "attention_analysis.py",
    "attention_analysis":   "attention_analysis.py",
    "causal_patching":      "causal_patching.py",
    "causal_validation":    "causal_validation.py",
    "advanced_statistical": "advanced_statistical.py",
    "analyze_circuits":     "analyze_circuits.py",
    "circuitplot":          "circuitplot.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    keyword = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if keyword not in KEYWORD_MAP:
        print(f"[ERROR] Unknown keyword '{keyword}'. Choose from: {', '.join(sorted(KEYWORD_MAP))}")
        sys.exit(1)

    script = os.path.join(_SCRIPT_DIR, KEYWORD_MAP[keyword])
    if not os.path.exists(script):
        print(f"[ERROR] Script not found: {script}")
        sys.exit(1)

    # Forward extra args via sys.argv so scripts that read sys.argv work correctly
    sys.argv = [script] + extra_args

    print(f"[INFO] Running: {script}")
    with open(script, "r", encoding="utf-8") as fh:
        code = fh.read()

    # Execute in a fresh namespace that has __file__ set to the target script
    ns = {"__file__": script, "__name__": "__main__"}
    exec(compile(code, script, "exec"), ns)


if __name__ == "__main__":
    main()
