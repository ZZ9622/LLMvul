#!/usr/bin/env python3
"""LLMvul entry: run analysis by keyword. Usage: python scripts/main.py <keyword> [extra...]"""
import argparse
import os
import runpy
import sys

TASKS = {
    "prime": "prime.py",
    "attention": "attention_analysis.py",
    "causal_patching": "causal_patching.py",
    "causal_validation": "causal_validation.py",
    "advanced_statistical": "advanced_statistical.py",
    "analyze_circuits": "analyze_circuits.py",
    "circuitplot": "circuitplot.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run LLMvul analysis by keyword.")
    parser.add_argument("task", choices=list(TASKS.keys()), help="Task: " + ", ".join(TASKS.keys()))
    parser.add_argument("extra", nargs="*", help="Extra args (e.g. JSON path for analyze_circuits)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, TASKS[args.task])
    if not os.path.isfile(script_path):
        print(f"[ERROR] Not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    sys.argv = [script_path] + args.extra
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
