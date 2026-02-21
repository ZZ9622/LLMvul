#!/bin/bash
#SBATCH --job-name=LLMvul_all
#SBATCH --output=slurm_output/run_all_%j.out
#SBATCH --error=slurm_output/run_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100i
#SBATCH --gres=gpu:1
#SBATCH --account=lu2025-2-37
#SBATCH --cpus-per-task=4

# Run all LLMvul scripts in order (for remote HPC cluster).
# Submit from repo root: sbatch demo/run_all.sbatch
# Or from demo: sbatch run_all.sbatch (script detects repo root)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p slurm_output

export PATH="${PATH}"
# Optional: set conda env if needed
# export PATH="/home/$USER/.conda/envs/ct-env/bin:$PATH"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Running full pipeline: prime -> attention -> causal_patching -> causal_validation -> advanced_statistical -> analyze_circuits -> circuitplot"

echo "[1/7] prime.py ..."
python scripts/prime.py

# Latest log dir for downstream scripts (prime writes to log/<ts>/)
LOG_BASE="${LLMVUL_OUTPUT_DIR:-$REPO_ROOT}/log"
LATEST_LOG=$(ls -td "$LOG_BASE"/*/ 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
  echo "[WARN] No log dir found under $LOG_BASE; analyze_circuits will be skipped or use a path manually."
  LATEST_OUT=""
else
  LATEST_OUT="${LATEST_LOG}/out.json"
fi

echo "[2/7] attention_analysis.py ..."
python scripts/attention_analysis.py

echo "[3/7] causal_patching.py ..."
python scripts/causal_patching.py

echo "[4/7] causal_validation.py ..."
python scripts/causal_validation.py

echo "[5/7] advanced_statistical.py ..."
python scripts/advanced_statistical.py

echo "[6/7] analyze_circuits.py ..."
if [ -n "$LATEST_OUT" ] && [ -f "$LATEST_OUT" ]; then
  python scripts/analyze_circuits.py "$LATEST_OUT"
else
  echo "[SKIP] analyze_circuits: no prime out.json (provide path as arg to run manually)."
fi

echo "[7/7] circuitplot.py ..."
python scripts/circuitplot.py

echo "[DONE] All scripts finished."
