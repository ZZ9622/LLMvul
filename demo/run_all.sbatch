#!/bin/bash
#SBATCH --job-name=llmvul_all
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# ── Adjust the following two lines for your cluster ──────────────────────────
##SBATCH --partition=gpu
##SBATCH --account=your_account
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================================
# LLMvul – run ALL scripts on an HPC/Slurm cluster
#
# Usage (from the repo root):
#   sbatch demo/run_all.sbatch
#
# What this script does:
#   Step 1  prime.py              – main pipeline (prediction + L0 attribution)
#   Step 2  attention_analysis.py – attention head importance
#   Step 3  causal_patching.py    – causal patching experiment
#   Step 4  causal_validation.py  – ablation / causal validation
#   Step 5  advanced_statistical.py – L2 norm statistics
#   Step 6  circuitplot.py        – circuit visualisation for key samples
#
# Notes:
#   • Steps 2–6 require data/tp_tn_samples.jsonl produced by step 1.
#     The script waits for step 1 to finish before continuing.
#   • Set CONDA_ENV below to match your conda/venv environment.
#   • Set REPO_ROOT if you submit from a different working directory.
# ============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV="ct-env"          # name of your conda environment
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
DATA_DIR="${REPO_ROOT}/data"
OUT_DIR="${REPO_ROOT}/out"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_DIR}"

# ── Activate environment ──────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" || true
elif [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export LLMVUL_OUTPUT_DIR="${OUT_DIR}"

echo "============================================================"
echo " LLMvul – Full Run"
echo " Repo   : ${REPO_ROOT}"
echo " Output : ${OUT_DIR}"
echo " Started: $(date)"
echo "============================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
run_step() {
    local step_num="$1"
    local step_name="$2"
    local script="$3"
    shift 3
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " Step ${step_num}: ${step_name}"
    echo " Script : ${script}"
    echo " Started: $(date)"
    echo "──────────────────────────────────────────────────────────"
    python "${SCRIPTS_DIR}/${script}" "$@"
    local rc=$?
    if [ ${rc} -eq 0 ]; then
        echo " [OK] Step ${step_num} finished at $(date)"
    else
        echo " [WARN] Step ${step_num} exited with code ${rc} – continuing…"
    fi
    return ${rc}
}

# ── Step 1: Prime (main pipeline) ─────────────────────────────────────────────
# Generates: out/.../log/out.json  and  out/.../log/all_predictions.json
run_step 1 "Main Pipeline (prime.py)" "prime.py"

# ── Locate the most recent prime output JSON for downstream use ───────────────
PRIME_OUT_JSON="$(find "${OUT_DIR}" -name "out.json" -newer "${SCRIPTS_DIR}/prime.py" \
    2>/dev/null | sort -r | head -1)"
echo ""
echo "Most recent prime output JSON: ${PRIME_OUT_JSON:-<not found>}"

# ── Steps 2–5 require tp_tn_samples.jsonl ──────────────────────────────────────
# Check if user pre-placed the file; if not, warn and skip downstream scripts
if [ ! -f "${DATA_DIR}/tp_tn_samples.jsonl" ]; then
    echo ""
    echo "[WARN] data/tp_tn_samples.jsonl not found."
    echo "       Downstream analyses (steps 2-5) require this file."
    echo "       Either:"
    echo "         a) Extract TP/TN predictions from prime output and save to"
    echo "            ${DATA_DIR}/tp_tn_samples.jsonl"
    echo "         b) Re-run with this file in place."
    echo ""
    echo "Skipping steps 2-5. Running circuitplot only."
    SKIP_DOWNSTREAM=1
else
    SKIP_DOWNSTREAM=0
fi

if [ "${SKIP_DOWNSTREAM}" -eq 0 ]; then
    # ── Step 2: Attention Analysis ─────────────────────────────────────────────
    run_step 2 "Attention Head Analysis" "attention_analysis.py" || true

    # ── Step 3: Causal Patching ────────────────────────────────────────────────
    run_step 3 "Causal Patching" "causal_patching.py" || true

    # ── Step 4: Causal Validation ──────────────────────────────────────────────
    run_step 4 "Causal Validation (Ablation)" "causal_validation.py" || true

    # ── Step 5: Advanced Statistical Analysis ─────────────────────────────────
    run_step 5 "Advanced Statistical Analysis" "advanced_statistical.py" || true
fi

# ── Step 6: Circuit Visualization ─────────────────────────────────────────────
run_step 6 "Circuit Visualization" "circuitplot.py" || true

# ── Step 7: Circuit Analysis (post-hoc, if prime output exists) ───────────────
if [ -n "${PRIME_OUT_JSON}" ]; then
    run_step 7 "Circuit Analysis (post-hoc)" "analyze_circuits.py" "${PRIME_OUT_JSON}" || true
else
    echo ""
    echo "[SKIP] Step 7 (analyze_circuits.py): no prime output JSON found."
fi

echo ""
echo "============================================================"
echo " LLMvul – All steps finished"
echo " Output dir : ${OUT_DIR}"
echo " Finished   : $(date)"
echo "============================================================"
