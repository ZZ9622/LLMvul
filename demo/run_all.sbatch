#!/bin/bash
#SBATCH --job-name=llmvul_all
#SBATCH --output=slurm_%j_%x.out
#SBATCH --error=slurm_%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# ── Adjust the following two lines for your cluster ──────────────────────────
#SBATCH --account=lu2025-2-37
#SBATCH --partition=gpua100
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
#   prime.py               – main pipeline (prediction + L0 attribution)
#   attention_analysis.py  – attention head importance
#   causal_patching.py     – bidirectional steering experiment
#   causal_validation.py   – ablation / causal validation
#   advanced_statistical.py – L2 norm statistics
#   mlp_neuron_analysis.py – MLP neuron activation analysis (TP/TN)
#   circuit_plot.py        – CWE-based circuit visualisation (circles)
#
# Notes:
#   • Set CONDA_ENV below to match your conda/venv environment.
#   • Set REPO_ROOT if you submit from a different working directory.
# ============================================================================

set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV="ct-env"          # name of your conda environment
# SLURM_SUBMIT_DIR is the directory from which sbatch was called (repo root).
# Do NOT use BASH_SOURCE here – Slurm copies the script to its spool dir.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
DATA_DIR="${REPO_ROOT}/data"
OUT_DIR="${REPO_ROOT}/out"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_DIR}"

echo "[SLURM] Job ID     : ${SLURM_JOB_ID:-unknown}"
echo "[SLURM] Submit dir : ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "[SLURM] Output file: ${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_${SLURM_JOB_ID:-0}_${SLURM_JOB_NAME:-llmvul_all}.out"

# Use ct-env directly (avoid conda init issues inside Slurm nodes).
# Priority: ct-env absolute path → module Anaconda3 → system python
CT_ENV_PYTHON="/home/chun7871/.conda/envs/ct-env/bin/python"

if [ -x "${CT_ENV_PYTHON}" ]; then
    PYTHON="${CT_ENV_PYTHON}"
    echo "[ENV] Using ct-env: ${PYTHON}"
elif module load Anaconda3 2>/dev/null && conda run -n ct-env python --version &>/dev/null; then
    PYTHON="conda run -n ct-env python"
    echo "[ENV] Using conda run ct-env"
else
    PYTHON="python3"
    echo "[WARN] ct-env not found, using system python3 (may lack packages)"
fi

export LLMVUL_CT_ENV_PYTHON="${CT_ENV_PYTHON}"
export LLMVUL_OUTPUT_DIR="${OUT_DIR}"

echo " LLMvul – Full Run"
echo " Repo   : ${REPO_ROOT}"
echo " Output : ${OUT_DIR}"
echo " Started: $(date)"

run_step() {
    local step_name="$1"
    local script="$2"
    shift 2
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " ${step_name}"
    echo " Script : ${script}"
    echo " Started: $(date)"
    echo "──────────────────────────────────────────────────────────"
    ${PYTHON} "${SCRIPTS_DIR}/${script}" "$@"
    local rc=$?
    if [ ${rc} -eq 0 ]; then
        echo " [OK] ${step_name} finished at $(date)"
    else
        echo " [WARN] ${step_name} exited with code ${rc} – continuing…"
    fi
    return ${rc}
}

# ── Prime (main pipeline) ─────────────────────────────────────────────
# Generates: out/.../log/out.json  and  out/.../log/all_predictions.json
run_step "Main Pipeline (prime.py)" "prime.py"

PRIME_OUT_JSON="$(find "${OUT_DIR}" -name "out.json" -newer "${SCRIPTS_DIR}/prime.py" \
    2>/dev/null | sort -r | head -1)"
echo ""
echo "Most recent prime output JSON: ${PRIME_OUT_JSON:-<not found>}"

if [ ! -f "${DATA_DIR}/tp_tn_samples.jsonl" ]; then
    echo ""
    echo "[WARN] data/tp_tn_samples.jsonl not found."
    echo "       Downstream analyses require this file."
    echo "       Either:"
    echo "         a) Extract TP/TN predictions from prime output and save to"
    echo "            ${DATA_DIR}/tp_tn_samples.jsonl"
    echo "         b) Re-run with this file in place."
    echo ""
    echo "Skipping downstream analyses. Running circuit_plot only."
    SKIP_DOWNSTREAM=1
else
    SKIP_DOWNSTREAM=0
fi

if [ "${SKIP_DOWNSTREAM}" -eq 0 ]; then
    run_step "Attention Head Analysis"          "attention_analysis.py"   || true
    run_step "Causal Patching (Bidirectional)" "causal_patching.py"      || true
    run_step "Causal Validation (Ablation)"    "causal_validation.py"    || true
    run_step "Advanced Statistical Analysis"   "advanced_statistical.py" || true
    run_step "MLP Neuron Activation Analysis"  "mlp_neuron_analysis.py"  || true
fi

run_step "Circuit Visualization" "circuit_plot.py" || true

if [ -n "${PRIME_OUT_JSON}" ]; then
    run_step "Circuit Analysis (post-hoc)" "analyze_circuits.py" "${PRIME_OUT_JSON}" || true
else
    echo ""
    echo "[SKIP] analyze_circuits.py: no prime output JSON found."
fi

echo ""
echo " LLMvul – All steps finished"
echo " Output dir : ${OUT_DIR}"
echo " Finished   : $(date)"
