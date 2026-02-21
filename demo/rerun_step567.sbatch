#!/bin/bash
#SBATCH --job-name=llmvul_567
#SBATCH --output=slurm_%j_%x.out
#SBATCH --error=slurm_%j_%x.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=lu2025-2-37
#SBATCH --partition=gpua100i

# Reruns only Steps 5, 6, 7 – skips already-completed steps 1-4.
# Submit from repo root: sbatch demo/rerun_step567.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
OUT_DIR="${REPO_ROOT}/out"

export LLMVUL_OUTPUT_DIR="${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CT_ENV_PYTHON="/home/chun7871/.conda/envs/ct-env/bin/python"
if [ -x "${CT_ENV_PYTHON}" ]; then
    PYTHON="${CT_ENV_PYTHON}"
    echo "[ENV] Using ct-env: ${PYTHON}"
else
    PYTHON="python3"
    echo "[WARN] ct-env not found, using system python3"
fi

# Locate most recent prime output JSON for step 7
PRIME_OUT_JSON="$(find "${OUT_DIR}" -name "out.json" 2>/dev/null | sort -r | head -1)"

echo "[SLURM] Job ID     : ${SLURM_JOB_ID:-unknown}"
echo "[SLURM] Submit dir : ${REPO_ROOT}"
echo "[INFO]  Prime JSON : ${PRIME_OUT_JSON:-<not found>}"
echo "============================================================"
echo " LLMvul – Rerun Steps 5, 6, 7"
echo " Started: $(date)"
echo "============================================================"

run_step() {
    local step_num="$1" step_name="$2" script="$3"
    shift 3
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " Step ${step_num}: ${step_name}"
    echo " Started: $(date)"
    echo "──────────────────────────────────────────────────────────"
    ${PYTHON} "${SCRIPTS_DIR}/${script}" "$@"
    local rc=$?
    if [ ${rc} -eq 0 ]; then
        echo " [OK] Step ${step_num} finished at $(date)"
    else
        echo " [WARN] Step ${step_num} exited with code ${rc} – continuing…"
    fi
}

run_step 5 "Advanced Statistical Analysis" "advanced_statistical.py"
run_step 6 "Circuit Visualization"         "circuitplot.py"

if [ -n "${PRIME_OUT_JSON}" ]; then
    run_step 7 "Circuit Analysis (post-hoc)" "analyze_circuits.py" "${PRIME_OUT_JSON}"
else
    echo "[SKIP] Step 7: no prime output JSON found."
fi

echo ""
echo "============================================================"
echo " Finished: $(date)"
echo "============================================================"
