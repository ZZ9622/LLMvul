#!/bin/bash
#SBATCH --job-name=llmvul_step6
#SBATCH --output=slurm_%j_%x.out
#SBATCH --error=slurm_%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=lu2025-2-37
#SBATCH --partition=gpua100

# Reruns only Step 6 (CWE-based circuit visualization).
# Submit from repo root: sbatch demo/rerun_step6.sbatch

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

echo "[SLURM] Job ID     : ${SLURM_JOB_ID:-unknown}"
echo "[SLURM] Submit dir : ${REPO_ROOT}"
echo "============================================================"
echo " LLMvul – Step 6: CWE-Based Circuit Visualization"
echo " Started: $(date)"
echo "============================================================"

${PYTHON} "${SCRIPTS_DIR}/circuitplot.py"
rc=$?
if [ ${rc} -eq 0 ]; then
    echo " [OK] Step 6 finished at $(date)"
else
    echo " [WARN] Step 6 exited with code ${rc}"
fi

echo ""
echo "============================================================"
echo " Finished: $(date)"
echo "============================================================"
