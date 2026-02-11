#!/bin/bash
#SBATCH --job-name=prime236
#SBATCH --output=slurm_output/slurm236_%j.out
#SBATCH --error=slurm_output/slurm236_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpua100i
#SBATCH --gres=gpu:1
#SBATCH --account=lu2025-2-37
#SBATCH --cpus-per-task=4

# Load environment
export PATH="/home/chun7871/.conda/envs/ct-env/bin:$PATH"

# Navigate to work directory
cd /home/chun7871/lu2025-17-14/chun7871/script

# Run the evaluation script
echo "[INFO] Running script: /home/chun7871/lu2025-17-14/chun7871/script/prime.py"
python prime.py