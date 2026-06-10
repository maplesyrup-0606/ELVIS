#!/bin/bash
# Submit all ELVIS zero-shot evaluation jobs.
# Run from the repo root on the Fir login node:
#   bash slurm/submit_all.sh

mkdir -p logs

SCRIPTS=(
    slurm/eval_internvl_2b.sh
    slurm/eval_internvl_8b.sh
    slurm/eval_internvl_14b.sh
    slurm/eval_internvl_38b.sh
    slurm/eval_internvl_78b.sh
    slurm/eval_llava_7b.sh
)

for script in "${SCRIPTS[@]}"; do
    job_id=$(sbatch "$script" | awk '{print $4}')
    echo "Submitted $script → job $job_id"
done
