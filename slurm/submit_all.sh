#!/bin/bash
# Submit all ELVIS zero-shot evaluation jobs.
# Run from the repo root on the Fir login node:
#   bash slurm/submit_all.sh

mkdir -p logs

SCRIPTS=(
    # slurm/eval_internvl_2b.sh    # done 2026-06-09
    # slurm/eval_internvl_8b.sh    # done 2026-06-09
    # slurm/eval_internvl_14b.sh   # done 2026-06-09
    slurm/eval_internvl_38b.sh     # failed 2026-06-09 (CUDA init), fixed
    # slurm/eval_internvl_78b.sh     # pending
    # slurm/eval_llava_7b.sh       # done 2026-06-09
)

for script in "${SCRIPTS[@]}"; do
    job_id=$(sbatch --account=def-lsigal "$script" | awk '{print $4}')
    echo "Submitted $script → job $job_id"
done
