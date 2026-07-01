#!/bin/bash
# Submit all ELVIS zero-shot evaluation jobs.
# Run from the repo root on the Fir login node:
#   bash slurm/submit_all.sh

mkdir -p logs

SCRIPTS=(
    # --- zero-shot (done) ---
    # slurm/eval_internvl_2b.sh    # done 2026-06-09
    # slurm/eval_internvl_8b.sh    # done 2026-06-09
    # slurm/eval_internvl_14b.sh   # done 2026-06-09
    # slurm/eval_internvl_38b.sh   # done 2026-06-12
    # slurm/eval_internvl_78b.sh   # skipped
    # slurm/eval_llava_7b.sh       # done 2026-06-09
    # --- baseline ---
    # --- InternVL3 baseline (done 2026-06-29) ---
    # slurm/eval_baseline_2b.sh
    # slurm/eval_baseline_8b.sh
    # slurm/eval_baseline_14b.sh
    # slurm/eval_baseline_38b.sh
    # --- Qwen3-VL baseline ---
    # slurm/eval_baseline_qwen3_2b.sh   # done 2026-06-30
    # slurm/eval_baseline_qwen3_4b.sh   # done 2026-06-30
    slurm/eval_baseline_qwen3_8b.sh
    slurm/eval_baseline_qwen3_32b.sh
    # --- LLaVA-OneVision-1.5 baseline ---
    slurm/eval_baseline_llava_ov15_4b.sh
    slurm/eval_baseline_llava_ov15_8b.sh
)

for script in "${SCRIPTS[@]}"; do
    job_id=$(sbatch --account=def-lsigal "$script" | awk '{print $4}')
    echo "Submitted $script → job $job_id"
done
