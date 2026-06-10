#!/bin/bash
#SBATCH --job-name=elvis-llava-7b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1

module load python/3.11.5 cuda/12.6 opencv/4.13.0
source $SCRATCH/venv/elvis/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export ELVIS_DATA=$SCRATCH/ELVIS/data
export ELVIS_RESULTS=$SCRATCH/ELVIS/results
cd $SCRATCH/ELVIS

PRINCIPLES="proximity similarity closure symmetry continuity"

for mode in zs_named zs_blind; do
    for principle in $PRINCIPLES; do
        echo "=== llava_${mode} | ${principle} ==="
        python -m scripts.evaluate_models \
            --model llava_${mode} \
            --principle $principle \
            --img_num 3 \
            --img_size 224 \
            --batch_size 1 \
            --device_id 0

    done
done
