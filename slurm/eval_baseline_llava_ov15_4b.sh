#!/bin/bash
#SBATCH --job-name=elvis-baseline-llava-ov15-4b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

module load python/3.11.5 cuda/12.6 opencv/4.13.0
source $SCRATCH/venv/elvis/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export ELVIS_DATA=$SCRATCH/ELVIS/data/res_448_pin_False
export ELVIS_RESULTS=$SCRATCH/ELVIS/results
cd $SCRATCH/ELVIS

PRINCIPLES="proximity similarity closure symmetry continuity"

for principle in $PRINCIPLES; do
    echo "=== llava_ov15_4B_baseline | ${principle} ==="
    python -m scripts.evaluate_models \
        --model llava_ov15_4B_baseline \
        --principle $principle \
        --img_num 3 \
        --img_size 224 \
        --batch_size 1 \
        --device_id 0
done
