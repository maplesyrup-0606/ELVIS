#!/bin/bash
#SBATCH --job-name=elvis-baseline-qwen3-8b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
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
    echo "=== qwen3_8B_baseline | ${principle} ==="
    python -m scripts.evaluate_models \
        --model qwen3_8B_baseline \
        --principle $principle \
        --img_num 3 \
        --img_size 224 \
        --batch_size 1 \
        --device_id 0
done
