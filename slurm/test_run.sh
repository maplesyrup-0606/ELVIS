#!/bin/bash
#SBATCH --job-name=elvis-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mercurymcindoe@gmail.com

module load python/3.11.5 cuda/12.6 opencv/4.13.0
source $SCRATCH/venv/elvis/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export ELVIS_DATA=$SCRATCH/ELVIS/data/res_448_pin_False
export ELVIS_RESULTS=$SCRATCH/ELVIS/results
cd $SCRATCH/ELVIS

python -m scripts.evaluate_models \
    --model internVL_zs_named \
    --principle proximity \
    --img_num 3 \
    --img_size 448 \
    --batch_size 1 \
    --device_id 0 \
    --task_num 2
