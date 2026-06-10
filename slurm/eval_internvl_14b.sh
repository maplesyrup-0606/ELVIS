#!/bin/bash
#SBATCH --job-name=elvis-internvl-14b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1

source $HOME/venv/bin/activate

PRINCIPLES="proximity similarity closure symmetry continuity"

for mode in zs_named zs_blind; do
    for principle in $PRINCIPLES; do
        echo "=== internVL_14B_${mode} | ${principle} ==="
        python -m scripts.evaluate_models \
            --model internVL_14B_${mode} \
            --principle $principle \
            --img_num 3 \
            --img_size 224 \
            --batch_size 1 \
            --device_id 0 \
            --remote
    done
done
