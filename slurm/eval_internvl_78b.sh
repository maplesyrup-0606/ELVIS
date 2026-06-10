#!/bin/bash
#SBATCH --job-name=elvis-internvl-78b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:3

source $HOME/venv/bin/activate

PRINCIPLES="proximity similarity closure symmetry continuity"

for mode in zs_named zs_blind; do
    for principle in $PRINCIPLES; do
        echo "=== internVL_X_${mode} | ${principle} ==="
        python -m scripts.evaluate_models \
            --model internVL_X_${mode} \
            --principle $principle \
            --img_num 3 \
            --img_size 224 \
            --batch_size 1 \
            --device_id 0 \
            --remote
    done
done
