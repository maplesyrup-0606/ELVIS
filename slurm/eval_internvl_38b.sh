#!/bin/bash
#SBATCH --job-name=elvis-internvl-38b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=h100:1

source $HOME/venv/bin/activate

PRINCIPLES="proximity similarity closure symmetry continuity"

for mode in zs_named zs_blind; do
    for principle in $PRINCIPLES; do
        echo "=== internVL_38B_${mode} | ${principle} ==="
        python -m scripts.evaluate_models \
            --model internVL_38B_${mode} \
            --principle $principle \
            --img_num 3 \
            --img_size 224 \
            --batch_size 1 \
            --device_id 0 \
            --remote
    done
done
