# ELVIS — Claude Code Instructions

## What This Is
Research benchmark for evaluating VLMs on Gestalt perceptual principles (proximity, similarity, closure, symmetry, continuity). Published at NeSy 2025. Dataset on HuggingFace (`akweury/ELVIS`).

## Tech Stack
- Python + PyTorch (CUDA 12.1), HuggingFace Transformers 4.51.0
- openai SDK for GPT-5, wandb for experiment tracking, timm for ViT
- Docker for GPU cluster runs

## Build & Run
```bash
pip install -r requirements.txt

# Evaluate a model
python -m scripts.evaluate_models --model <model_key> --principle <principle> --img_num 3 --batch_size 1 --device_id 0

# Generate dataset patterns
python scripts/main.py

# Docker
docker build -t elvis .
```

## Supported Model Keys
Registered in `scripts/evaluate_models.py:14-27`:
`vit`, `gpt5`, `gpt5_no_principle`, `gpt5_grp`, `internVL`, `internVL_no_principle`, `internVL_X`, `internVL_X_no_principle`, `llava`, `llava_no_principle`, `grm_grp`, `neumann`

## Adding a New Model
1. Create `scripts/baseline_models/<model_name>.py` with a `run_<model_name>(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num)` function
2. Register it in `scripts/evaluate_models.py:14-27`
3. Add prompt templates to `scripts/baseline_models/conversations.py` if needed

## Key Files
| File | Purpose |
|---|---|
| `scripts/evaluate_models.py` | Benchmark entry point |
| `scripts/config.py` | Shapes, colors, paths, principle categories |
| `scripts/baseline_models/conversations.py` | Prompt templates for all VLMs |
| `scripts/main.py` | Dataset generation entry point |
| `pure_perception/models/` | ViT fine-tune evaluation (no prompting) |

## Conventions
- Results saved to `/elvis_result/<principle>/` (remote) or `results/` (local)
- All experiments log to WandB — project names follow `<MODEL>-Gestalt-<principle>`
- `--remote` flag switches data paths to Docker mount points
- `_no_principle` model variants withhold the Gestalt principle name from the prompt
