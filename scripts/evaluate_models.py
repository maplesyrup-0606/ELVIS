# Created by jing at 26.02.25
import argparse
from scripts import config
from scripts.baseline_models import vit
from scripts.baseline_models import gpt5
from scripts.baseline_models import llama
from scripts.baseline_models import llava
from scripts.baseline_models import internVL
from scripts.baseline_models import grm
from scripts.baseline_models import qwen
from scripts.baseline_models import deepseek
from scripts.baseline_models import phi4
import torch
import os

# List of baseline models
baseline_models = {
    "vit": vit.run_vit,
    "gpt5": gpt5.run_gpt5,
    "gpt5_no_principle": gpt5.run_gpt5_no_principle,
    "gpt5_grp": gpt5.run_gpt5_grouping_zero_shot,
    "grm_grp": grm.run_grm_grouping,
    "neumann":grm.run_neumann,
    "internVL": internVL.run_internVL,
    "internVL_no_principle": internVL.run_internVL_no_principle_given,
    "internVL_X": internVL.run_internVL_X,
    "internVL_X_no_principle": internVL.run_internVL_X_no_principle_given,
    "llava": llava.run_llava,
    "llava_no_principle": llava.run_llava_no_principle,
    # --- baseline (few-shot with principle name) ---
    "internVL_2B_baseline":  internVL.run_internVL_2B_baseline,
    "internVL_8B_baseline":  internVL.run_internVL_8B_baseline,
    "internVL_14B_baseline": internVL.run_internVL_14B_baseline,
    "internVL_38B_baseline": internVL.run_internVL_38B_baseline,
    "llava_baseline":        llava.run_llava_baseline,
    # --- Qwen3-VL ---
    "qwen3_2B_baseline":     qwen.run_qwen3_2B_baseline,
    "qwen3_4B_baseline":     qwen.run_qwen3_4B_baseline,
    "qwen3_8B_baseline":     qwen.run_qwen3_8B_baseline,
    "qwen3_32B_baseline":    qwen.run_qwen3_32B_baseline,
    # --- DeepSeek-VL2 ---
    "deepseek_tiny_baseline":  deepseek.run_deepseek_tiny_baseline,
    "deepseek_small_baseline": deepseek.run_deepseek_small_baseline,
    "deepseek_full_baseline":  deepseek.run_deepseek_full_baseline,
    # --- Phi-4 multimodal ---
    "phi4_baseline":           phi4.run_phi4_baseline,
    # --- zero-shot named (principle name given, no examples) ---
    "internVL_zs_named":     internVL.run_internVL_zs_named,
    "internVL_8B_zs_named":  internVL.run_internVL_8B_zs_named,
    "internVL_14B_zs_named": internVL.run_internVL_14B_zs_named,
    "internVL_38B_zs_named": internVL.run_internVL_38B_zs_named,
    "internVL_X_zs_named":   internVL.run_internVL_X_zs_named,
    "llava_zs_named":        llava.run_llava_zs_named,
    # --- zero-shot blind (no principle name, no examples) ---
    "internVL_zs_blind":     internVL.run_internVL_zs_blind,
    "internVL_8B_zs_blind":  internVL.run_internVL_8B_zs_blind,
    "internVL_14B_zs_blind": internVL.run_internVL_14B_zs_blind,
    "internVL_38B_zs_blind": internVL.run_internVL_38B_zs_blind,
    "internVL_X_zs_blind":   internVL.run_internVL_X_zs_blind,
    "llava_zs_blind":        llava.run_llava_zs_blind,
}


def evaluate_model(model, img_size, principle, batch_size, data_path, device, img_num, epochs, start_num, task_num):
    print(f"{principle} Evaluating on {device}...")
    model(data_path, img_size, principle, batch_size, device=device, img_num=img_num, epochs=epochs, start_num=start_num, task_num=task_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_num", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224, choices=[224, 448, 1024])
    parser.add_argument("--task_num", type=str, default="full")
    parser.add_argument("--start_num", type=int, default=0)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    # Determine device based on device_id flag
    if args.device_id is not None and torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    else:
        device = "cpu"

    # When ELVIS_DATA is set, data is structured as {principle}/test/ directly.
    # The res_{img_size}_pin_False subdirectory only exists in the Docker/remote layout.
    if os.getenv("ELVIS_DATA"):
        data_path = config.get_raw_patterns_path() / args.principle
    else:
        data_path = config.get_raw_patterns_path(args.remote) / f"res_{args.img_size}_pin_False" / args.principle

    print(f"Starting model evaluations with data from {data_path}...")
    model = baseline_models[args.model]
    evaluate_model(model, args.img_size, args.principle, args.batch_size, data_path, device, args.img_num, args.epochs, args.start_num, args.task_num)

    print("All model evaluations completed.")
