import torch
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime, date
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from scripts import config
from scripts.baseline_models import conversations
from scripts.utils import data_utils, file_utils

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
USER_PROMPT = "<|user|>"
ASSISTANT_PROMPT = "<|assistant|>"
PROMPT_SUFFIX = "<|end|>"


def load_phi4_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation="eager",
    ).eval()
    model.load_adapter(MODEL_ID, adapter_name="vision", device_map="auto",
                       adapter_kwargs={"subfolder": "vision-lora"})
    model.set_adapter("vision")
    generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    return model, processor, generation_config


def load_images(image_dir, img_size, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(p).convert("RGB").resize((img_size, img_size)) for p in image_paths]


def _phi4_generate(model, processor, generation_config, prompt, images, max_new_tokens):
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    generate_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, generation_config=generation_config
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def infer_logic_rules(model, processor, generation_config, train_positive, train_negative, principle):
    images = train_positive + train_negative
    text = conversations.phi4_stage1_prompt(principle, len(train_positive), len(train_negative))
    prompt = f"{USER_PROMPT}{text}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    return _phi4_generate(model, processor, generation_config, prompt, images, max_new_tokens=1024)


def evaluate_llm(model, processor, generation_config, test_images, logic_rules, principle):
    correct, total = 0, 0
    all_labels, all_predictions, samples = [], [], []
    for image, label, img_id in test_images:
        text = conversations.phi4_eval_prompt(logic_rules)
        prompt = f"{USER_PROMPT}{text}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
        response = _phi4_generate(model, processor, generation_config, prompt, [image], max_new_tokens=32)
        print(f"Answer: {response}")
        predicted_label = 1 if "positive" in response.lower() else 0
        all_labels.append(label)
        all_predictions.append(predicted_label)
        samples.append({"id": img_id, "label": label, "predicted": predicted_label, "response": response})
        total += 1
        correct += (predicted_label == label)

    accuracy = 100 * correct / total if total > 0 else 0
    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)
    print(f"({principle}) Acc: {accuracy:.2f}% | F1: {f1_score:.4f} | P: {precision:.4f} | R: {recall:.4f}")
    return accuracy, f1_score, precision, recall, samples


def run_phi4_baseline(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    principle_path = Path(data_path)
    pattern_folders = sorted(file_utils.list_folders(str(principle_path / "train")))
    if not pattern_folders:
        print("No pattern folders found in", principle_path / "train")
        return

    if task_num != "full":
        pattern_folders = pattern_folders[start_num:start_num + int(task_num)]

    model, processor, generation_config = load_phi4_model()

    date_str = date.today().strftime("%Y%m%d")
    output_dir = config.get_results_path(principle) / "baseline" / date_str
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Phi4-multimodal_baseline_{img_size}_{timestamp}_img_num_{img_num}.json"
    tmp_path = output_dir / f"{filename}.tmp.json"
    final_path = output_dir / filename

    total_accuracy, total_f1, total_precision, total_recall = [], [], [], []
    results = {}

    for pattern_folder in tqdm(pattern_folders):
        print(f"Evaluating pattern: {pattern_folder.name}")
        train_positive = load_images(pattern_folder / "positive", img_size, img_num)
        train_negative = load_images(pattern_folder / "negative", img_size, img_num)
        test_positive = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_size, img_num)
        test_negative = load_images((principle_path / "test" / pattern_folder.name) / "negative", img_size, img_num)

        logic_rules = infer_logic_rules(model, processor, generation_config,
                                        train_positive, train_negative, principle)
        test_images = (
            [(img, 1, f"positive_{i}") for i, img in enumerate(test_positive)] +
            [(img, 0, f"negative_{i}") for i, img in enumerate(test_negative)]
        )
        accuracy, f1, precision, recall, samples = evaluate_llm(
            model, processor, generation_config, test_images, logic_rules, principle)

        results[pattern_folder.name] = {
            "accuracy": accuracy, "f1_score": f1,
            "precision": precision, "recall": recall,
            "logic_rules": logic_rules, "samples": samples,
        }
        total_accuracy.append(accuracy)
        total_f1.append(f1)
        total_precision.append(precision)
        total_recall.append(recall)

        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=4)
        torch.cuda.empty_cache()

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0
    os.replace(tmp_path, final_path)
    print(f"Results saved to {final_path}")
    print(f"Overall Avg Acc: {avg_accuracy:.2f}% | Avg F1: {avg_f1:.4f}")
    return avg_accuracy, avg_f1
