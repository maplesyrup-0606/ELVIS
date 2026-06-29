import torch
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime, date
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from scripts import config
from scripts.utils import data_utils, file_utils

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_qwen_model(model_id):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_images(image_dir, img_size, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(p).convert("RGB").resize((img_size, img_size)) for p in image_paths]


def _qwen_generate(model, processor, messages, max_new_tokens):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def infer_logic_rules(model, processor, train_positive, train_negative, principle):
    content = []
    for img in train_positive:
        content.append({"type": "image", "image": img})
    for img in train_negative:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": (
        f"You are an AI reasoning about visual patterns based on Gestalt principles.\n"
        f"Principle: {principle}\n\n"
        f"The first {len(train_positive)} images are Positive examples, "
        f"the next {len(train_negative)} are Negative examples.\n"
        f"Please state the logic/rule that distinguishes them. "
        f"Focus on the Gestalt principle of {principle}."
    )})
    messages = [{"role": "user", "content": content}]
    return _qwen_generate(model, processor, messages, max_new_tokens=1024)


def evaluate_llm(model, processor, test_images, logic_rules, principle):
    correct, total = 0, 0
    all_labels, all_predictions, samples = [], [], []
    for image, label, img_id in test_images:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Using the following reasoning rules: {logic_rules}. "
                f"Classify this image as Positive or Negative. "
                f"Only answer with positive or negative."
            )},
        ]}]
        response = _qwen_generate(model, processor, messages, max_new_tokens=32)
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


def _run_qwen_baseline(model_id, model_name, data_path, img_size, principle,
                       batch_size, img_num, start_num, task_num):
    principle_path = Path(data_path)
    pattern_folders = sorted(file_utils.list_folders(str(principle_path / "train")))
    if not pattern_folders:
        print("No pattern folders found in", principle_path / "train")
        return

    if task_num != "full":
        pattern_folders = pattern_folders[start_num:start_num + int(task_num)]

    model, processor = load_qwen_model(model_id)

    date_str = date.today().strftime("%Y%m%d")
    output_dir = config.get_results_path(principle) / "baseline" / date_str
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_baseline_{img_size}_{timestamp}_img_num_{img_num}.json"
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

        logic_rules = infer_logic_rules(model, processor, train_positive, train_negative, principle)
        test_images = (
            [(img, 1, f"positive_{i}") for i, img in enumerate(test_positive)] +
            [(img, 0, f"negative_{i}") for i, img in enumerate(test_negative)]
        )
        accuracy, f1, precision, recall, samples = evaluate_llm(
            model, processor, test_images, logic_rules, principle)

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


def run_qwen_3B_baseline(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    return _run_qwen_baseline("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B",
                               data_path, img_size, principle, batch_size, img_num, start_num, task_num)


def run_qwen_7B_baseline(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    return _run_qwen_baseline("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B",
                               data_path, img_size, principle, batch_size, img_num, start_num, task_num)
