# Created by MacBook Pro at 16.07.25

import torch
import argparse
import json
import os
import wandb
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime, date
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from scripts import config
from scripts.baseline_models import conversations

from scripts.utils import data_utils, file_utils

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def init_wandb(batch_size):
    wandb.init(project="LLM-Gestalt-Patterns", config={"batch_size": batch_size})


def load_deepseek_model(device):
    model_name = "deepseek-ai/deepseek-vl2-small"
    cache_dir = "/models/deepseek_cache"  # Ensure this is mounted in Docker

    processor = DeepseekVLV2Processor.from_pretrained(model_name, cache_dir=cache_dir)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    # model = model.to(device).eval()
    tokenizer = processor.tokenizer
    return model, processor, tokenizer


def load_images(image_dir, num_samples=5):
    # print("img dir " + str(image_dir))
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return image_paths
    # return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]




def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
    # Prepare conversation as per official example
    # print("img path:" + str(train_negative[0]))
    conversation = conversations.deepseek_conversation(train_positive, train_negative, principle)
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    )

    model_device = next(model.parameters()).device

    # Move all tensors in prepare_inputs to model_device
    for attr in prepare_inputs.__dict__:
        v = getattr(prepare_inputs, attr)
        if isinstance(v, torch.Tensor):
            setattr(prepare_inputs, attr, v.to(model_device))

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )
    answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


# def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
#     # Prepare conversation history
#     conversations = [
#         {
#             "role": "system",
#             "content": f"You are an AI analyzing Gestalt patterns. Principle: {principle}."
#         },
#     ]
#
#     # Add positive examples
#     for img in train_positive:
#         conversations.append({
#             "role": "user",
#             "content": [{"type": "image", "image": img}, {"type": "text", "text": "Positive example"}]
#         })
#
#     # Add negative examples
#     for img in train_negative:
#         conversations.append({
#             "role": "user",
#             "content": [{"type": "image", "image": img}, {"type": "text", "text": "Negative example"}]
#         })
#
#     # Final reasoning prompt
#     conversations.append({
#         "role": "user",
#         "content": "What rule distinguishes positive from negative examples?"
#     })
#
#     # Process and generate
#     inputs = processor(conversations).to(device)
#     inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
#     outputs = model.generate(**inputs, max_new_tokens=512)
#     return processor.decode(outputs[0], skip_special_tokens=True)


def evaluate_deepseek(model, processor, test_images, logic_rules, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    torch.cuda.empty_cache()

    for image, label in test_images:

        conversation = conversations.deepseek_eval_conversation(image, logic_rules)
        pil_images = load_pil_images(conversation)


        inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )

        model_device = next(model.parameters()).device

        # Move all tensors in prepare_inputs to model_device
        for attr in inputs.__dict__:
            v = getattr(inputs, attr)
            if isinstance(v, torch.Tensor):
                setattr(inputs, attr, v.to(model_device))

        inputs_embeds = model.prepare_inputs_embeds(**inputs)

        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


        # # inputs = tokenizer.apply_chat_template(
        # #     conversation,
        # #     add_generation_prompt=True,
        # #     return_tensors="pt"
        # # ).to(device)
        #
        # generate_ids = model.generate(
        #     inputs,
        #     max_new_tokens=10,  # Short output expected
        #     do_sample=False
        # )
        #
        #
        # processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # prediction_label = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        #
        #

        prediction_label = answer.split("response:")[-1].strip().lower()
        # prediction_label = prediction_label.split("response:")[-1].strip().lower()
        # print(f"({label}) evaluating answer: {prediction_label}")
        predicted_label = 1 if "positive" in prediction_label else 0
        all_labels.append(label)
        all_predictions.append(predicted_label)

        total += 1
        correct += (predicted_label == label)

    accuracy = 100 * correct / total if total > 0 else 0

    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)

    wandb.log({
        f"{principle}/test_accuracy": accuracy,
        f"{principle}/f1_score": f1_score,
        f"{principle}/precision": precision,
        f"{principle}/recall": recall
    })

    print(f"({principle}) Test Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return accuracy, f1_score, precision, recall


def run_deepseek(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size)
    model, processor, tokenizer = load_deepseek_model(device)
    principle_path = Path(data_path)

    pattern_folders = sorted((principle_path / "train").iterdir())
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return

    pattern_folders = sorted(
        [f for f in (principle_path / "train").iterdir() if f.is_dir() and not f.name.startswith('.')]
    )

    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    for pattern_folder in pattern_folders:
        train_positive = load_images(pattern_folder / "positive", img_num)
        train_negative = load_images(pattern_folder / "negative", img_num)
        test_positive = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative = load_images((principle_path / "test" / pattern_folder.name) / "negative", img_num)

        logic_rules = infer_logic_rules(model, processor, train_positive, train_negative, device, principle)

        test_images = [(img, 1) for img in test_positive] + [(img, 0) for img in test_negative]
        # print("len test images", len(test_images))
        accuracy, f1, precision, recall = evaluate_deepseek(model, processor, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "logic_rules": logic_rules,
            "precision": precision,
            "recall": recall
        }
        total_accuracy.append(accuracy)
        total_f1.append(f1)
        total_precision_scores.append(precision)
        total_recall_scores.append(recall)

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    results["average"] = {"accuracy": avg_accuracy, "f1_score": avg_f1}
    results_path = Path(data_path) / f"deepseek_{principle}.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Evaluation complete. Results saved to evaluation_results.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1


# ---------- new baseline runners with per-sample tracking ----------

def load_deepseek_model_by_id(model_id):
    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    return model, processor


def load_images_pil(image_dir, img_size, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(p).convert("RGB").resize((img_size, img_size)) for p in image_paths]


def _deepseek_generate(model, processor, conversation, pil_images, max_new_tokens):
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    )
    model_device = next(model.parameters()).device
    for attr in prepare_inputs.__dict__:
        v = getattr(prepare_inputs, attr)
        if isinstance(v, torch.Tensor):
            setattr(prepare_inputs, attr, v.to(model_device))
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


def infer_logic_rules_new(model, processor, train_positive, train_negative, principle):
    conversation = conversations.deepseek_conversation(train_positive, train_negative, principle)
    pil_images = train_positive + train_negative
    return _deepseek_generate(model, processor, conversation, pil_images, max_new_tokens=1024)


def evaluate_llm_new(model, processor, test_images, logic_rules, principle):
    correct, total = 0, 0
    all_labels, all_predictions, samples = [], [], []
    for image, label, img_id in test_images:
        conversation = conversations.deepseek_eval_conversation(image, logic_rules)
        response = _deepseek_generate(model, processor, conversation, [image], max_new_tokens=32)
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


def _run_deepseek_baseline(model_id, model_name, data_path, img_size, principle,
                           batch_size, img_num, start_num, task_num):
    principle_path = Path(data_path)
    pattern_folders = sorted(file_utils.list_folders(str(principle_path / "train")))
    if not pattern_folders:
        print("No pattern folders found in", principle_path / "train")
        return

    if task_num != "full":
        pattern_folders = pattern_folders[start_num:start_num + int(task_num)]

    model, processor = load_deepseek_model_by_id(model_id)

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
        train_positive = load_images_pil(pattern_folder / "positive", img_size, img_num)
        train_negative = load_images_pil(pattern_folder / "negative", img_size, img_num)
        test_positive = load_images_pil((principle_path / "test" / pattern_folder.name) / "positive", img_size, img_num)
        test_negative = load_images_pil((principle_path / "test" / pattern_folder.name) / "negative", img_size, img_num)

        logic_rules = infer_logic_rules_new(model, processor, train_positive, train_negative, principle)
        test_images = (
            [(img, 1, f"positive_{i}") for i, img in enumerate(test_positive)] +
            [(img, 0, f"negative_{i}") for i, img in enumerate(test_negative)]
        )
        accuracy, f1, precision, recall, samples = evaluate_llm_new(
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


def run_deepseek_tiny_baseline(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    return _run_deepseek_baseline("deepseek-ai/deepseek-vl2-tiny", "DeepSeek-VL2-Tiny",
                                   data_path, img_size, principle, batch_size, img_num, start_num, task_num)


def run_deepseek_small_baseline(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    return _run_deepseek_baseline("deepseek-ai/deepseek-vl2-small", "DeepSeek-VL2-Small",
                                   data_path, img_size, principle, batch_size, img_num, start_num, task_num)


def run_deepseek_full_baseline(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    return _run_deepseek_baseline("deepseek-ai/deepseek-vl2", "DeepSeek-VL2",
                                   data_path, img_size, principle, batch_size, img_num, start_num, task_num)
