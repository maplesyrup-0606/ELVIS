"""
Summarize zero-shot evaluation results across models, principles and modes.
Usage: python -m scripts.analyze_zeroshot
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parents[1] / "results"
PRINCIPLES = ["proximity", "similarity", "closure", "symmetry", "continuity"]


def parse_filename(path):
    name = path.stem
    mode = "zs_named" if "zs_named" in name else "zs_blind"
    if "InternVL3" in name:
        model = name.split("_zs_")[0]
    elif "llava" in name:
        model = "llava"
    else:
        model = "unknown"
    return model, mode


def load_results(json_path):
    with open(json_path) as f:
        data = json.load(f)
    accuracies, ambiguous_total, total_samples = [], 0, 0
    for task_data in data.values():
        accuracies.append(task_data["accuracy"])
        ambiguous_total += task_data.get("ambiguous", 0)
        total_samples += len(task_data.get("samples", []))
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
    return avg_acc, ambiguous_total, total_samples, len(accuracies)


def main():
    summary = defaultdict(lambda: defaultdict(dict))

    for principle in PRINCIPLES:
        zeroshot_dir = RESULTS_DIR / principle / "zeroshot"
        if not zeroshot_dir.exists():
            continue
        for date_dir in sorted(zeroshot_dir.iterdir()):
            for json_file in sorted(date_dir.glob("*.json")):
                model, mode = parse_filename(json_file)
                avg_acc, ambiguous, total_samples, num_tasks = load_results(json_file)
                summary[model][mode][principle] = {
                    "accuracy": avg_acc,
                    "ambiguous": ambiguous,
                    "total_samples": total_samples,
                    "num_tasks": num_tasks,
                }

    models = sorted(summary.keys())
    modes = ["zs_named", "zs_blind"]

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Mode: {mode}")
        print(f"{'='*70}")
        header = f"{'Model':<20}" + "".join(f"{p[:8]:>10}" for p in PRINCIPLES) + f"{'AVG':>10}"
        print(header)
        print("-" * 70)
        for model in models:
            if mode not in summary[model]:
                continue
            accs = []
            row = f"{model:<20}"
            for p in PRINCIPLES:
                if p in summary[model][mode]:
                    acc = summary[model][mode][p]["accuracy"]
                    accs.append(acc)
                    row += f"{acc:>10.1f}"
                else:
                    row += f"{'N/A':>10}"
            avg = sum(accs) / len(accs) if accs else 0
            row += f"{avg:>10.1f}"
            print(row)

    print(f"\n{'='*70}")
    print("Ambiguous responses (excluded from metrics)")
    print(f"{'='*70}")
    for model in models:
        for mode in modes:
            if mode not in summary[model]:
                continue
            total_amb = sum(v["ambiguous"] for v in summary[model][mode].values())
            total_samp = sum(v["total_samples"] for v in summary[model][mode].values())
            pct = 100 * total_amb / total_samp if total_samp else 0
            print(f"{model:<20} / {mode}: {total_amb} ambiguous / {total_samp} total ({pct:.1f}%)")


if __name__ == "__main__":
    main()
