"""
Grouped bar chart of baseline accuracy per model per principle.

Usage: python -m scripts.per_principle_bars
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parents[1] / "results"
REPORTS_DIR = Path(__file__).parents[1] / "reports"
PRINCIPLES = ["proximity", "similarity", "closure", "symmetry", "continuity"]

MODEL_ORDER = [
    "InternVL3-2B", "InternVL3-8B", "InternVL3-14B", "InternVL3-38B",
    "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen3-VL-32B",
    "LLaVA-OV-7B",
]

FAMILY_COLORS = {
    "InternVL3": ["#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "Qwen3-VL":  ["#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "LLaVA-OV":  ["#a1d99b"],
}


def parse_model_name(path):
    name = path.stem
    for prefix in ("InternVL3", "Qwen3-VL"):
        if name.startswith(prefix):
            return name.split("_baseline_")[0]
    if "llava" in name:
        return "LLaVA-OV-7B"
    return None


def load_all_baseline():
    results = defaultdict(dict)
    latest = {}
    for principle in PRINCIPLES:
        d = RESULTS_DIR / principle / "baseline"
        if not d.exists():
            continue
        for date_dir in sorted(d.iterdir()):
            for f in sorted(date_dir.glob("*.json")):
                if f.name.endswith(".tmp.json"):
                    continue
                model = parse_model_name(f)
                if not model:
                    continue
                key = (model, principle)
                if key in latest and latest[key] > f.name:
                    continue
                latest[key] = f.name
                data = json.load(open(f))
                accs = [v["accuracy"] for v in data.values()]
                results[model][principle] = sum(accs) / len(accs) if accs else 0
    return results


def model_color(model):
    if model.startswith("InternVL3-"):
        idx = ["2B", "8B", "14B", "38B"].index(model.split("-")[1])
        return FAMILY_COLORS["InternVL3"][idx]
    if model.startswith("Qwen3-VL-"):
        idx = ["2B", "4B", "8B", "32B"].index(model.split("-")[2])
        return FAMILY_COLORS["Qwen3-VL"][idx]
    return FAMILY_COLORS["LLaVA-OV"][0]


def plot(baseline, out_path):
    models_present = [m for m in MODEL_ORDER if m in baseline]
    n_models = len(models_present)
    x = np.arange(len(PRINCIPLES))
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models_present):
        vals = [baseline[model].get(p, 0) for p in PRINCIPLES]
        ax.bar(x + (i - n_models / 2 + 0.5) * width, vals, width,
               label=model, color=model_color(model),
               edgecolor="black", linewidth=0.3)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PRINCIPLES])
    ax.set_ylabel("Baseline accuracy (%)")
    ax.set_ylim(45, 85)
    ax.set_title("ELVIS Baseline Accuracy per Model per Principle")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=5, fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", color="lightgray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)
    baseline = load_all_baseline()
    plot(baseline, REPORTS_DIR / "per_principle_bars.png")
