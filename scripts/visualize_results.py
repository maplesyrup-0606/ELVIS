"""
Generate bar chart visualizations of zero-shot evaluation results.

Usage:
    python -m scripts.visualize_results
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parents[1] / "results"
REPORTS_DIR = Path(__file__).parents[1] / "reports"
PRINCIPLES = ["proximity", "similarity", "closure", "symmetry", "continuity"]
MODES = ["baseline", "zs_named", "zs_blind"]
MODE_LABELS = {"baseline": "Baseline", "zs_named": "ZS Named", "zs_blind": "ZS Blind"}
MODE_COLORS = {"baseline": "#4C72B0", "zs_named": "#DD8452", "zs_blind": "#55A868"}

MODEL_ORDER = ["InternVL3-2B", "InternVL3-8B", "InternVL3-14B", "InternVL3-38B", "llava"]
MODEL_LABELS = {
    "InternVL3-2B": "InternVL3\n2B",
    "InternVL3-8B": "InternVL3\n8B",
    "InternVL3-14B": "InternVL3\n14B",
    "InternVL3-38B": "InternVL3\n38B",
    "llava": "LLaVA\n7B",
}


def parse_filename(path):
    name = path.stem
    mode = "zs_named" if "zs_named" in name else "zs_blind"
    if "InternVL3" in name:
        for sep in ("_zs_", "_baseline_"):
            if sep in name:
                model = name.split(sep)[0]
                break
        else:
            model = name
    elif "llava" in name:
        model = "llava"
    else:
        model = "unknown"
    return model, mode


def load_results(json_path):
    with open(json_path) as f:
        data = json.load(f)
    accuracies = [v["accuracy"] for v in data.values()]
    return sum(accuracies) / len(accuracies) if accuracies else 0


def collect_summary():
    summary = defaultdict(lambda: defaultdict(dict))
    for principle in PRINCIPLES:
        for mode_dir in ["baseline", "zeroshot"]:
            results_dir = RESULTS_DIR / principle / mode_dir
            if not results_dir.exists():
                continue
            for date_dir in sorted(results_dir.iterdir()):
                for json_file in sorted(date_dir.glob("*.json")):
                    model, mode = parse_filename(json_file)
                    if mode_dir == "baseline":
                        mode = "baseline"
                    acc = load_results(json_file)
                    summary[model][mode][principle] = acc
    return summary


def avg_across_principles(summary, model, mode):
    accs = [summary[model][mode][p] for p in PRINCIPLES if p in summary[model].get(mode, {})]
    return sum(accs) / len(accs) if accs else None


def plot_main(summary):
    """Option D: grouped by mode, one chart using AVG across principles."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [m for m in MODEL_ORDER if m in summary]
    x = np.arange(len(models))
    width = 0.25
    offsets = [-width, 0, width]

    for i, mode in enumerate(MODES):
        avgs = [avg_across_principles(summary, m, mode) for m in models]
        bars = ax.bar(
            x + offsets[i],
            [a if a is not None else 0 for a in avgs],
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            alpha=0.85,
        )
        for bar, avg in zip(bars, avgs):
            if avg is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{avg:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ELVIS Zero-Shot Evaluation — Average Accuracy Across All Principles")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(40, 80)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = REPORTS_DIR / "avg_accuracy.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_per_principle(summary):
    """Option A: one figure per principle, grouped bars = models, color = mode."""
    models = [m for m in MODEL_ORDER if m in summary]
    x = np.arange(len(models))
    width = 0.25
    offsets = [-width, 0, width]

    for principle in PRINCIPLES:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, mode in enumerate(MODES):
            accs = [summary[m][mode].get(principle) for m in models]
            bars = ax.bar(
                x + offsets[i],
                [a if a is not None else 0 for a in accs],
                width,
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                alpha=0.85,
            )
            for bar, acc in zip(bars, accs):
                if acc is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{acc:.1f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Chance (50%)")
        ax.set_title(f"ELVIS — {principle.capitalize()}", fontsize=12)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
        ax.set_ylim(40, 85)
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = REPORTS_DIR / f"accuracy_{principle}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)
    summary = collect_summary()
    plot_main(summary)
    plot_per_principle(summary)
