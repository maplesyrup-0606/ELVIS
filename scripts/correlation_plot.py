"""
Build benchmark correlation plots.

Usage: python -m scripts.correlation_plot
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = Path(__file__).parents[1] / "results"
REPORTS_DIR = Path(__file__).parents[1] / "reports"
PRINCIPLES = ["proximity", "similarity", "closure", "symmetry", "continuity"]
BENCHMARKS = ["MMMU", "MMStar", "MMBench", "AI2D"]

# Published benchmark scores (verified from technical reports / model cards).
BENCH_SCORES = {
    "InternVL3-2B":  {"MMMU": 48.6, "MMStar": 60.7, "MMBench": 78.6, "AI2D": 78.7},
    "InternVL3-8B":  {"MMMU": 62.7, "MMStar": 68.2, "MMBench": 81.7, "AI2D": 85.2},
    "InternVL3-14B": {"MMMU": 67.1, "MMStar": 68.8, "MMBench": 83.5, "AI2D": 86.0},
    "InternVL3-38B": {"MMMU": 70.1, "MMStar": 71.5, "MMBench": 83.9, "AI2D": 89.1},
    "Qwen3-VL-2B":   {"MMMU": 53.4, "MMStar": 58.3, "MMBench": 78.4, "AI2D": 76.9},
    "Qwen3-VL-4B":   {"MMMU": 67.4, "MMStar": 69.8, "MMBench": 83.9, "AI2D": 84.1},
    "Qwen3-VL-8B":   {"MMMU": 69.6, "MMStar": 70.9, "MMBench": 84.5, "AI2D": 85.7},
    "Qwen3-VL-32B":  {"MMMU": 76.0, "MMStar": 77.7, "MMBench": 87.6, "AI2D": 89.5},
    "LLaVA-OV-7B":   {"MMMU": 48.8, "MMStar": 61.7, "MMBench": 82.5, "AI2D": 81.4},
}

FAMILY = {
    "InternVL3-2B": "InternVL3", "InternVL3-8B": "InternVL3",
    "InternVL3-14B": "InternVL3", "InternVL3-38B": "InternVL3",
    "Qwen3-VL-2B": "Qwen3-VL", "Qwen3-VL-4B": "Qwen3-VL",
    "Qwen3-VL-8B": "Qwen3-VL", "Qwen3-VL-32B": "Qwen3-VL",
    "LLaVA-OV-7B": "LLaVA-OV",
}
FAMILY_COLORS = {"InternVL3": "#4C72B0", "Qwen3-VL": "#DD8452", "LLaVA-OV": "#55A868"}
FAMILY_MARKERS = {"InternVL3": "o", "Qwen3-VL": "s", "LLaVA-OV": "^"}


def parse_model_name(path):
    name = path.stem
    for prefix in ("InternVL3", "Qwen3-VL"):
        if name.startswith(prefix):
            return name.split("_baseline_")[0]
    if "llava" in name:
        return "LLaVA-OV-7B"
    return None


def load_all_baseline():
    """Return dict {model: {principle: accuracy}} using the most recent file per (model, principle)."""
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


def build_summary(baseline):
    """Return list of dicts with model, family, avg_elvis, per-principle scores, and bench scores."""
    rows = []
    for model, per_principle in baseline.items():
        if model not in BENCH_SCORES or not per_principle:
            continue
        avg = sum(per_principle.values()) / len(per_principle)
        row = {
            "model": model,
            "family": FAMILY[model],
            "elvis_avg": avg,
            "elvis_n_principles": len(per_principle),
            **{f"elvis_{p}": per_principle.get(p) for p in PRINCIPLES},
            **BENCH_SCORES[model],
        }
        rows.append(row)
    return rows


def plot_correlations(rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for ax, bench in zip(axes, BENCHMARKS):
        xs = np.array([r[bench] for r in rows])
        ys = np.array([r["elvis_avg"] for r in rows])
        for fam, color in FAMILY_COLORS.items():
            mask = np.array([r["family"] == fam for r in rows])
            if not mask.any():
                continue
            ax.scatter(xs[mask], ys[mask], s=90, color=color,
                       marker=FAMILY_MARKERS[fam], edgecolor="black",
                       linewidth=0.6, label=fam, alpha=0.9)
        for r in rows:
            short = (r["model"].replace("InternVL3-", "IVL3-")
                                .replace("Qwen3-VL-", "Q3-")
                                .replace("LLaVA-OV-", "LO-"))
            ax.annotate(short, (r[bench], r["elvis_avg"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="#333")
        if len(xs) >= 2:
            m, b = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xline, m * xline + b, "--", color="gray", linewidth=1, alpha=0.6)
        pear_r, pear_p = pearsonr(xs, ys)
        spear_r, spear_p = spearmanr(xs, ys)
        ax.text(0.03, 0.97,
                f"Pearson r = {pear_r:.2f} (p={pear_p:.3f})\n"
                f"Spearman r = {spear_r:.2f} (p={spear_p:.3f})",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9))
        ax.axhline(50, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_xlabel(f"{bench} score")
        ax.set_ylabel("ELVIS avg accuracy (%)")
        ax.set_title(f"ELVIS vs {bench}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("ELVIS Baseline Performance vs Standard VLM Benchmarks",
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_heatmap(rows, out_path):
    matrix = np.zeros((len(PRINCIPLES), len(BENCHMARKS)))
    for i, p in enumerate(PRINCIPLES):
        for j, b in enumerate(BENCHMARKS):
            xs = np.array([r[b] for r in rows if r.get(f"elvis_{p}") is not None])
            ys = np.array([r[f"elvis_{p}"] for r in rows if r.get(f"elvis_{p}") is not None])
            if len(xs) >= 2:
                r, _ = pearsonr(xs, ys)
                matrix[i, j] = r
            else:
                matrix[i, j] = np.nan
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(BENCHMARKS)))
    ax.set_xticklabels(BENCHMARKS)
    ax.set_yticks(range(len(PRINCIPLES)))
    ax.set_yticklabels([p.capitalize() for p in PRINCIPLES])
    for i in range(len(PRINCIPLES)):
        for j in range(len(BENCHMARKS)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=10)
    ax.set_title("Per-principle correlation of ELVIS with benchmarks (Pearson r)")
    fig.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def print_table(rows):
    print(f"{'Model':<16} {'Family':<12} {'ELVIS':>7} {'MMMU':>7} {'MMStar':>7} {'MMBench':>8} {'AI2D':>7}")
    for r in sorted(rows, key=lambda r: (r["family"], r["MMMU"])):
        print(f"{r['model']:<16} {r['family']:<12} {r['elvis_avg']:>7.2f}"
              f" {r['MMMU']:>7.1f} {r['MMStar']:>7.1f} {r['MMBench']:>8.1f} {r['AI2D']:>7.1f}")


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)
    baseline = load_all_baseline()
    rows = build_summary(baseline)
    print_table(rows)
    plot_correlations(rows, REPORTS_DIR / "correlation_scatter.png")
    plot_heatmap(rows, REPORTS_DIR / "correlation_heatmap.png")
