"""Draw the README figures from the saved metrics.json files.

    python plot_results.py      # writes assets/ablation.png and assets/training_curves.png
"""

import json
import os

import matplotlib.pyplot as plt

from bart_generation import ROOT, RUNS_DIR

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ASSETS = os.path.join(ROOT, "assets")

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def load(run):
    with open(os.path.join(RUNS_DIR, run, "metrics.json"), encoding="utf-8") as file:
        return json.load(file)


def ablation():
    runs = ["natural_segments", "natural_language", "natural_copy", "sep_baseline",
            "natural_reverse", "natural_brio", "full"]
    data = [load(run)["dev_rerank"] for run in runs]
    penalized = [d["penalized_bleu"] for d in data]
    copy = [100 - d["input_bleu"] for d in data]   # metrics.json stores 100 - BLEU(output, input)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, values, title in ((left, penalized, "Penalized BLEU (graded metric, higher is better)"),
                              (right, copy, "BLEU vs. input (copying, lower is better)")):
        ax.barh(runs, values, height=0.55, color=BLUE)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        for y, value in enumerate(values):
            ax.text(value + 0.8, y, f"{value:.1f}", va="center", color=MUTED, fontsize=9)
        ax.set_xlim(0, max(values) * 1.15)
    fig.suptitle("Ablation on the ETPC dev split (546 pairs)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "ablation.png"), dpi=150)


def training_curves():
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for run, color, label in (("full", BLUE, "full (all improvements)"), ("natural_language", ORANGE, "baseline")):
        epochs = load(run)["epochs"]
        x = [e["epoch"] for e in epochs]
        for ax, y in ((left, [e["penalized_bleu"] for e in epochs]), (right, [100 - e["input_bleu"] for e in epochs])):
            ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", label=label)
            ax.plot(x[-1], y[-1], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for ax, title in ((left, "Penalized BLEU"), (right, "BLEU vs. input (copying)")):
        ax.axvline(2.5, color=MUTED, linewidth=1)
        ax.text(2.6, ax.get_ylim()[1], " BRIO on", va="top", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("epoch")
    left.legend(frameon=False, loc="lower right")
    fig.suptitle("BRIO ranking loss breaks the copy habit", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "training_curves.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    ablation()
    training_curves()
    print(f"Wrote figures to {ASSETS}")
