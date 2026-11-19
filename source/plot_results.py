"""Plot paired-test accuracy by component count from the experiment summary."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
CURVES = {
    0.5: ("#395F9D", "s", "-", "α = 0.50"),
    0.75: ("#007B66", "o", "-", "α = 0.75"),
    1.0: ("#292D31", "D", "--", "PCA (α = 1)"),
    1.25: ("#AA5825", "^", "-", "α = 1.25"),
    1.5: ("#8061A7", "v", "-", "α = 1.50"),
}


def plot(summary_path, figure_path):
    with summary_path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    by_configuration = {
        (float(record["alpha"]), int(record["components"])): record
        for record in records
    }
    component_counts = sorted({k for _, k in by_configuration})
    iteration_limited = {
        float(record["alpha"]) for record in records
        if record["classifier_converged"] == "False"
    }
    if not component_counts or len(records) != len(CURVES) * len(component_counts):
        raise ValueError("The summary must contain the complete experimental grid.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "#222222",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(figsize=(8.9, 3.45), ncols=2, sharey=True)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.29, top=0.88, wspace=0.20)

    panels = [
        ("clean_accuracy_pct", "Clean test images"),
        ("corrupted_mean_accuracy_pct", "Test images with pixel outliers"),
    ]
    for axis, (metric, title) in zip(axes, panels):
        for alpha, (color, marker, line, label) in CURVES.items():
            accuracy = [float(by_configuration[alpha, k][metric]) for k in component_counts]
            axis.plot(
                component_counts, accuracy, color=color, marker=marker, linestyle=line,
                markeredgecolor="white", markeredgewidth=0.55, markersize=5.4,
                linewidth=1.85, label=f"{label}*" if alpha in iteration_limited else label,
            )
        axis.set_title(title, loc="left", pad=9)
        axis.set_xticks(component_counts)
        axis.set_xlabel("Principal components")
        axis.grid(axis="y", color="#DCE2E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["left"].set_color("#B7C0C6")
        axis.spines["bottom"].set_color("#B7C0C6")
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
                  bbox_to_anchor=(0.52, 0.045), columnspacing=1.3, handletextpad=0.5)
    if iteration_limited:
        figure.text(0.52, 0.025, "* Classifier reached the iteration limit at one or more dimensions.",
                    ha="center", fontsize=8.5, color="#42484C")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=260, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=PROJECT / "results" / "summary.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "report" / "accuracy-by-components.png")
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
