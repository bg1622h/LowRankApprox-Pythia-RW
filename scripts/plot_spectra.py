"""Create local PNG plots from singular-value diagnostics JSONL files."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def iter_records(directory):
    for path in sorted(Path(directory).glob("singular_values_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["_source_file"] = path.name
                    yield record


def _safe_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace("?", "unknown")
        .replace(" ", "_")
    )


def plot_param_spectrum(records, out_dir):
    records = sorted(records, key=lambda record: int(record["step"]))
    if not records:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), tight_layout=True)
    ax = axes[0]
    ax2 = axes[1]

    max_step = max(int(record["step"]) for record in records)
    for record in records:
        sv = np.asarray(record["singular_values"], dtype=float)
        if sv.size == 0:
            continue
        energy = sv ** 2
        cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-12)
        step = int(record["step"])
        alpha = 0.35 + 0.65 * (step / max(max_step, 1))
        ax.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            alpha=alpha,
            label=f"step {step}",
        )

        angles = record.get("principal_angles_degrees", [])
        if angles:
            ax2.plot(
                np.arange(1, len(angles) + 1),
                angles,
                marker="o",
                alpha=alpha,
                label=f"step {step}",
            )

    ax.set_title("cumulative energy")
    ax.set_xlabel("rank")
    ax.set_ylabel("energy explained")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)

    ax2.set_title("principal angles")
    ax2.set_xlabel("component")
    ax2.set_ylabel("degrees")
    ax2.set_ylim(0, 95)
    if ax2.lines:
        ax2.legend(fontsize=7)

    name = _safe_name(records[0]["param_name"])
    projector = _safe_name(records[0]["projector"])
    fig.suptitle(f"{projector}: {records[0]['param_name']}")
    out_path = Path(out_dir) / f"spectrum_{projector}_{name}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_group_summary(records, out_dir):
    by_group = defaultdict(list)
    for record in records:
        by_group[record.get("param_group", "unknown")].append(record)

    rows = []
    labels = []
    for group, group_records in sorted(by_group.items()):
        rank_energy = [
            float(record["rank_energy"])
            for record in group_records
            if record.get("rank_energy") is not None
        ]
        effective_rank = [
            float(record["effective_rank"])
            for record in group_records
            if record.get("effective_rank") is not None
        ]
        if not rank_energy or not effective_rank:
            continue
        labels.append(group)
        rows.append((np.mean(rank_energy), np.mean(effective_rank)))

    if not rows:
        return

    rank_energy, effective_rank = np.asarray(rows).T
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    axes[0].bar(x, rank_energy, color="steelblue")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("mean rank energy")
    axes[0].set_title("energy at selected rank")

    axes[1].bar(x, effective_rank, color="darkorange")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("mean effective rank")
    axes[1].set_title("spectral effective rank")

    fig.savefig(Path(out_dir) / "spectrum_group_summary.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spectrum_log_dir")
    parser.add_argument("--out-dir", default="runs/spectrum_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_records(args.spectrum_log_dir))

    by_param = defaultdict(list)
    for record in records:
        by_param[(record["projector"], record["param_name"])].append(record)

    for param_records in by_param.values():
        plot_param_spectrum(param_records, out_dir)
    plot_group_summary(records, out_dir)
    print(f"wrote {len(by_param)} parameter plots and group summary to {out_dir}")


if __name__ == "__main__":
    main()
