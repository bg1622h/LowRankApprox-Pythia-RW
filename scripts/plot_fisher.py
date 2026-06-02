"""Grouped boxplots from Fisher diagnostics JSONL (fisher_*.jsonl)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def iter_fisher_records(directory: Path):
    for path in sorted(directory.glob("fisher_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _metric_value(record, metric: str):
    fisher = np.asarray(record.get("fisher", []), dtype=float)
    svd = np.asarray(record.get("svd_energy", []), dtype=float)
    rank = int(record.get("rank", 0))
    r = min(rank, fisher.size, svd.size) if fisher.size and svd.size else 0
    selected = [int(i) for i in record.get("selected_idx", [])]

    if metric == "fisher_effective_rank":
        return float(record["fisher_effective_rank"])
    if metric == "fisher_mean":
        return float(record["fisher_mean"])
    if metric == "fisher_max":
        return float(record["fisher_max"])
    if metric == "fisher_top_r_overlap":
        if r <= 0:
            return None
        top_fisher = set(np.argsort(fisher)[::-1][:r].tolist())
        return len(set(selected) & top_fisher) / r
    if metric == "svd_first_r_overlap":
        if r <= 0:
            return None
        return len(set(selected) & set(range(r))) / r
    if metric == "fisher_mass_in_selected":
        total = float(fisher.sum())
        if total <= 0 or not selected:
            return None
        return float(fisher[selected].sum() / total)
    if metric == "svd_mass_in_selected":
        total = float(svd.sum())
        if total <= 0 or not selected:
            return None
        return float(svd[selected].sum() / total)
    if metric == "svd_mass_first_r":
        total = float(svd.sum())
        if total <= 0 or r <= 0:
            return None
        return float(svd[:r].sum() / total)
    if metric == "assumption_gap":
        # Fisher mass in GaLore's first-r vs in actually selected directions.
        a = _metric_value(record, "svd_mass_first_r")
        b = _metric_value(record, "svd_mass_in_selected")
        if a is None or b is None:
            return None
        return b - a
    raise KeyError(metric)


def _draw_group_boxplot(ax, values, labels, title, ylim=None):
    positions = np.arange(1, len(values) + 1)
    ax.boxplot(values, positions=positions, showfliers=True)
    rng = np.random.default_rng(0)
    for pos, group_values in zip(positions, values):
        jitter = rng.normal(0, 0.035, size=len(group_values))
        ax.scatter(
            pos + jitter,
            group_values,
            s=14,
            alpha=0.35,
            color="tab:blue",
            edgecolors="none",
        )
        ax.text(
            pos,
            max(group_values),
            f"n={len(group_values)}",
            ha="center",
            va="bottom",
            fontsize=7,
            alpha=0.75,
        )
    ax.set_title(title)
    ax.set_xticks(positions, labels, rotation=30, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)


def _projector_label(name: str) -> str:
    return name.replace("_galore", "").replace("_", " ")


def plot_group_summary(records, out_dir: Path, projector_filter: str | None = None):
    if projector_filter:
        records = [r for r in records if r.get("projector") == projector_filter]

    by_group_projector = defaultdict(list)
    for record in records:
        key = (record.get("param_group", "unknown"), record.get("projector", "unknown"))
        by_group_projector[key].append(record)

    projectors = sorted({k[1] for k in by_group_projector})
    groups = sorted({k[0] for k in by_group_projector})

    plot_specs = [
        (
            "fisher_group_effective_rank_boxplots.png",
            "Fisher effective rank (per update)",
            [("fisher_effective_rank", "Fisher effective rank", None)],
        ),
        (
            "fisher_group_overlap_boxplots.png",
            "GaLore vs Fisher selection overlap",
            [
                ("svd_first_r_overlap", "overlap with first r (SVD / GaLore)", (0, 1.02)),
                ("fisher_top_r_overlap", "overlap with top r (Fisher)", (0, 1.02)),
            ],
        ),
        (
            "fisher_group_mass_boxplots.png",
            "Normalized mass captured by selected subspace",
            [
                ("fisher_mass_in_selected", "Fisher mass @ selected", (0, 1.02)),
                ("svd_mass_in_selected", "SVD energy @ selected", (0, 1.02)),
                ("svd_mass_first_r", "SVD energy @ first r", (0, 1.02)),
            ],
        ),
    ]

    written = 0
    for filename, figure_title, metric_specs in plot_specs:
        fig, axes = plt.subplots(
            1,
            len(metric_specs),
            figsize=(5.5 * len(metric_specs), 4.8),
            tight_layout=True,
        )
        axes = np.atleast_1d(axes)
        plotted = 0

        for ax, (metric, title, ylim) in zip(axes, metric_specs):
            values = []
            labels = []
            for group in groups:
                for projector in projectors:
                    key = (group, projector)
                    group_values = []
                    for record in by_group_projector.get(key, []):
                        try:
                            value = _metric_value(record, metric)
                        except (TypeError, ValueError, KeyError):
                            continue
                        if value is not None:
                            group_values.append(value)
                    if group_values:
                        labels.append(f"{group}\n{_projector_label(projector)}")
                        values.append(group_values)

            if not values:
                ax.axis("off")
                continue

            _draw_group_boxplot(ax, values, labels, title, ylim)
            plotted += 1

        if plotted:
            suffix = f" ({_projector_label(projector_filter)})" if projector_filter else ""
            fig.suptitle(figure_title + suffix, fontsize=13)
            fig.savefig(out_dir / filename, dpi=160)
            written += 1
        plt.close(fig)

    return written


def main():
    parser = argparse.ArgumentParser(description="Fisher diagnostics boxplots from JSONL")
    parser.add_argument(
        "fisher_log_dirs",
        nargs="+",
        help="One or more directories containing fisher_*.jsonl",
    )
    parser.add_argument("--out-dir", default="runs/fisher_boxplots")
    parser.add_argument(
        "--per-projector",
        action="store_true",
        help="Also write separate PNG sets per projector name",
    )
    args = parser.parse_args()

    records = []
    for directory in args.fisher_log_dirs:
        path = Path(directory)
        if not path.is_dir():
            print(f"skip missing dir: {path}")
            continue
        batch = list(iter_fisher_records(path))
        print(f"{path}: {len(batch)} records from {len(list(path.glob('fisher_*.jsonl')))} files")
        records.extend(batch)

    if not records:
        raise SystemExit("No fisher records found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projectors = sorted({r.get("projector", "unknown") for r in records})
    print(f"projectors: {projectors}")

    n = plot_group_summary(records, out_dir)
    print(f"wrote {n} combined figure(s) to {out_dir}")

    if args.per_projector:
        for projector in projectors:
            sub = out_dir / projector
            sub.mkdir(parents=True, exist_ok=True)
            n_sub = plot_group_summary(records, sub, projector_filter=projector)
            print(f"  {projector}: {n_sub} figure(s) in {sub}")


if __name__ == "__main__":
    main()
