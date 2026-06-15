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

from method_colors import GROUP_PALETTE


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


def _draw_group_boxplot(ax, values, labels, title, ylim=None, *, poster=False):
    positions = np.arange(1, len(values) + 1)
    box_width = 0.72 if poster else 0.6
    box_colors = [GROUP_PALETTE[i % len(GROUP_PALETTE)] for i in range(len(values))]
    bp = ax.boxplot(
        values,
        positions=positions,
        showfliers=True,
        widths=box_width,
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.8)
    rng = np.random.default_rng(0)
    point_size = 28 if poster else 14
    n_font = 11 if poster else 7
    for pos, group_values, color in zip(positions, values, box_colors):
        jitter = rng.normal(0, 0.035, size=len(group_values))
        ax.scatter(
            pos + jitter,
            group_values,
            s=point_size,
            alpha=0.35,
            color=color,
            edgecolors="white",
            linewidths=0.3,
        )
    title_size = 16 if poster else None
    ax.set_title(title, fontsize=title_size)
    tick_size = 13 if poster else None
    ax.set_xticks(positions, labels, rotation=25 if poster else 30, ha="right", fontsize=tick_size)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    y_top = ax.get_ylim()[1]
    y_pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    for pos, group_values in zip(positions, values):
        ax.text(
            pos,
            y_top + y_pad,
            f"n={len(group_values)}",
            ha="center",
            va="bottom",
            fontsize=n_font,
            alpha=0.75,
            clip_on=False,
        )
    ax.set_ylim(ax.get_ylim()[0], y_top + y_pad * 4)
    if poster:
        ax.tick_params(axis="y", labelsize=12)


def _projector_label(name: str) -> str:
    return name.replace("_galore", "").replace("_", " ")


def plot_group_summary(records, out_dir: Path, projector_filter: str | None = None, *, poster=False, extra_out: Path | None = None):
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
        n_panels = len(metric_specs)
        if poster:
            figsize = (20, 12) if n_panels >= 3 else (18, 10)
            dpi = 300
            suptitle_size = 20
            layout_pad = 2.5
        else:
            figsize = (5.5 * n_panels, 4.8)
            dpi = 160
            suptitle_size = 13
            layout_pad = 1.0

        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize,
        )
        fig.subplots_adjust(wspace=0.28 if poster else 0.2)
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

            _draw_group_boxplot(ax, values, labels, title, ylim, poster=poster)
            plotted += 1

        if plotted:
            suffix = f" ({_projector_label(projector_filter)})" if projector_filter else ""
            fig.suptitle(figure_title + suffix, fontsize=suptitle_size, y=0.98)
            fig.tight_layout(pad=layout_pad)
            out_path = out_dir / filename
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            if extra_out is not None:
                extra_out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(extra_out / filename, dpi=dpi, bbox_inches="tight")
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
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Larger figure size, fonts, and dpi for poster slides",
    )
    parser.add_argument(
        "--extra-out",
        type=Path,
        help="Also write figures to this directory (e.g. figures/)",
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

    n = plot_group_summary(records, out_dir, poster=args.poster, extra_out=args.extra_out)
    print(f"wrote {n} combined figure(s) to {out_dir}")

    if args.per_projector:
        for projector in projectors:
            sub = out_dir / projector
            sub.mkdir(parents=True, exist_ok=True)
            n_sub = plot_group_summary(
                records, sub, projector_filter=projector, poster=args.poster, extra_out=None
            )
            print(f"  {projector}: {n_sub} figure(s) in {sub}")


if __name__ == "__main__":
    main()
