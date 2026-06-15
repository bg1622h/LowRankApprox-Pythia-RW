"""Create local PNG plots from singular-value diagnostics JSONL files."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from method_colors import GROUP_PALETTE


def iter_records(directory):
    for path in sorted(Path(directory).glob("singular_values_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["_source_file"] = path.name
                    yield record


def iter_records_from_summary_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = dict(row)
            for key in (
                "step",
                "rank",
                "rank_energy",
                "rank_90",
                "rank_95",
                "rank_99",
                "effective_rank",
                "mean_angle",
                "max_angle",
                "adaptive_rank",
                "temperature",
                "candidate_count",
            ):
                raw = record.get(key, "")
                if raw not in ("", None):
                    try:
                        record[key] = float(raw) if "." in str(raw) else int(raw)
                    except ValueError:
                        pass
            angles = []
            if record.get("mean_angle") not in ("", None):
                angles.append(float(record["mean_angle"]))
            if angles:
                record["principal_angles_degrees"] = angles
            yield record


def _safe_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace("?", "unknown")
        .replace(" ", "_")
    )


def _metric_value(record, metric):
    if metric in ("mean_angle", "max_angle"):
        angles = record.get("principal_angles_degrees", [])
        if not angles:
            return None
        values = np.asarray(angles, dtype=float)
        if metric == "mean_angle":
            return float(values.mean())
        return float(values.max())

    value = record.get(metric, "")
    if value in ("", None):
        return None
    return float(value)


def _draw_group_boxplot(ax, values, labels, title, ylim=None, *, poster=False):
    positions = np.arange(1, len(values) + 1)
    box_width = 0.82 if poster else 0.78
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
    point_size = 28 if poster else 16
    n_font = 11 if poster else 7
    for pos, group_values, color in zip(positions, values, box_colors):
        jitter = rng.normal(0, 0.035, size=len(group_values))
        ax.scatter(
            pos + jitter,
            group_values,
            s=point_size,
            alpha=0.45,
            color=color,
            edgecolors="white",
            linewidths=0.4,
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
            alpha=0.85,
            clip_on=False,
        )
    ax.set_ylim(ax.get_ylim()[0], y_top + y_pad * 4)
    if poster:
        ax.tick_params(axis="y", labelsize=12)


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


def plot_group_summary(records, out_dir, *, poster=False, only: str | None = None, extra_out: Path | None = None):
    by_group = defaultdict(list)
    for record in records:
        by_group[record.get("param_group", "unknown")].append(record)

    plot_specs = [
        (
            "spectrum_group_energy_boxplots.png",
            "Energy captured by selected rank",
            [("rank_energy", "rank energy @ selected rank", (0, 1.02))],
        ),
        (
            "spectrum_group_effective_rank_boxplots.png",
            "Effective spectral rank",
            [("effective_rank", "effective rank", None)],
        ),
        (
            "spectrum_group_threshold_rank_boxplots.png",
            "Ranks needed for cumulative energy thresholds",
            [
                ("rank_90", "rank for 90% energy", None),
                ("rank_95", "rank for 95% energy", None),
                ("rank_99", "rank for 99% energy", None),
            ],
        ),
        (
            "spectrum_group_angle_boxplots.png",
            "Principal angle stability between basis updates",
            [
                ("mean_angle", "mean principal angle", (0, 95)),
                ("max_angle", "max principal angle", (0, 95)),
            ],
        ),
        (
            "spectrum_group_adaptive_rank_boxplots.png",
            "Adaptive rank selected by stochastic projector",
            [("adaptive_rank", "adaptive rank", None)],
        ),
    ]
    labels = sorted(by_group)

    written = 0
    for filename, figure_title, metric_specs in plot_specs:
        if only and filename != only:
            continue

        n_panels = len(metric_specs)
        if poster:
            figsize = (20, 12) if n_panels >= 3 else (18, 10) if n_panels == 2 else (12, 8)
            dpi = 300
            suptitle_size = 20
            layout_pad = 2.5
        else:
            figsize = (6.2 * n_panels, 5.0)
            dpi = 200
            suptitle_size = 13
            layout_pad = 1.0

        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        fig.subplots_adjust(wspace=0.28 if poster else 0.2)
        axes = np.atleast_1d(axes)
        plotted = 0

        for ax, (metric, title, ylim) in zip(axes, metric_specs):
            values = []
            used_labels = []
            for group in labels:
                group_values = []
                for record in by_group[group]:
                    try:
                        value = _metric_value(record, metric)
                    except (TypeError, ValueError):
                        continue
                    if value is not None:
                        group_values.append(value)
                if group_values:
                    used_labels.append(group)
                    values.append(group_values)

            if not values:
                ax.axis("off")
                continue

            _draw_group_boxplot(ax, values, used_labels, title, ylim, poster=poster)
            if metric.startswith("rank") or metric == "adaptive_rank":
                low = int(np.floor(min(min(group_values) for group_values in values)))
                high = int(np.ceil(max(max(group_values) for group_values in values)))
                if poster and (high - low) > 12:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                else:
                    ax.set_yticks(np.arange(low, high + 1))
            plotted += 1

        if plotted:
            fig.suptitle(figure_title, fontsize=suptitle_size, y=0.98 if poster else 0.98)
            fig.tight_layout(pad=layout_pad)
            out_path = Path(out_dir) / filename
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            if extra_out is not None:
                extra_out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(extra_out / filename, dpi=dpi, bbox_inches="tight")
            written += 1
        plt.close(fig)

    if written:
        print(f"wrote {written} grouped boxplot figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spectrum_log_dir",
        nargs="?",
        help="Directory with singular_values_*.jsonl (omit when using --from-summary-csv)",
    )
    parser.add_argument("--out-dir", default="runs/spectrum_plots")
    parser.add_argument(
        "--from-summary-csv",
        type=Path,
        help="Read pre-summarized spectrum metrics instead of JSONL logs",
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Larger figure size, fonts, and dpi for poster slides",
    )
    parser.add_argument(
        "--only",
        help="Generate only this output filename (e.g. spectrum_group_threshold_rank_boxplots.png)",
    )
    parser.add_argument(
        "--extra-out",
        type=Path,
        help="Also write figures to this directory (e.g. figures/)",
    )
    parser.add_argument(
        "--with-param-plots",
        action="store_true",
        help="Also write one spectrum plot per projected parameter.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_summary_csv:
        records = list(iter_records_from_summary_csv(args.from_summary_csv))
    elif args.spectrum_log_dir:
        records = list(iter_records(args.spectrum_log_dir))
    else:
        parser.error("Provide spectrum_log_dir or --from-summary-csv")

    by_param = defaultdict(list)
    for record in records:
        by_param[(record["projector"], record["param_name"])].append(record)

    if args.with_param_plots:
        for param_records in by_param.values():
            plot_param_spectrum(param_records, out_dir)
    plot_group_summary(
        records,
        out_dir,
        poster=args.poster,
        only=args.only,
        extra_out=args.extra_out,
    )
    param_plot_count = len(by_param) if args.with_param_plots else 0
    print(f"wrote {param_plot_count} parameter plots and grouped boxplots to {out_dir}")


if __name__ == "__main__":
    main()
