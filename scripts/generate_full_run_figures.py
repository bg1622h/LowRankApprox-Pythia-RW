"""Regenerate article/poster figures from full streaming-v2 run CSV logs.

Source data: streaming_v2_results/clean/*_1500clean_fp16.csv (batch=16, 1500 steps).
Output: figures/full_run/
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "streaming_v2_results" / "clean"
DEFAULT_OUT = ROOT / "figures" / "full_run"

DATA = DEFAULT_DATA
FULL_RUN_CSV: dict[str, Path] = {}


def _set_data_dir(data_dir: Path) -> None:
    global DATA, FULL_RUN_CSV
    DATA = data_dir
    FULL_RUN_CSV = {
        "galore2": DATA / "adammini_galore2_r8_1500clean_fp16.csv",
        "lotus": DATA / "adammini_lotus_r8_1500clean_fp16.csv",
        "adaptive_stochastic": DATA / "adammini_adaptive_stochastic_r8_1500clean_fp16.csv",
        "stochastic": DATA / "adammini_stochastic_r8_1500clean_fp16.csv",
        "stochastic_old": DATA / "adammini_stochastic_old_r8_1500clean_fp16.csv",
    }


_set_data_dir(DEFAULT_DATA)

STYLE = {
    "baseline": {"color": "#4472C4", "lw": 2.4},
    "baseline2": {"color": "#ED7D31", "lw": 2.4},
    "stochastic": {"color": "#70AD47", "lw": 2.6},
    "stochastic_old": {"color": "#548235", "lw": 2.2},
    "fisher": {"color": "#C00000", "lw": 2.4},
}

COMPARISON_RUNS = [
    ("GaLore2", "galore2", None, "baseline"),
    ("Lotus", "lotus", None, "baseline2"),
    ("Adaptive stoch.", "adaptive_stochastic", None, "stochastic"),
    ("Stochastic", "stochastic", None, "stochastic"),
    ("Stoch. (old)", "stochastic_old", None, "stochastic_old"),
]

BASELINE_RUNS = [
    ("GaLore2", "galore2", None, "baseline"),
    ("Lotus", "lotus", None, "baseline2"),
]


def _read_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, losses = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            loss = row.get("train_loss", "")
            if loss in ("", "nan", "NaN"):
                continue
            steps.append(int(row["step"]))
            losses.append(float(loss))
    return np.asarray(steps), np.asarray(losses)


def _rolling_median(values: np.ndarray, window: int = 51) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _read_rows(path: Path, proj: str | None) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if proj:
        rows = [row for row in rows if row.get("proj") == proj]
    if not rows:
        return []
    starts = [0]
    for index in range(1, len(rows)):
        if float(rows[index]["wall_time_sec"]) < float(rows[index - 1]["wall_time_sec"]):
            starts.append(index)
    return rows[starts[-1] :]


def _val_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    steps, vals = [], []
    for row in rows:
        raw = row.get("val_perplexity", "")
        if raw in ("", "nan", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            steps.append(int(row["step"]))
            vals.append(value)
    return np.asarray(steps), np.asarray(vals)


def _final_val(rows: list[dict]) -> float:
    _, vals = _val_curve(rows)
    return float(vals[-1]) if vals.size else float("nan")


def _plot_final_bars(
    labels: list[str],
    vals: list[float],
    colors: list[str],
    *,
    title: str = "Final validation — lower is better",
    xlabel: str = "Validation perplexity (last checkpoint)",
    label_decimals: int = 3,
    xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Horizontal bar chart with optional x-axis zoom for close values."""
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.65)
    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        lo, hi = min(vals), max(vals)
        pad = max(0.08, (hi - lo) * 0.35)
        ax.set_xlim(lo - pad, hi + pad)

    x0, x1 = ax.get_xlim()
    label_offset = (x1 - x0) * 0.012
    fmt = f"{{:.{label_decimals}f}}"
    for pos, value in zip(y, vals):
        ax.text(value + label_offset, pos, fmt.format(value), va="center", fontsize=9)
    fig.tight_layout()
    return fig


def _resolve_csv(key: str) -> Path:
    path = FULL_RUN_CSV[key]
    if not path.is_file():
        raise FileNotFoundError(f"Missing full-run CSV: {path}")
    return path


def plot_loss_curves(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    loss_runs = [
        ("GaLore2", _resolve_csv("galore2"), "#4472C4"),
        ("LOTUS", _resolve_csv("lotus"), "#ED7D31"),
    ]
    for label, path, color in loss_runs:
        steps, losses = _read_loss(path)
        ax.plot(steps, losses, color=color, alpha=0.35, linewidth=0.8)
        ax.plot(steps, _rolling_median(losses), color=color, linewidth=2.0, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("Full dataset · batch 16 · 1500 steps")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = out_dir / "article" / "experiment_loss_curves.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_vram_components(out_dir: Path) -> Path:
    rows = list(csv.DictReader(_resolve_csv("galore2").open(encoding="utf-8", newline="")))
    peak = max(rows, key=lambda r: float(r["vram_total_peak_gb"]))
    labels = ["Model", "Optimizer", "Gradients", "Activations"]
    values = [
        float(peak["vram_model_gb"]),
        float(peak["vram_optimizer_gb"]),
        float(peak["vram_gradients_gb"]),
        float(peak["vram_activations_gb"]),
    ]
    colors = ["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("VRAM (GB)")
    ax.set_title(f"Peak VRAM decomposition — GaLore2 (step {peak['step']})")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = out_dir / "article" / "experiment_vram_components.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_baselines(out_dir: Path) -> list[Path]:
    base_dir = out_dir / "baselines"
    base_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})

    fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)
    final_labels, final_vals, final_colors = [], [], []
    for label, key, proj, group in BASELINE_RUNS:
        rows = _read_rows(_resolve_csv(key), proj)
        steps, vals = _val_curve(rows)
        if steps.size == 0:
            continue
        style = STYLE[group]
        ax.plot(steps, vals, label=label, color=style["color"], linewidth=style["lw"])
        final_labels.append(label)
        final_vals.append(float(vals[-1]))
        final_colors.append(style["color"])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Baselines · full dataset · batch 16 · 1500 steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    curves = base_dir / "baseline_val_curves.png"
    fig.savefig(curves, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), tight_layout=True)
    x = np.arange(len(final_labels))
    bars = ax.bar(x, final_vals, color=final_colors, width=0.55)
    ax.set_xticks(x, final_labels, fontsize=11)
    ax.set_ylabel("Val perplexity (last checkpoint)")
    ax.set_title("Baseline final validation")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, final_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.2f}", ha="center", fontsize=10)
    bars_out = base_dir / "baseline_final_bars.png"
    fig.savefig(bars_out, dpi=300)
    plt.close(fig)
    return [curves, bars_out]


def plot_comparison(out_dir: Path) -> list[Path]:
    cmp_dir = out_dir / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})

    fig, ax = plt.subplots(figsize=(10, 4.2), tight_layout=True)
    for label, key, proj, group in COMPARISON_RUNS:
        rows = _read_rows(_resolve_csv(key), proj)
        steps, vals = _val_curve(rows)
        if steps.size == 0:
            continue
        style = STYLE[group]
        ax.plot(steps, vals, label=label, **{k: v for k, v in style.items() if k != "alpha"})

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("TinyLlama 1.1B · full RefinedWeb stream · rank 8 · 1500 steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    curves = cmp_dir / "all_methods_val_curves.png"
    fig.savefig(curves, dpi=300)
    poster_curves = ROOT / "figures" / "all_methods_val_curves.png"
    poster_curves.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(poster_curves, dpi=300)
    plt.close(fig)
    print(f"  also wrote {poster_curves.relative_to(ROOT)}")

    labels, vals, colors = [], [], []
    for label, key, proj, group in COMPARISON_RUNS:
        rows = _read_rows(_resolve_csv(key), proj)
        value = _final_val(rows)
        if math.isfinite(value):
            labels.append(label)
            vals.append(value)
            colors.append(STYLE[group]["color"])

    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    fig = _plot_final_bars(labels, vals, colors, xlim=(11.4, 12.0))
    bars = cmp_dir / "all_methods_final_bars.png"
    fig.savefig(bars, dpi=300, bbox_inches="tight")
    poster_bars = ROOT / "figures" / "all_methods_final_bars.png"
    poster_bars.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(poster_bars, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  also wrote {poster_bars.relative_to(ROOT)}")

    pick = [
        ("GaLore2", "galore2", None, "baseline"),
        ("Lotus", "lotus", None, "baseline2"),
        ("Adaptive stoch.", "adaptive_stochastic", None, "stochastic"),
    ]
    bl_labels, bl_vals, bl_colors = [], [], []
    for label, key, proj, group in pick:
        rows = _read_rows(_resolve_csv(key), proj)
        value = _final_val(rows)
        if math.isfinite(value):
            bl_labels.append(label)
            bl_vals.append(value)
            bl_colors.append(STYLE[group]["color"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), tight_layout=True)
    x = np.arange(len(bl_labels))
    bars_plot = ax.bar(x, bl_vals, color=bl_colors, width=0.62)
    ax.set_xticks(x, bl_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Val perplexity")
    ax.set_title("Baselines vs adaptive stochastic (full run)")
    ax.grid(axis="y", alpha=0.3)
    y0, y1 = ax.get_ylim()
    label_offset = (y1 - y0) * 0.02
    for bar, value in zip(bars_plot, bl_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    vs_stoch = cmp_dir / "comparison_baselines_vs_stochastic.png"
    fig.savefig(vs_stoch, dpi=300)
    poster_vs = ROOT / "figures" / "comparison_baselines_vs_stochastic.png"
    poster_vs.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(poster_vs, dpi=300)
    plt.close(fig)
    print(f"  also wrote {poster_vs.relative_to(ROOT)}")

    return [curves, bars, vs_stoch]


def copy_source_csvs(out_dir: Path) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for key, src in FULL_RUN_CSV.items():
        if src.is_file():
            shutil.copy2(src, data_dir / src.name)


def write_readme(out_dir: Path, generated: list[Path]) -> Path:
    lines = [
        "# Full-run figures",
        "",
        "Generated from `streaming_v2_results/clean/*_1500clean_fp16.csv`",
        "(extracted from `lowrank_experiments_full.tar`).",
        "",
        "Settings: TinyLlama 1.1B, rank 8, batch 16, seq 2048, 1500 steps, fp16.",
        "",
        "Regenerate:",
        "```bash",
        "python scripts/generate_full_run_figures.py",
        "```",
        "",
        "## Generated figures",
        "",
    ]
    for path in sorted(generated):
        rel = path.relative_to(out_dir).as_posix()
        lines.append(f"- `{rel}`")

    lines.extend(
        [
            "",
            "## Source CSV logs",
            "",
        ]
    )
    for key, src in FULL_RUN_CSV.items():
        lines.append(f"- **{key}**: `{src.relative_to(ROOT).as_posix()}`")

    lines.extend(
        [
            "",
            "## Not regenerated (no full-run diagnostics in archive)",
            "",
            "The archive contains training CSV logs only. Fisher JSONL, spectrum JSONL,",
            "and NPZ diagnostics were not included. Existing pilot figures remain at:",
            "",
            "- `figures/fisher_overlap_boxplots.png` — from `runs/fisher_*_jsonl/` (pilot)",
            "- `figures/spectrum_energy_boxplots.png` — from `runs/spectrum_all_layers.csv` (pilot)",
            "- `figures/spectrum_group_threshold_rank_boxplots.png` — same pilot spectrum data",
            "- `comparison/comparison_baselines_vs_fisher.png` — Fisher runs not in archive",
            "- `runs/poster/fisher_boxplots/**` — pilot Fisher JSONL only",
            "",
        ]
    )
    readme = out_dir / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from full streaming-v2 runs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="Directory with *_1500clean_fp16.csv")
    args = parser.parse_args()
    _set_data_dir(args.data_dir)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.append(plot_loss_curves(out_dir))
    generated.append(plot_vram_components(out_dir))
    generated.extend(plot_baselines(out_dir))
    generated.extend(plot_comparison(out_dir))
    copy_source_csvs(out_dir)
    write_readme(out_dir, generated)

    print(f"Wrote {len(generated)} figures to {out_dir}")
    for path in generated:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
