"""Compact poster figures from pipeline CSV logs (no tables)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# label, csv path, optional proj filter, linestyle group
RUNS = [
    ("GaLore2", ROOT / "runs/refinedweb_full_galore2_1500_seq2048.csv", None, "baseline"),
    ("Lotus", ROOT / "runs/refinedweb_full_lotus_1500_seq2048.csv", None, "baseline"),
    (
        "Adaptive stoch.",
        ROOT / "runs/refinedweb_full_adaptive_stochastic_1500_seq2048.csv",
        None,
        "stochastic",
    ),
    ("Fisher", ROOT / "runs/refinedweb_full_fisher_both_1500_seq2048_cr128.csv", "fisher_projector", "fisher"),
    (
        "Block Fisher",
        ROOT / "runs/refinedweb_full_fisher_both_1500_seq2048_cr128.csv",
        "block_fisher_projector",
        "fisher",
    ),
    (
        "Top-k Fisher",
        ROOT / "runs/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv",
        "topk_fisher_projector",
        "fisher",
    ),
    (
        "Softmax Fisher",
        ROOT / "runs/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv",
        "softmax_fisher_projector",
        "fisher",
    ),
]

STYLE = {
    "baseline": {"color": "#4472C4", "lw": 2.4},
    "baseline2": {"color": "#ED7D31", "lw": 2.4},
    "stochastic": {"color": "#70AD47", "lw": 2.6},
    "fisher": {"color": "#C00000", "lw": 1.6, "alpha": 0.85},
}

BASELINES = [
    ("GaLore2", ROOT / "runs/refinedweb_full_galore2_1500_seq2048.csv", None, "baseline"),
    ("Lotus", ROOT / "runs/refinedweb_full_lotus_1500_seq2048.csv", None, "baseline2"),
]


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


def plot_curves(out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(10, 4.2), tight_layout=True)
    for label, path, proj, group in RUNS:
        if not path.exists():
            continue
        rows = _read_rows(path, proj)
        steps, vals = _val_curve(rows)
        if steps.size == 0:
            continue
        style = STYLE[group]
        ax.plot(steps, vals, label=label, **{k: v for k, v in style.items() if k != "alpha"})
        if "alpha" in style:
            ax.lines[-1].set_alpha(style["alpha"])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("TinyLlama 1.1B · RefinedWeb 0.5 GB · rank 8 · 1500 steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    fig.savefig(out_dir / "comparison" / "all_methods_val_curves.png", dpi=300)
    plt.close(fig)


def plot_final_bars(out_dir: Path) -> None:
    labels, vals, colors = [], [], []
    for label, path, proj, group in RUNS:
        if not path.exists():
            continue
        rows = _read_rows(path, proj)
        value = _final_val(rows)
        if not math.isfinite(value):
            continue
        labels.append(label)
        vals.append(value)
        colors.append(STYLE[group]["color"])

    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.65)
    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlabel("Validation perplexity (step 1480)")
    ax.set_title("Final validation — lower is better")
    ax.grid(axis="x", alpha=0.3)
    for pos, value in zip(y, vals):
        ax.text(value + 0.05, pos, f"{value:.2f}", va="center", fontsize=9)
    fig.savefig(out_dir / "comparison" / "all_methods_final_bars.png", dpi=300)
    plt.close(fig)


def plot_baselines_only(out_dir: Path) -> None:
    """GaLore2 + Lotus only — dedicated baseline poster figures."""
    base_dir = out_dir / "baselines"
    base_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})

    fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)
    final_labels, final_vals, final_colors = [], [], []
    for label, path, proj, group in BASELINES:
        rows = _read_rows(path, proj)
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
    ax.set_title("Baselines · TinyLlama 1.1B · RefinedWeb 0.5 GB · rank 8")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.savefig(base_dir / "baseline_val_curves.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), tight_layout=True)
    x = np.arange(len(final_labels))
    bars = ax.bar(x, final_vals, color=final_colors, width=0.55)
    ax.set_xticks(x, final_labels, fontsize=11)
    ax.set_ylabel("Val perplexity @ step 1480")
    ax.set_title("Baseline final validation")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, final_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.2f}", ha="center", fontsize=10)
    fig.savefig(base_dir / "baseline_final_bars.png", dpi=300)
    plt.close(fig)


def plot_baselines_vs_fisher(out_dir: Path) -> None:
    """Single slide: baselines + best stochastic vs Fisher family."""
    pick = [
        ("GaLore2", ROOT / "runs/refinedweb_full_galore2_1500_seq2048.csv", None, "baseline"),
        ("Lotus", ROOT / "runs/refinedweb_full_lotus_1500_seq2048.csv", None, "baseline"),
        (
            "Adaptive stoch.",
            ROOT / "runs/refinedweb_full_adaptive_stochastic_1500_seq2048.csv",
            None,
            "stochastic",
        ),
        ("Fisher (avg)", ROOT / "runs/refinedweb_full_fisher_both_1500_seq2048_cr128.csv", "fisher_projector", "fisher"),
    ]
    labels, vals, colors = [], [], []
    for label, path, proj, group in pick:
        rows = _read_rows(path, proj)
        value = _final_val(rows)
        if math.isfinite(value):
            labels.append(label)
            vals.append(value)
            colors.append(STYLE[group]["color"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), tight_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    ax.set_xticks(x, labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Val perplexity")
    ax.set_title("Baselines vs Fisher (poster slide)")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center", fontsize=9)
    fig.savefig(out_dir / "comparison" / "comparison_baselines_vs_fisher.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "poster"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "comparison").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_baselines_only(out_dir)
    plot_curves(out_dir)
    plot_final_bars(out_dir)
    plot_baselines_vs_fisher(out_dir)
    print(f"wrote poster PNGs to {out_dir}")
    print(f"  baselines/     — GaLore2 + Lotus")
    print(f"  comparison/    — all methods + vs Fisher slide")
    print(f"  fisher_boxplots/ — run scripts/build_poster_assets.py")


if __name__ == "__main__":
    main()
