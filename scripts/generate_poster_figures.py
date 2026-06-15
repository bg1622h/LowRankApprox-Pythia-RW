"""Poster / social bundle: 2x2 summary panel + spectrum/fisher boxplots.

Runs generate_comparison_figures first, then regenerates boxplots in poster style.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_OUT = ROOT / "figures"

# Import plotting helpers from the main comparison script.
sys.path.insert(0, str(SCRIPTS))
from generate_comparison_figures import (  # noqa: E402
    ADAMMINI_SPEC,
    ALL_METHODS,
    FISHER_DIAG_SPEC,
    KEY_METHODS,
    KEY_VAL_METHODS,
    POSTER_RC,
    STOCHASTIC_TRIO,
    TARGET_STEP,
    _apply_poster_style,
    _collect_final_vals,
    _linear_extrapolate,
    _metric_curve,
    _needs_val_extension,
    _plot_val_series,
    _read_rows,
    _save_fig,
    _stochastic_cluster_xlim,
    STYLE,
)


def plot_key_metrics_2x2(out_dir: Path) -> Path:
    """Four-panel summary for poster / social posts."""
    _apply_poster_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    # (0,0) Key method val curves (low-rank only; AdamMini diverges)
    ax = axes[0, 0]
    key_specs = [s for s in KEY_VAL_METHODS if s.path.is_file()]
    for spec in key_specs:
        _plot_val_series(ax, spec, xmax=TARGET_STEP)
    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Val perplexity")
    ax.set_title("Key methods · validation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    # (0,1) Stochastic cluster zoom
    ax = axes[0, 1]
    for spec in STOCHASTIC_TRIO:
        _plot_val_series(ax, spec, xmax=TARGET_STEP)
    ax.set_xlim(400, TARGET_STEP)
    ax.set_ylim(11.45, 11.95)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Val perplexity")
    ax.set_title(f"Stochastic cluster · steps 400–{TARGET_STEP}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (1,0) Stochastic final bars (zoomed)
    ax = axes[1, 0]
    labels, vals, colors = _collect_final_vals(STOCHASTIC_TRIO)
    if labels:
        order = np.argsort(vals)
        labels = [labels[i] for i in order]
        vals = [vals[i] for i in order]
        colors = [colors[i] for i in order]
        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, height=0.65)
        ax.set_yticks(y, labels, fontsize=11)
        xlo, xhi = _stochastic_cluster_xlim(vals)
        ax.set_xlim(xlo, xhi)
        pad = (xhi - xlo) * 0.02
        for pos, value in zip(y, vals):
            ax.text(value + pad, pos, f"{value:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Val perplexity")
    ax.set_title(f"Stochastic cluster @ step {TARGET_STEP}")
    ax.grid(axis="x", alpha=0.3)

    # (1,1) Train loss (subset)
    ax = axes[1, 1]
    loss_specs = [
        ADAMMINI_SPEC,
        FISHER_DIAG_SPEC,
        ALL_METHODS[0],
        ALL_METHODS[4],
        ALL_METHODS[2],
    ]
    for spec in loss_specs:
        if not spec.path.is_file():
            continue
        rows = _read_rows(spec.path, spec.proj)
        steps, vals = _metric_curve(rows, "train_loss", max_step=TARGET_STEP)
        if steps.size == 0:
            continue
        style = STYLE[spec.style_key]
        ax.plot(steps, vals, label=spec.label, color=style["color"], linewidth=style["lw"])
        if spec.extend_val and _needs_val_extension(spec) and steps[-1] < TARGET_STEP:
            ext_steps, ext_vals = _linear_extrapolate(steps, vals, TARGET_STEP)
            mask = ext_steps > steps[-1]
            ax.plot(ext_steps[mask], ext_vals[mask], color=style["color"], linewidth=style["lw"], linestyle="--", alpha=0.85)
    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss · selected methods")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"TinyLlama 1.1B · RefinedWeb · rank 8 · {TARGET_STEP} steps",
        fontsize=16,
        y=0.98,
    )
    return _save_fig(fig, out_dir / "poster_key_metrics_2x2.png", dpi=250)


def _run_subprocess(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def regenerate_boxplots(out_dir: Path) -> None:
    spectrum_summary = ROOT / "runs" / "spectrum_all_layers_summary.csv"
    fisher_diag = ROOT / "diag" / "spectrum_parts" / "fisher"
    py = sys.executable

    if spectrum_summary.is_file():
        _run_subprocess(
            [
                py,
                str(SCRIPTS / "plot_spectra.py"),
                "--from-summary-csv",
                str(spectrum_summary),
                "--out-dir",
                str(ROOT / "runs" / "spectrum_all_layers_plots"),
                "--poster",
                "--extra-out",
                str(out_dir),
            ]
        )
    else:
        print(f"  skip spectrum boxplots: missing {spectrum_summary}")

    if fisher_diag.is_dir():
        _run_subprocess(
            [
                py,
                str(SCRIPTS / "plot_fisher.py"),
                str(fisher_diag),
                "--out-dir",
                str(ROOT / "runs" / "fisher_boxplots_diag"),
                "--poster",
                "--extra-out",
                str(out_dir),
            ]
        )
    else:
        print(f"  skip fisher boxplots: missing {fisher_diag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poster-ready figure bundle")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-comparison", action="store_true", help="Skip generate_comparison_figures.py")
    parser.add_argument("--skip-boxplots", action="store_true", help="Skip spectrum/fisher boxplot regeneration")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_comparison:
        _run_subprocess([sys.executable, str(SCRIPTS / "generate_comparison_figures.py"), "--out-dir", str(out_dir)])

    panel = plot_key_metrics_2x2(out_dir)
    print(f"  {panel.relative_to(ROOT)}")

    if not args.skip_boxplots:
        regenerate_boxplots(out_dir)

    print(f"Poster figures written to {out_dir}")


if __name__ == "__main__":
    main()
