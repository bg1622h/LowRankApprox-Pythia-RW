"""Generate method-comparison figures from streaming-v2 clean logs and diag archive.

Sources:
  - streaming_v2_results/clean/*_1500clean_fp16.csv — full 1500-step runs
  - diag/*.csv — diag methods from diag_final_20260615_1918.tar.gz

StochasticLotus: full 1500-step diag CSV (val checkpoints through 1450); no extrapolation.
Incomplete diag runs (e.g. diag_lotus) may still extrapolate training metrics only.

Stochastic zoom:
  Adaptive / stochastic / stochastic_old curves are zoomed to the comparable region
  before the step-300 validation spike (xmax = last common checkpoint before the
  first spread > 0.5 among the three curves).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from method_colors import METHOD_COLORS, LINEWIDTH

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "streaming_v2_results" / "clean"
DIAG = ROOT / "diag"
RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "figures"
TARGET_STEP = 1300
FISHER_CSV = RUNS / "refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv"
FISHER_PROJ = "softmax_fisher_projector"
ADAMMINI_CSV = DIAG / "adammini.csv"
FISHER_DIAG_CSV = DIAG / "fisher.csv"
DIVERGENCE_SPREAD = 0.5
DIVERGENCE_MIN_STEP = 200

POSTER_RC: dict[str, float | int] = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}

STYLE = {key: {"color": METHOD_COLORS[key], "lw": LINEWIDTH.get(key, 2.4)} for key in METHOD_COLORS}
STYLE["galore"] = {"color": METHOD_COLORS["galore2"], "lw": LINEWIDTH["galore2"]}


def _style(key: str) -> dict[str, float | str]:
    return STYLE[key]


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    proj: str | None
    style_key: str
    # If True, linearly extrapolate val curve to TARGET_STEP.
    extend_val: bool = False


# Full comparison at TARGET_STEP (1300)
ALL_METHODS: list[RunSpec] = [
    RunSpec("GaLore2", CLEAN / "adammini_galore2_r8_1500clean_fp16.csv", None, "galore2"),
    RunSpec("Lotus", CLEAN / "adammini_lotus_r8_1500clean_fp16.csv", None, "lotus"),
    RunSpec("StochasticLotus", DIAG / "StochasticLotus_1500_seq2048_b16.csv", "StochasticLotus", "stoch_lotus"),
    RunSpec("Stoch. GaLore2", DIAG / "stochastic_galore2_1500_seq2048_b16.csv", "stochastic_galore2", "stoch_galore2"),
    RunSpec("Adaptive stoch.", CLEAN / "adammini_adaptive_stochastic_r8_1500clean_fp16.csv", None, "adaptive_stochastic"),
    RunSpec("Stochastic", CLEAN / "adammini_stochastic_r8_1500clean_fp16.csv", None, "stochastic"),
    RunSpec("Stoch. (old)", CLEAN / "adammini_stochastic_old_r8_1500clean_fp16.csv", None, "stochastic_old"),
    RunSpec("Adam8bit", CLEAN / "adam8bit_none_r8_1500clean_fp16.csv", None, "adam8bit"),
]

BASELINE_METHODS = [r for r in ALL_METHODS if r.label in ("GaLore2", "Lotus")]

STOCHASTIC_TRIO = [r for r in ALL_METHODS if r.label in ("Adaptive stoch.", "Stochastic", "Stoch. (old)")]

ADAMMINI_SPEC = RunSpec("AdamMini", ADAMMINI_CSV, "none", "adammini")
FISHER_DIAG_SPEC = RunSpec("Softmax Fisher", FISHER_DIAG_CSV, FISHER_PROJ, "fisher")

PROJECTOR_METHODS: list[RunSpec] = [
    ALL_METHODS[0],  # GaLore2
    ALL_METHODS[1],  # Lotus
    ALL_METHODS[2],  # StochasticLotus
    ALL_METHODS[3],  # Stoch. GaLore2
]

KEY_METHODS: list[RunSpec] = [
    ADAMMINI_SPEC,
    ALL_METHODS[0],  # GaLore2
    ALL_METHODS[1],  # Lotus
    ALL_METHODS[4],  # Adaptive stoch.
    FISHER_DIAG_SPEC,
    ALL_METHODS[2],  # StochasticLotus
]

# Low-rank methods with well-behaved validation (excludes diverging AdamMini).
KEY_VAL_METHODS: list[RunSpec] = [s for s in KEY_METHODS if s is not ADAMMINI_SPEC]

# Training-metric plots: diag archive + clean streaming + adammini/fisher baselines
DIAG_TRAIN_RUNS: list[RunSpec] = [
    ADAMMINI_SPEC,
    FISHER_DIAG_SPEC,
    RunSpec("GaLore2", CLEAN / "adammini_galore2_r8_1500clean_fp16.csv", None, "galore2"),
    RunSpec("Lotus", CLEAN / "adammini_lotus_r8_1500clean_fp16.csv", None, "lotus"),
    RunSpec("Adaptive stoch.", CLEAN / "adammini_adaptive_stochastic_r8_1500clean_fp16.csv", None, "adaptive_stochastic"),
    RunSpec("Stochastic", CLEAN / "adammini_stochastic_r8_1500clean_fp16.csv", None, "stochastic"),
    RunSpec("StochasticLotus", DIAG / "StochasticLotus_1500_seq2048_b16.csv", "StochasticLotus", "stoch_lotus"),
    RunSpec("LoTus (diag)", DIAG / "diag_lotus_1500_seq2048_b16.csv", "lotus", "lotus", extend_val=True),
    RunSpec("GaLore2 (diag)", DIAG / "diag_galore2_1500_seq2048_b16.csv", "galore2", "galore2"),
    RunSpec("Stoch. GaLore2", DIAG / "stochastic_galore2_1500_seq2048_b16.csv", "stochastic_galore2", "stoch_galore2"),
]


def _read_rows(path: Path, proj: str | None) -> list[dict]:
    if not path.is_file():
        return []
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


def _val_curve(rows: list[dict], max_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    by_step: dict[int, float] = {}
    for row in rows:
        step = int(row["step"])
        if max_step is not None and step > max_step:
            continue
        raw = row.get("val_perplexity", "")
        if raw in ("", "nan", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            by_step[step] = value
    if not by_step:
        return np.array([]), np.array([])
    steps = np.array(sorted(by_step))
    return steps, np.array([by_step[s] for s in steps])


def _metric_curve(rows: list[dict], field: str, max_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    steps, vals = [], []
    for row in rows:
        step = int(row["step"])
        if max_step is not None and step > max_step:
            continue
        raw = row.get(field, "")
        if raw in ("", "nan", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            steps.append(step)
            vals.append(value)
    return np.asarray(steps), np.asarray(vals)


def _linear_extrapolate(steps: np.ndarray, vals: np.ndarray, target_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Extend (steps, vals) to target_step using linear fit on the last three points."""
    if steps.size == 0:
        return steps, vals
    if steps[-1] >= target_step:
        mask = steps <= target_step
        return steps[mask], vals[mask]

    tail_n = min(3, steps.size)
    coef = np.polyfit(steps[-tail_n:], vals[-tail_n:], 1)
    last_measured = int(steps[-1])
    # Validation checkpoints are every 50 steps in these runs.
    extra_steps = np.arange(last_measured + 50, target_step + 1, 50)
    if extra_steps.size == 0:
        extra_steps = np.array([target_step])
    extra_vals = np.polyval(coef, extra_steps)
    # Do not extrapolate below the last measured value (conservative for perplexity).
    extra_vals = np.maximum(extra_vals, vals[-1])

    out_steps = np.concatenate([steps, extra_steps])
    out_vals = np.concatenate([vals, extra_vals])
    return out_steps, out_vals


def _needs_val_extension(spec: RunSpec) -> bool:
    if not spec.extend_val:
        return False
    rows = _read_rows(spec.path, spec.proj)
    steps, _ = _val_curve(rows, max_step=TARGET_STEP)
    return steps.size > 0 and steps[-1] < TARGET_STEP


def _prepare_val_curve(spec: RunSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (meas_steps, meas_vals, plot_steps, plot_vals) with optional extension."""
    rows = _read_rows(spec.path, spec.proj)
    steps, vals = _val_curve(rows, max_step=TARGET_STEP)
    if steps.size == 0:
        return steps, vals, steps, vals

    if _needs_val_extension(spec):
        ext_steps, ext_vals = _linear_extrapolate(steps, vals, TARGET_STEP)
        return steps, vals, ext_steps, ext_vals

    return steps, vals, steps, vals


def _val_at_step(steps: np.ndarray, vals: np.ndarray, step: int) -> float:
    if steps.size == 0:
        return float("nan")
    if step in steps:
        return float(vals[np.where(steps == step)[0][0]])
    if step < steps[0] or step > steps[-1]:
        return float("nan")
    return float(np.interp(step, steps, vals))


def _plot_final_bars(
    labels: list[str],
    vals: list[float],
    colors: list[str],
    *,
    title: str = "Final validation — lower is better",
    xlabel: str = "Validation perplexity",
    label_decimals: int = 3,
    xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Horizontal bar chart with optional x-axis zoom for close values."""
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.65)
    ax.set_yticks(y, labels, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
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
        ax.text(value + label_offset, pos, fmt.format(value), va="center", fontsize=10)
    fig.tight_layout()
    return fig


def _bar_xlim(vals: list[float]) -> tuple[float, float] | None:
    """Use a tight x-axis when all perplexities sit in the main comparison band."""
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if lo >= 11.35 and hi <= 12.05:
        return (11.4, 12.0)
    return None


def _find_stochastic_zoom_end(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> int:
    common = None
    for steps, _ in curves.values():
        common = set(steps) if common is None else common & set(steps)
    if not common:
        return 250
    ordered = sorted(s for s in common if s >= DIVERGENCE_MIN_STEP)
    for step in ordered:
        spread_vals = []
        for steps, vals in curves.values():
            spread_vals.append(float(vals[np.where(steps == step)[0][0]]))
        if max(spread_vals) - min(spread_vals) > DIVERGENCE_SPREAD:
            idx = ordered.index(step)
            return ordered[idx - 1] if idx > 0 else step
    return ordered[-1] if ordered else 250


def _apply_poster_style() -> None:
    plt.rcParams.update(POSTER_RC)


def _legend_below(ax: plt.Axes, ncol: int = 2, fontsize: int = 10) -> None:
    ax.legend(
        fontsize=fontsize,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
    )


def _legend_upper_outside(ax: plt.Axes, ncol: int = 1, fontsize: int = 10) -> None:
    ax.legend(
        fontsize=fontsize,
        ncol=ncol,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )


def _annotate_curve_endpoints(
    ax: plt.Axes,
    endpoints: list[tuple[float, float, str, str]],
    *,
    x_pad_frac: float = 0.01,
) -> None:
    """Place endpoint labels with adjustText to avoid overlap."""
    if not endpoints:
        return
    texts = []
    xlim = ax.get_xlim()
    x_off = (xlim[1] - xlim[0]) * x_pad_frac
    for x, y, color, label in endpoints:
        texts.append(
            ax.text(
                x + x_off,
                y,
                label,
                color=color,
                fontsize=10,
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
            )
        )
    adjust_text(
        texts,
        ax=ax,
        only_move={"text": "y"},
        autoalign="y",
        force_text=(0.3, 0.6),
        force_points=(0.2, 0.4),
    )


def _stochastic_cluster_xlim(vals: list[float]) -> tuple[float, float]:
    lo, hi = min(vals), max(vals)
    pad = max(0.002, (hi - lo) * 0.45)
    return lo - pad, hi + pad


def _collect_final_vals(specs: list[RunSpec]) -> tuple[list[str], list[float], list[str]]:
    labels, vals, colors = [], [], []
    for spec in specs:
        if not spec.path.is_file():
            continue
        _, _, plot_steps, plot_vals = _prepare_val_curve(spec)
        value = _val_at_step(plot_steps, plot_vals, TARGET_STEP)
        if math.isfinite(value):
            labels.append(spec.label)
            vals.append(value)
            colors.append(STYLE[spec.style_key]["color"])
    return labels, vals, colors


def _add_zoom_inset(
    ax: plt.Axes,
    specs: list[RunSpec],
    *,
    xlim: tuple[int, int],
    ylim: tuple[float, float],
    title: str,
    loc: str,
    width: str = "38%",
    height: str = "42%",
) -> None:
    inset = inset_axes(ax, width=width, height=height, loc=loc, borderpad=1.2)
    for spec in specs:
        if not spec.path.is_file():
            continue
        _plot_val_series(inset, spec, xmax=xlim[1])
    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.set_title(title, fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.25)


def _save_fig(fig: plt.Figure, path: Path, *, dpi: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_val_series(
    ax: plt.Axes,
    spec: RunSpec,
    *,
    xmax: int | None = None,
) -> None:
    meas_steps, meas_vals, plot_steps, plot_vals = _prepare_val_curve(spec)
    if meas_steps.size == 0:
        return
    style = STYLE[spec.style_key]
    color = style["color"]
    lw = style["lw"]

    if xmax is not None:
        mask = plot_steps <= xmax
        plot_steps, plot_vals = plot_steps[mask], plot_vals[mask]
        meas_mask = meas_steps <= xmax
        meas_steps, meas_vals = meas_steps[meas_mask], meas_vals[meas_mask]

    ax.plot(meas_steps, meas_vals, label=spec.label, color=color, linewidth=lw)

    if spec.extend_val and _needs_val_extension(spec) and plot_steps.size > meas_steps.size:
        ext_mask = plot_steps > meas_steps[-1]
        if xmax is not None:
            ext_mask &= plot_steps <= xmax
        if np.any(ext_mask):
            ax.plot(
                plot_steps[ext_mask],
                plot_vals[ext_mask],
                color=color,
                linewidth=lw,
                linestyle="--",
                alpha=0.85,
            )


def plot_all_methods_curves(out_dir: Path) -> Path:
    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for spec in ALL_METHODS:
        if not spec.path.is_file():
            print(f"  skip missing: {spec.path}")
            continue
        _plot_val_series(ax, spec)

    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(f"All methods · TinyLlama 1.1B · rank 8 · {TARGET_STEP} steps")
    ax.grid(alpha=0.3)
    _legend_below(ax, ncol=4, fontsize=9)
    fig.subplots_adjust(bottom=0.22)
    return _save_fig(fig, out_dir / "all_methods_val_curves.png")


def plot_all_methods_curves_with_insets(out_dir: Path) -> Path:
    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for spec in ALL_METHODS:
        if spec.path.is_file():
            _plot_val_series(ax, spec)

    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(f"All methods · full range + zoom panels · step {TARGET_STEP}")
    ax.grid(alpha=0.3)
    _legend_below(ax, ncol=4, fontsize=9)
    fig.subplots_adjust(bottom=0.22)

    _add_zoom_inset(
        ax,
        STOCHASTIC_TRIO,
        xlim=(400, TARGET_STEP),
        ylim=(11.45, 11.95),
        title="Stochastic cluster",
        loc="lower left",
        width="36%",
        height="40%",
    )
    _add_zoom_inset(
        ax,
        PROJECTOR_METHODS,
        xlim=(0, TARGET_STEP),
        ylim=(11.5, 12.5),
        title="Projector methods",
        loc="upper right",
        width="34%",
        height="36%",
    )
    return _save_fig(fig, out_dir / "all_methods_val_curves_insets.png")


def plot_all_methods_bars(out_dir: Path) -> Path:
    labels, vals, colors = [], [], []
    for spec in ALL_METHODS:
        if not spec.path.is_file():
            continue
        _, _, plot_steps, plot_vals = _prepare_val_curve(spec)
        value = _val_at_step(plot_steps, plot_vals, TARGET_STEP)
        if not math.isfinite(value):
            continue
        labels.append(spec.label)
        vals.append(value)
        colors.append(STYLE[spec.style_key]["color"])

    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    fig = _plot_final_bars(
        labels,
        vals,
        colors,
        title="Final validation — lower is better",
        xlabel=f"Validation perplexity @ step {TARGET_STEP}",
        label_decimals=3,
        xlim=_bar_xlim(vals),
    )
    return _save_fig(fig, out_dir / "all_methods_final_bars.png")


def plot_stochastic_final_bars_zoom(out_dir: Path) -> Path | None:
    labels, vals, colors = _collect_final_vals(STOCHASTIC_TRIO)
    if not labels:
        return None
    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]
    fig = _plot_final_bars(
        labels,
        vals,
        colors,
        title=f"Stochastic cluster @ step {TARGET_STEP} (zoomed)",
        xlabel="Validation perplexity",
        label_decimals=3,
        xlim=_stochastic_cluster_xlim(vals),
    )
    return _save_fig(fig, out_dir / "all_methods_final_bars_stoch_zoom.png")


def plot_baseline_curves(out_dir: Path) -> Path:
    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for spec in BASELINE_METHODS:
        _plot_val_series(ax, spec, xmax=TARGET_STEP)
    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(f"Baselines · GaLore2 & Lotus · {TARGET_STEP} steps")
    ax.grid(alpha=0.3)
    _legend_upper_outside(ax, fontsize=11)
    fig.subplots_adjust(right=0.82)
    return _save_fig(fig, out_dir / "baseline_val_curves.png")


def plot_stochastic_zoom(out_dir: Path) -> Path:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in STOCHASTIC_TRIO:
        rows = _read_rows(spec.path, spec.proj)
        steps, vals = _val_curve(rows)
        if steps.size:
            curves[spec.label] = (steps, vals)

    zoom_end = _find_stochastic_zoom_end(curves)
    print(f"  stochastic zoom xmax={zoom_end} (before spread>{DIVERGENCE_SPREAD})")

    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    endpoints: list[tuple[float, float, str, str]] = []
    for spec in STOCHASTIC_TRIO:
        _plot_val_series(ax, spec, xmax=zoom_end)
        rows = _read_rows(spec.path, spec.proj)
        steps, vals = _val_curve(rows, max_step=zoom_end)
        if steps.size == 0:
            continue
        x, y = int(steps[-1]), float(vals[-1])
        color = STYLE[spec.style_key]["color"]
        endpoints.append((x, y, color, f"{y:.3f}"))

    ax.set_xlim(0, zoom_end)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(f"Stochastic variants · comparable region (steps ≤ {zoom_end})")
    ax.grid(alpha=0.3)
    _legend_upper_outside(ax, fontsize=11)

    yvals = []
    for spec in STOCHASTIC_TRIO:
        rows = _read_rows(spec.path, spec.proj)
        steps, vals = _val_curve(rows, max_step=zoom_end)
        if vals.size:
            yvals.extend(vals.tolist())
    if yvals:
        lo, hi = min(yvals), max(yvals)
        pad = max(0.08, (hi - lo) * 0.25)
        ax.set_ylim(lo - pad, hi + pad)

    _annotate_curve_endpoints(ax, endpoints)
    fig.subplots_adjust(right=0.78)
    return _save_fig(fig, out_dir / "stochastic_methods_val_curves_zoom.png")


def _plot_baseline_pair_vs(
    out_dir: Path,
    *,
    stochastic_spec: RunSpec,
    filename: str,
    title: str,
) -> Path | None:
    if not stochastic_spec.path.is_file():
        print(f"  skip missing stochastic run: {stochastic_spec.path}")
        return None

    plt.rcParams.update(POSTER_RC)
    fig, ax = plt.subplots(figsize=(12, 7))
    for spec in BASELINE_METHODS + [stochastic_spec]:
        _plot_val_series(ax, spec, xmax=TARGET_STEP)

    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _legend_upper_outside(ax, fontsize=11)
    fig.subplots_adjust(right=0.78)
    return _save_fig(fig, out_dir / filename)


def plot_baselines_vs_stoch_galore2(out_dir: Path) -> Path | None:
    return _plot_baseline_pair_vs(
        out_dir,
        stochastic_spec=ALL_METHODS[3],
        filename="comparison_baselines_vs_stoch_galore2.png",
        title=f"GaLore2 & Lotus vs Stoch. GaLore2 · {TARGET_STEP} steps",
    )


def plot_baselines_vs_stoch_lotus(out_dir: Path) -> Path | None:
    return _plot_baseline_pair_vs(
        out_dir,
        stochastic_spec=ALL_METHODS[2],
        filename="comparison_baselines_vs_stoch_lotus.png",
        title=f"GaLore2 & Lotus vs StochasticLotus · {TARGET_STEP} steps"
        + (" (dashed = extrapolated)" if _needs_val_extension(ALL_METHODS[2]) else ""),
    )


def _plot_method_subset_curves(
    specs: list[RunSpec],
    out_path: Path,
    *,
    title: str,
) -> Path:
    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for spec in specs:
        if not spec.path.is_file():
            print(f"  skip missing: {spec.path}")
            continue
        _plot_val_series(ax, spec, xmax=TARGET_STEP)
    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _legend_upper_outside(ax, fontsize=11)
    fig.subplots_adjust(right=0.78)
    return _save_fig(fig, out_path)


def plot_stochastic_galore_vs_baselines(out_dir: Path) -> Path:
    specs = [ALL_METHODS[0], ALL_METHODS[1], ALL_METHODS[3]]  # GaLore2, Lotus, Stoch. GaLore2
    return _plot_method_subset_curves(
        specs,
        out_dir / "comparison_stochastic_galore_vs_baselines.png",
        title=f"Stoch. GaLore2 vs baselines · rank 8 · {TARGET_STEP} steps",
    )


def plot_stochastic_lotus_vs_baselines(out_dir: Path) -> Path:
    specs = [ALL_METHODS[0], ALL_METHODS[1], ALL_METHODS[2]]  # GaLore2, Lotus, StochasticLotus
    return _plot_method_subset_curves(
        specs,
        out_dir / "comparison_stochastic_lotus_vs_baselines.png",
        title=f"StochasticLotus vs baselines · rank 8 · {TARGET_STEP} steps"
        + (" (dashed = extrapolated)" if _needs_val_extension(ALL_METHODS[2]) else ""),
    )


def plot_fisher_vs_baselines_curves(out_dir: Path) -> Path | None:
    """Validation curves: Fisher (diag/fisher.csv) vs GaLore2 & Lotus."""
    specs = [ALL_METHODS[0], ALL_METHODS[1], FISHER_DIAG_SPEC]
    available = [s for s in specs if s.path.is_file()]
    if len(available) < 2:
        print("  skip fisher vs baselines curves: insufficient data")
        return None
    return _plot_method_subset_curves(
        available,
        out_dir / "fisher_vs_baselines_val_curves.png",
        title=f"Softmax Fisher vs baselines · step {TARGET_STEP}",
    )


def plot_adammini_fisher_key_methods(out_dir: Path) -> list[Path]:
    """Dedicated figure: AdamMini + Fisher + key projector/stochastic methods."""
    outputs: list[Path] = []
    val_specs = [s for s in KEY_VAL_METHODS if s.path.is_file()]
    if val_specs:
        _apply_poster_style()
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.subplots_adjust(wspace=0.22, right=0.88)

        ax_main = axes[0]
        for spec in val_specs:
            _plot_val_series(ax_main, spec, xmax=TARGET_STEP)
        ax_main.set_xlim(0, TARGET_STEP)
        ax_main.set_xlabel("Training step")
        ax_main.set_ylabel("Validation perplexity")
        ax_main.set_title("Low-rank projector methods")
        ax_main.grid(alpha=0.3)
        ax_main.legend(fontsize=9, loc="upper right")

        ax_adam = axes[1]
        if ADAMMINI_SPEC.path.is_file():
            _plot_val_series(ax_adam, ADAMMINI_SPEC, xmax=TARGET_STEP)
            ax_adam.set_yscale("log")
            ax_adam.set_xlim(0, TARGET_STEP)
            ax_adam.set_xlabel("Training step")
            ax_adam.set_ylabel("Validation perplexity (log)")
            ax_adam.set_title("AdamMini baseline (dense, diverges)")
            ax_adam.grid(alpha=0.3, which="both")
            ax_adam.legend(fontsize=9, loc="upper right")

        fig.suptitle(
            f"AdamMini · Fisher · key methods · step {TARGET_STEP}",
            fontsize=14,
            y=1.02,
        )
        outputs.append(_save_fig(fig, out_dir / "adammini_fisher_key_methods_curves.png"))

    bar_specs = val_specs  # exclude diverging AdamMini from bar comparison
    labels, vals, colors = _collect_final_vals(bar_specs)
    if labels:
        order = np.argsort(vals)
        labels = [labels[i] for i in order]
        vals = [vals[i] for i in order]
        colors = [colors[i] for i in order]
        fig = _plot_final_bars(
            labels,
            vals,
            colors,
            title=f"AdamMini · Fisher · key methods @ step {TARGET_STEP}",
            xlabel="Validation perplexity",
            label_decimals=3,
            xlim=_bar_xlim(vals),
        )
        outputs.append(_save_fig(fig, out_dir / "adammini_fisher_key_methods_bars.png"))
    return outputs


def plot_train_loss_comparison(out_dir: Path) -> Path:
    """Train loss from clean streaming + diag adammini/fisher."""
    train_specs = [
        ADAMMINI_SPEC,
        FISHER_DIAG_SPEC,
        ALL_METHODS[0],
        ALL_METHODS[1],
        ALL_METHODS[4],
        ALL_METHODS[5],
        ALL_METHODS[2],
        ALL_METHODS[3],
    ]
    out = out_dir / "train_loss_comparison.png"
    _plot_train_metric(
        [s for s in train_specs if s.path.is_file()],
        "train_loss",
        "Training loss",
        f"Training loss comparison · rank 8 · {TARGET_STEP} steps",
        out,
    )
    return out


def plot_baselines_vs_fisher(out_dir: Path) -> Path | None:
    if FISHER_DIAG_CSV.is_file():
        fisher_path = FISHER_DIAG_CSV
    elif FISHER_CSV.is_file():
        fisher_path = FISHER_CSV
    else:
        fisher_path = DIAG / "diag_softmax_fisher_cr64_1500_seq2048_b16.csv"
    if not fisher_path.is_file():
        print(f"  skip missing Fisher CSV: {fisher_path}")
        return None

    fisher_spec = RunSpec("Softmax Fisher", fisher_path, FISHER_PROJ, "fisher", extend_val=fisher_path.parent == DIAG)

    pick = [
        ALL_METHODS[0],  # GaLore2
        ALL_METHODS[1],  # Lotus
        ALL_METHODS[4],  # Adaptive stoch.
        fisher_spec,
    ]
    labels, vals, colors = [], [], []
    for spec in pick:
        if not spec.path.is_file():
            print(f"  skip missing: {spec.path}")
            continue
        _, _, plot_steps, plot_vals = _prepare_val_curve(spec)
        value = _val_at_step(plot_steps, plot_vals, TARGET_STEP)
        if math.isfinite(value):
            labels.append(spec.label)
            vals.append(value)
            colors.append(STYLE[spec.style_key]["color"])

    if not labels:
        return None

    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("Val perplexity", fontsize=12)
    ax.set_title(f"Baselines vs Softmax Fisher @ step {TARGET_STEP}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    y0, y1 = ax.get_ylim()
    label_offset = (y1 - y0) * 0.03
    ax.set_ylim(y0, y1 + label_offset * 3)
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )
    fig.subplots_adjust(bottom=0.22)
    return _save_fig(fig, out_dir / "comparison_baselines_vs_fisher.png")


def plot_baselines_vs_stochastic(out_dir: Path) -> Path:
    pick = [
        ALL_METHODS[0],  # GaLore2
        ALL_METHODS[1],  # Lotus
        ALL_METHODS[4],  # Adaptive stoch.
        ALL_METHODS[2],  # StochasticLotus
    ]
    labels, vals, colors = [], [], []
    for spec in pick:
        if not spec.path.is_file():
            continue
        _, _, plot_steps, plot_vals = _prepare_val_curve(spec)
        value = _val_at_step(plot_steps, plot_vals, TARGET_STEP)
        if math.isfinite(value):
            labels.append(spec.label)
            vals.append(value)
            colors.append(STYLE[spec.style_key]["color"])

    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("Val perplexity", fontsize=12)
    ax.set_title(f"Selected methods @ step {TARGET_STEP}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    y0, y1 = ax.get_ylim()
    label_offset = (y1 - y0) * 0.03
    ax.set_ylim(y0, y1 + label_offset * 3)
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )
    fig.subplots_adjust(bottom=0.22)
    return _save_fig(fig, out_dir / "comparison_baselines_vs_stochastic.png")


def _plot_train_metric(
    specs: list[RunSpec],
    field: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    _apply_poster_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for spec in specs:
        if not spec.path.is_file():
            continue
        rows = _read_rows(spec.path, spec.proj)
        steps, vals = _metric_curve(rows, field, max_step=TARGET_STEP)
        if steps.size == 0:
            continue
        style = STYLE[spec.style_key]
        ax.plot(steps, vals, label=spec.label, color=style["color"], linewidth=style["lw"])
        if spec.extend_val and steps[-1] < TARGET_STEP:
            ext_steps, ext_vals = _linear_extrapolate(steps, vals, TARGET_STEP)
            mask = ext_steps > steps[-1]
            ax.plot(ext_steps[mask], ext_vals[mask], color=style["color"], linewidth=style["lw"], linestyle="--", alpha=0.85)

    ax.set_xlim(0, TARGET_STEP)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _legend_below(ax, ncol=3, fontsize=9)
    fig.subplots_adjust(bottom=0.2)
    _save_fig(fig, out_path)


def plot_diag_training_curves(out_dir: Path) -> list[Path]:
    diag_out = out_dir / "diag"
    diag_out.mkdir(parents=True, exist_ok=True)
    outputs = []
    specs = [s for s in DIAG_TRAIN_RUNS if s.path.is_file()]
    for field, ylabel, fname, title_suffix in [
        ("train_loss", "Training loss", "diag_train_loss.png", "training loss"),
        ("train_ppl_capped", "Training perplexity (capped)", "diag_train_ppl.png", "training perplexity"),
        ("tokens_per_sec", "Tokens/sec", "diag_throughput.png", "throughput"),
        ("grad_norm", "Gradient norm", "diag_grad_norm.png", "gradient norm"),
    ]:
        out = diag_out / fname
        _plot_train_metric(
            specs,
            field,
            ylabel,
            f"Diag runs · {title_suffix} · rank 8 · {TARGET_STEP} steps",
            out,
        )
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison figures from clean + diag logs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    print(f"Target step: {TARGET_STEP}")
    generated.append(plot_all_methods_curves(out_dir))
    generated.append(plot_all_methods_curves_with_insets(out_dir))
    generated.append(plot_all_methods_bars(out_dir))
    stoch_bars = plot_stochastic_final_bars_zoom(out_dir)
    if stoch_bars is not None:
        generated.append(stoch_bars)
    generated.append(plot_baseline_curves(out_dir))
    generated.append(plot_stochastic_zoom(out_dir))
    generated.append(plot_stochastic_galore_vs_baselines(out_dir))
    generated.append(plot_stochastic_lotus_vs_baselines(out_dir))
    fisher_curves = plot_fisher_vs_baselines_curves(out_dir)
    if fisher_curves is not None:
        generated.append(fisher_curves)
    generated.extend(plot_adammini_fisher_key_methods(out_dir))
    generated.append(plot_train_loss_comparison(out_dir))
    for path in (plot_baselines_vs_stoch_galore2(out_dir), plot_baselines_vs_stoch_lotus(out_dir)):
        if path is not None:
            generated.append(path)
    generated.append(plot_baselines_vs_stochastic(out_dir))
    fisher_path = plot_baselines_vs_fisher(out_dir)
    if fisher_path is not None:
        generated.append(fisher_path)
    generated.extend(plot_diag_training_curves(out_dir))

    print(f"Wrote {len(generated)} figures to {out_dir}")
    for path in generated:
        try:
            print(f"  {path.relative_to(ROOT)}")
        except ValueError:
            print(f"  {path}")


if __name__ == "__main__":
    main()
