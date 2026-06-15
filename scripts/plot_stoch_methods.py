"""Training curves for stochastic methods (StochasticLotus, LoTus, GaLore2, Stoch GaLore2)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diag"
OUT = DIAG / "plots"

# label, csv path, proj filter (optional), color, group
RUNS = [
    ("StochasticLotus", DIAG / "StochasticLotus_1500_seq2048_b16.csv", "StochasticLotus", "#E63946", "stochastic"),
    ("LoTus", DIAG / "diag_lotus_1500_seq2048_b16.csv", "lotus", "#457B9D", "baseline"),
    ("GaLore2", DIAG / "diag_galore2_1500_seq2048_b16.csv", "galore2", "#2A9D8F", "baseline"),
    ("Stoch GaLore2", DIAG / "stochastic_galore2_1500_seq2048_b16.csv", "stochastic_galore2", "#F4A261", "stochastic"),
]


def _read_rows(path: Path, proj: str | None) -> list[dict]:
    """Read CSV, filter by proj if given."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if proj:
        rows = [row for row in rows if row.get("proj") == proj]
    return rows


def _loss_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract step and train_loss."""
    steps, vals = [], []
    for row in rows:
        raw = row.get("train_loss", "")
        if raw in ("", "nan", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            steps.append(int(row["step"]))
            vals.append(value)
    return np.asarray(steps), np.asarray(vals)


def _ppl_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract step and train_ppl_capped."""
    steps, vals = [], []
    for row in rows:
        raw = row.get("train_ppl_capped", "")
        if raw in ("", "nan", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            steps.append(int(row["step"]))
            vals.append(value)
    return np.asarray(steps), np.asarray(vals)


def _throughput_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract step and tokens_per_sec."""
    steps, vals = [], []
    for row in rows:
        raw = row.get("tokens_per_sec", "")
        if raw in ("", "nan", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            steps.append(int(row["step"]))
            vals.append(value)
    return np.asarray(steps), np.asarray(vals)


def _grad_norm_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract step and grad_norm."""
    steps, vals = [], []
    for row in rows:
        raw = row.get("grad_norm", "")
        if raw in ("", "nan", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            steps.append(int(row["step"]))
            vals.append(value)
    return np.asarray(steps), np.asarray(vals)


def plot_loss_curves() -> None:
    """Training loss comparison."""
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(11, 5), tight_layout=True)

    for label, path, proj, color, _group in RUNS:
        rows = _read_rows(path, proj)
        steps, vals = _loss_curve(rows)
        if steps.size == 0:
            print(f"  {label}: no loss data")
            continue
        print(f"  {label}: {len(steps)} points, steps {steps.min()}-{steps.max()}")
        ax.plot(steps, vals, label=label, color=color, linewidth=2.2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Stochastic Methods · TinyLlama 1.1B · RefinedWeb · rank 8")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc="upper right")
    fig.savefig(OUT / "train_loss.png", dpi=300)
    plt.close(fig)


def plot_ppl_curves() -> None:
    """Training perplexity comparison."""
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(11, 5), tight_layout=True)

    for label, path, proj, color, _group in RUNS:
        rows = _read_rows(path, proj)
        steps, vals = _ppl_curve(rows)
        if steps.size == 0:
            continue
        ax.plot(steps, vals, label=label, color=color, linewidth=2.2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Training perplexity (capped)")
    ax.set_title("Training Perplexity · rank 8")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc="upper right")
    fig.savefig(OUT / "train_ppl.png", dpi=300)
    plt.close(fig)


def plot_throughput() -> None:
    """Tokens/sec comparison."""
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(11, 5), tight_layout=True)

    for label, path, proj, color, _group in RUNS:
        rows = _read_rows(path, proj)
        steps, vals = _throughput_curve(rows)
        if steps.size == 0:
            continue
        ax.plot(steps, vals, label=label, color=color, linewidth=2.2, alpha=0.8)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Training Throughput · rank 8")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc="upper right")
    fig.savefig(OUT / "throughput.png", dpi=300)
    plt.close(fig)


def plot_grad_norm() -> None:
    """Gradient norm comparison."""
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(11, 5), tight_layout=True)

    for label, path, proj, color, _group in RUNS:
        rows = _read_rows(path, proj)
        steps, vals = _grad_norm_curve(rows)
        if steps.size == 0:
            continue
        ax.plot(steps, vals, label=label, color=color, linewidth=2.2, alpha=0.8)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm")
    ax.set_title("Gradient Norm · rank 8")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc="upper right")
    fig.savefig(OUT / "grad_norm.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Plotting stochastic methods...")
    plot_loss_curves()
    plot_ppl_curves()
    plot_throughput()
    plot_grad_norm()
    print(f"Written to {OUT}/")


if __name__ == "__main__":
    main()
