"""Generate preliminary experiment figures referenced by статья.tex."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LOSS_RUNS = [
    ("GaLore2-mini", ROOT / "runs/refinedweb_full_galore2_1500_seq2048.csv", "#4472C4"),
    ("LOTUS-mini", ROOT / "runs/refinedweb_full_lotus_1500_seq2048.csv", "#ED7D31"),
]
VRAM_CSV = ROOT / "runs/refinedweb_full_galore2_1500_seq2048.csv"


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


def plot_loss_curves() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for label, path, color in LOSS_RUNS:
        steps, losses = _read_loss(path)
        ax.plot(steps, losses, color=color, alpha=0.35, linewidth=0.8)
        ax.plot(steps, _rolling_median(losses), color=color, linewidth=2.0, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = ROOT / "experiment_loss_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_vram_components() -> None:
    with VRAM_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
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
    ax.set_title(f"Peak VRAM decomposition (step {peak['step']})")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = ROOT / "experiment_vram_components.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_vram_components()
