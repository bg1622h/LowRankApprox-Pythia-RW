import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace("?", "unknown")
        .replace(" ", "_")
    )


def _spectrum_stats(singular_values: torch.Tensor, rank: int) -> dict:
    sv = singular_values.float().detach().cpu()
    if sv.numel() == 0:
        return {
            "num_singular_values": 0,
            "rank_energy": float("nan"),
            "rank_90": None,
            "rank_95": None,
            "rank_99": None,
            "effective_rank": float("nan"),
        }

    energy = sv.square()
    total_energy = energy.sum().clamp_min(1e-12)
    cumulative = torch.cumsum(energy, dim=0) / total_energy
    probs = energy / total_energy
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    effective_rank = torch.exp(entropy).item()

    clipped_rank = min(max(rank, 1), sv.numel())
    return {
        "num_singular_values": int(sv.numel()),
        "rank_energy": float(cumulative[clipped_rank - 1].item()),
        "rank_90": int(torch.searchsorted(cumulative, 0.90).item()) + 1,
        "rank_95": int(torch.searchsorted(cumulative, 0.95).item()) + 1,
        "rank_99": int(torch.searchsorted(cumulative, 0.99).item()) + 1,
        "effective_rank": float(effective_rank),
    }


def _principal_angles_degrees(previous_basis, current_basis):
    if previous_basis is None or current_basis is None:
        return []
    if previous_basis.shape != current_basis.shape:
        return []

    M = previous_basis.float().T @ current_basis.float()
    sigma = torch.linalg.svdvals(M).clamp(-1, 1)
    return (torch.acos(sigma).detach().cpu().numpy() * (180 / np.pi)).tolist()


def _append_jsonl(directory, param_name, record):
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"singular_values_{_safe_name(param_name)}.jsonl"
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _plot_spectrum(param_name, step, rank, singular_values, angles):
    sv = singular_values.float().detach().cpu().numpy()
    energy = sv ** 2
    cumulative = np.cumsum(energy) / max(float(np.sum(energy)), 1e-12)
    clipped_rank = min(max(rank, 1), len(cumulative))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), tight_layout=True)
    ax = axes[0]
    ax.plot(range(1, len(cumulative) + 1), cumulative, color="steelblue", linewidth=1.5)
    ax.axvline(x=clipped_rank, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(
        y=cumulative[clipped_rank - 1],
        color="red",
        linestyle=":",
        linewidth=0.8,
        alpha=0.6,
    )
    ax.annotate(
        f"{cumulative[clipped_rank - 1]:.1%} at rank={clipped_rank}",
        xy=(clipped_rank, cumulative[clipped_rank - 1]),
        xytext=(clipped_rank + 2, max(cumulative[clipped_rank - 1] - 0.1, 0.0)),
        fontsize=8,
        color="red",
    )
    ax.set_xlabel("rank")
    ax.set_ylabel("cumulative variance explained")
    ax.set_ylim(0, 1)
    ax.set_title("spectrum", fontsize=9)

    ax2 = axes[1]
    if angles:
        ax2.bar(range(1, len(angles) + 1), angles, color="steelblue", width=0.7)
        ax2.axhline(y=45, color="orange", linestyle="--", linewidth=1, alpha=0.7)
        ax2.axhline(y=90, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_xlabel("component")
        ax2.set_ylabel("principal angle (degrees)")
        ax2.set_ylim(0, 95)
    else:
        ax2.text(
            0.5,
            0.5,
            "first update\n(no previous basis)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=9,
            color="gray",
        )
    ax2.set_title("subspace rotation since last update", fontsize=9)
    fig.suptitle(f"{param_name} | step {step}", fontsize=9)
    return fig


def log_spectrum_diagnostics(
    *,
    singular_values: torch.Tensor,
    basis: torch.Tensor,
    previous_basis,
    rank: int,
    projector_name: str,
    param_name: str | None,
    step: int | None,
    experiment=None,
    extra: dict | None = None,
):
    if param_name is None or step is None:
        return

    stats = _spectrum_stats(singular_values, rank)
    angles = _principal_angles_degrees(previous_basis, basis)
    sv_cpu = singular_values.float().detach().cpu()
    record = {
        "projector": projector_name,
        "param_name": param_name,
        "param_group": param_name.split("/", 1)[0],
        "step": int(step),
        "rank": int(rank),
        "singular_values": sv_cpu.tolist(),
        "top_singular_values": sv_cpu[: min(32, sv_cpu.numel())].tolist(),
        "principal_angles_degrees": angles,
        **stats,
    }
    if extra:
        record.update(extra)

    directory = getattr(experiment, "spectrum_log_dir", None) if experiment is not None else None
    _append_jsonl(directory, param_name, record)

    if experiment is None:
        return

    metrics = {
        f"spectrum/{projector_name}/{param_name}/rank_energy": stats["rank_energy"],
        f"spectrum/{projector_name}/{param_name}/effective_rank": stats["effective_rank"],
    }
    if angles:
        metrics[f"spectrum/{projector_name}/{param_name}/mean_angle"] = float(np.mean(angles))
        metrics[f"spectrum/{projector_name}/{param_name}/max_angle"] = float(np.max(angles))
    try:
        experiment.log_metrics(metrics, step=step)
        experiment.log_histogram_3d(
            sv_cpu.numpy(),
            name=f"singular_values/{projector_name}/{param_name}",
            step=step,
        )
        fig = _plot_spectrum(param_name, step, rank, singular_values, angles)
        experiment.log_figure(
            figure=fig,
            figure_name=f"spectrum/{projector_name}/{_safe_name(param_name)}/step{step:06d}",
        )
        plt.close(fig)
    except Exception:
        # Diagnostics should never break training.
        pass
