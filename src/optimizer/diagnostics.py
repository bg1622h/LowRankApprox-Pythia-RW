import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def _plot_fisher_spectrum(
    *,
    param_name: str,
    step: int,
    fisher: np.ndarray,
    svd_energy: np.ndarray,
    rank: int,
):
    """
    Two-panel figure for a single basis update.

    The top panel directly tests GaLore's core assumption — that the *first*
    ``r`` left-singular vectors carry the most Fisher information — by drawing
    two independent overlays on the same per-candidate bars:

      * FILL colour  — the ``first r`` candidates in SVD order (crimson),
                       i.e. exactly what vanilla GaLore keeps.
      * HATCH border — the ``top r`` candidates ranked by Fisher importance.

    When GaLore's assumption holds the crimson fills and the hatched borders
    coincide on the leftmost bars. Any hatched-but-blue bar on the right is a
    high-Fisher direction that GaLore discards (the assumption breaking).

    The bottom panel overlays normalized Fisher vs SVD energy for the same
    visual sanity check on the whole spectrum.
    """
    c = len(fisher)
    x = np.arange(1, c + 1)
    fisher_norm = fisher / max(float(np.sum(fisher)), 1e-12)
    svd_norm = svd_energy / max(float(np.sum(svd_energy)), 1e-12)

    r_eff = int(min(rank, c))
    # Layer 1 (fill): the first r candidates in SVD order — GaLore's pick.
    first_r_mask = np.zeros(c, dtype=bool)
    first_r_mask[:r_eff] = True
    # Layer 2 (hatch): the top r candidates by Fisher importance.
    fisher_top_mask = np.zeros(c, dtype=bool)
    if r_eff > 0:
        top_fisher_idx = np.argsort(fisher)[::-1][:r_eff]
        fisher_top_mask[top_fisher_idx] = True

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, tight_layout=True)

    ax = axes[0]
    colors = np.where(first_r_mask, "crimson", "steelblue")
    bars = ax.bar(x, fisher_norm, color=colors, width=0.8)
    for bar, is_top in zip(bars, fisher_top_mask):
        if is_top:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.2)
            bar.set_hatch("///")
    ax.set_ylabel("Fisher importance (norm.)")
    ax.set_title(
        f"Fisher per candidate u_i  |  fill=first {r_eff} (SVD), "
        f"hatch=top {r_eff} (Fisher)  |  r={rank}/{c}",
        fontsize=9,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="crimson", label=f"first {r_eff} (SVD / GaLore)"),
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="rest"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="black",
            hatch="///", label=f"top {r_eff} (Fisher)",
        ),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    ax2 = axes[1]
    ax2.plot(x, fisher_norm, color="crimson", linewidth=1.5, label="Fisher (norm.)")
    ax2.plot(
        x, svd_norm, color="steelblue", linewidth=1.5, linestyle="--",
        label="SVD energy (norm.)",
    )
    ax2.set_xlabel("candidate index i")
    ax2.set_ylabel("normalized score")
    ax2.legend(fontsize=8)

    fig.suptitle(f"{param_name} | step {step}", fontsize=9)
    return fig


def log_fisher_diagnostics(
    *,
    fisher: torch.Tensor,
    singular_values: torch.Tensor,
    selected_idx,
    rank: int,
    fisher_steps: int,
    projector_name: str,
    param_name: str | None,
    step: int | None,
    experiment=None,
):
    """
    Persist a per-update snapshot of the Fisher importance distribution.

    What is logged:
      * JSONL record with the full Fisher / SVD-energy / selected-index data.
      * Comet metrics: mean / max / effective rank / accumulation steps.
      * Comet histogram_3d of Fisher across candidates (time series).
      * Comet figure: per-candidate bar chart + Fisher-vs-SVD overlay.
    """
    if param_name is None or step is None:
        return

    fisher_cpu = fisher.float().detach().cpu()
    sv_cpu = singular_values.float().detach().cpu()
    svd_energy = sv_cpu.square()
    selected_list = [int(i) for i in selected_idx]

    fisher_total = float(fisher_cpu.sum())
    fisher_mean = float(fisher_cpu.mean()) if fisher_cpu.numel() else 0.0
    fisher_max = float(fisher_cpu.max()) if fisher_cpu.numel() else 0.0
    probs = fisher_cpu / max(fisher_total, 1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    fisher_effective_rank = float(torch.exp(entropy))

    record = {
        "projector": projector_name,
        "param_name": param_name,
        "param_group": param_name.split("/", 1)[0],
        "step": int(step),
        "rank": int(rank),
        "fisher_accumulation_steps": int(fisher_steps),
        "fisher": fisher_cpu.tolist(),
        "fisher_mean": fisher_mean,
        "fisher_max": fisher_max,
        "fisher_effective_rank": fisher_effective_rank,
        "svd_energy": svd_energy.tolist(),
        "selected_idx": selected_list,
    }

    directory = (
        getattr(experiment, "spectrum_log_dir", None)
        if experiment is not None else None
    )
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"fisher_{_safe_name(param_name)}.jsonl"
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if experiment is None:
        return

    metrics = {
        f"fisher/{projector_name}/{param_name}/mean": fisher_mean,
        f"fisher/{projector_name}/{param_name}/max": fisher_max,
        f"fisher/{projector_name}/{param_name}/effective_rank": fisher_effective_rank,
        f"fisher/{projector_name}/{param_name}/accumulation_steps": int(fisher_steps),
    }
    skip_media = bool(getattr(experiment, "no_comet_figures", True))
    try:
        experiment.log_metrics(metrics, step=step)
        if skip_media:
            return
        experiment.log_histogram_3d(
            fisher_cpu.numpy(),
            name=f"fisher_per_candidate/{projector_name}/{param_name}",
            step=step,
        )
        fig = _plot_fisher_spectrum(
            param_name=param_name,
            step=step,
            fisher=fisher_cpu.numpy(),
            svd_energy=svd_energy.numpy(),
            rank=rank,
        )
        try:
            experiment.log_figure(
                figure=fig,
                figure_name=(
                    f"fisher/{projector_name}/{_safe_name(param_name)}/step{step:06d}"
                ),
            )
        finally:
            plt.close(fig)
    except Exception:
        # Diagnostics should never break training.
        pass


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
    skip_media = bool(getattr(experiment, "no_comet_figures", True))
    try:
        experiment.log_metrics(metrics, step=step)
        if skip_media:
            return
        experiment.log_histogram_3d(
            sv_cpu.numpy(),
            name=f"singular_values/{projector_name}/{param_name}",
            step=step,
        )
        fig = _plot_spectrum(param_name, step, rank, singular_values, angles)
        try:
            experiment.log_figure(
                figure=fig,
                figure_name=f"spectrum/{projector_name}/{_safe_name(param_name)}/step{step:06d}",
            )
        finally:
            plt.close(fig)
    except Exception:
        # Diagnostics should never break training.
        pass
