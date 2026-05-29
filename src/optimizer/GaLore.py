import torch
from SVD import get_svd
import numpy as np
import matplotlib.pyplot as plt

class GaLoreProjector:
    def __init__(self, rank: int = 8, scale_factor: float = 0.25):
        self.rank = rank
        self.P = None
        self.scale_factor = scale_factor
        self.cfg = {"type": "classic"}
        self.transpose = None
        self.P_prev = None

    @torch.no_grad()
    def update_basis(
        self,
        grad: torch.Tensor,
        param_name: str | None = None,
        step: int | None = None,
        experiment=None,
    ):
        if self.transpose is None:
            self.transpose = grad.shape[0] > grad.shape[1]

        g = grad.T if self.transpose else grad
        U, S, _ = get_svd(g, **self.cfg)
        r = min(self.rank, U.shape[1])
        self.P = U[:, :r].to(grad.dtype)

        if experiment is not None and param_name is not None and step is not None:
            sv = S.float().cpu().numpy()
            cumvar = np.cumsum(sv ** 2) / np.sum(sv ** 2)
            experiment.log_histogram_3d(sv, name=f"singular_values/{param_name}", step=step)

            fig, axes = plt.subplots(1, 2, figsize=(10, 3), tight_layout=True)

            # ── cumulative variance ──────────────────────────────────────────────────
            ax = axes[0]
            ax.plot(range(1, len(cumvar) + 1), cumvar, color="steelblue", linewidth=1.5)
            ax.axvline(x=self.rank, color="red", linestyle="--", linewidth=1.2)
            ax.axhline(y=cumvar[self.rank - 1], color="red", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.annotate(
                f"{cumvar[self.rank - 1]:.1%} at rank={self.rank}",
                xy=(self.rank, cumvar[self.rank - 1]),
                xytext=(self.rank + 2, cumvar[self.rank - 1] - 0.1),
                fontsize=8, color="red",
            )
            ax.set_xlabel("rank")
            ax.set_ylabel("cumulative variance explained")
            ax.set_ylim(0, 1)
            ax.set_title("spectrum", fontsize=9)

            # ── principal angles ─────────────────────────────────────────────────────
            ax2 = axes[1]
            if self.P_prev is not None:
                M = self.P_prev.float().T @ self.P.float()
                sigma = torch.linalg.svdvals(M).clamp(-1, 1)
                angles = torch.acos(sigma).cpu().numpy() * (180 / np.pi)

                ax2.bar(range(1, len(angles) + 1), angles, color="steelblue", width=0.7)
                ax2.axhline(y=45, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="45°")
                ax2.axhline(y=90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="90°")
                ax2.set_xlabel("component")
                ax2.set_ylabel("principal angle (degrees)")
                ax2.set_ylim(0, 95)
                ax2.legend(fontsize=8)
            else:
                ax2.text(0.5, 0.5, "first update\n(no previous basis)",
                         ha="center", va="center", transform=ax2.transAxes,
                         fontsize=9, color="gray")

            ax2.set_title("subspace rotation since last update", fontsize=9)

            fig.suptitle(f"{param_name} | step {step}", fontsize=9)
            experiment.log_figure(figure=fig, figure_name=f"{param_name}/step{step:06d}")
            plt.close(fig)

            self.P_prev = self.P.clone()


    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.P.T @ grad.T
        return self.P.T @ grad

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.scale_factor * (self.P @ low_rank_grad).T
        return self.scale_factor * (self.P @ low_rank_grad)