import torch

from diagnostics import log_fisher_diagnostics, log_spectrum_diagnostics
from SVD import get_svd


class FisherGaLoreProjector:
    """
    GaLore-style projector that picks ``rank`` directions out of a wider pool
    of ``candidate_rank`` singular vectors.

    The selection score blends two signals:
      * ``fisher`` — the average squared projection of the *weight gradient*
        onto each candidate basis vector (an empirical Fisher proxy);
      * ``S_candidates`` — the singular values of the most recent SVD.

    Lifecycle (driven by the optimizer):
      1. ``accumulate_fisher(grad)`` is called on every optimizer step that
         produces a 2D weight gradient. It is a no-op before the first
         ``update_basis`` (no candidate pool yet).
      2. ``update_basis(grad, ...)`` is called periodically (per MiniAdam's
         ``update_gap``). It refreshes the candidate pool from a new SVD,
         picks the active basis ``P`` and logs diagnostics.

    The optimizer is responsible for guaranteeing that the SAME projector
    instance receives both ``accumulate_fisher`` and ``update_basis`` calls;
    see ``FisherGaLoreOptimizer`` for details.
    """

    PROJECTOR_NAME = "fisher_galore"
    SVD_CONFIG = {"type": "classic"}

    def __init__(
        self,
        rank: int = 128,
        candidate_rank: int = 512,
        scale_factor: float = 0.25,
        fisher_power: float = 1.0,
        singular_power: float = 0.0,
    ):
        self.rank = rank
        self.candidate_rank = candidate_rank
        self.scale_factor = scale_factor
        self.fisher_power = fisher_power
        self.singular_power = singular_power

        self.transpose: bool | None = None
        self.P: torch.Tensor | None = None
        self.P_prev: torch.Tensor | None = None

        self.U_candidates: torch.Tensor | None = None
        self.S_candidates: torch.Tensor | None = None

        self.fisher_energy: torch.Tensor | None = None
        self.fisher_steps: int = 0

    # ------------------------------------------------------------
    # Fisher accumulation (called by the optimizer every step)
    # ------------------------------------------------------------
    @torch.no_grad()
    def accumulate_fisher(self, grad: torch.Tensor) -> None:
        # Before the very first update_basis we have no candidate pool yet.
        if self.U_candidates is None or grad is None or grad.dim() != 2:
            return

        g = grad.T if self.transpose else grad
        if g.shape[0] != self.U_candidates.shape[0]:
            # Defensive: skip silently on unexpected shape changes.
            return

        g_f = g.float()
        U = self.U_candidates.to(device=g_f.device, dtype=g_f.dtype)
        proj = U.T @ g_f                                      # [c, M]
        energy = proj.pow(2).mean(dim=1).detach().cpu()        # [c]

        if self.fisher_energy is None:
            self.fisher_energy = energy
        else:
            self.fisher_energy = self.fisher_energy + energy
        self.fisher_steps += 1

    # ------------------------------------------------------------
    # Basis update (called periodically by the optimizer)
    # ------------------------------------------------------------
    @torch.no_grad()
    def update_basis(
        self,
        grad: torch.Tensor,
        param_name: str | None = None,
        step: int | None = None,
        experiment=None,
    ) -> None:
        if self.transpose is None:
            self.transpose = grad.shape[0] > grad.shape[1]

        g = grad.T if self.transpose else grad
        U, S, _ = get_svd(g, **self.SVD_CONFIG)

        c = min(self.candidate_rank, U.shape[1])
        U = U[:, :c]
        S = S[:c]

        previous_basis = self.P
        bootstrap = self.U_candidates is None

        # Snapshots of what we used for SELECTION (the OLD pool/Fisher),
        # needed for diagnostics before we overwrite them below.
        prev_S = self.S_candidates
        prev_fisher = None
        fisher_steps_logged = int(self.fisher_steps)

        if bootstrap:
            # First call: no Fisher data yet -> seed with top-rank from SVD.
            r = min(self.rank, c)
            self.P = U[:, :r].to(grad.dtype)
            selected_idx = list(range(r))
        else:
            prev_fisher = self.fisher_energy / max(self.fisher_steps, 1)
            score = (
                prev_fisher.pow(self.fisher_power)
                * prev_S.pow(self.singular_power)
            )
            r = min(self.rank, score.numel())
            idx = torch.topk(score, r).indices
            self.P = self.U_candidates[:, idx].to(
                device=grad.device, dtype=grad.dtype
            )
            selected_idx = idx.detach().cpu().tolist()

        # Refresh candidate pool and reset Fisher accumulator.
        self.U_candidates = U.float().cpu()
        self.S_candidates = S.float().cpu()
        self.fisher_energy = torch.zeros(c)
        self.fisher_steps = 0
        """
        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.rank,
            projector_name=self.PROJECTOR_NAME,
            param_name=param_name,
            step=step,
            experiment=experiment,
            extra={
                "fisher_steps": fisher_steps_logged,
                "bootstrap": bool(bootstrap),
                "selected_idx": selected_idx,
                "candidate_rank": int(c),
            },
        )
        """

        # The dedicated Fisher chart only makes sense once we have data.
        if prev_fisher is not None and prev_S is not None:
            log_fisher_diagnostics(
                fisher=prev_fisher,
                singular_values=prev_S,
                selected_idx=selected_idx,
                rank=self.rank,
                fisher_steps=fisher_steps_logged,
                projector_name=self.PROJECTOR_NAME,
                param_name=param_name,
                step=step,
                experiment=experiment,
            )

        self.P_prev = self.P.detach().clone()

    # ------------------------------------------------------------
    # Projection / reconstruction
    # ------------------------------------------------------------
    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        return self.P.T @ grad.T if self.transpose else self.P.T @ grad

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        out = self.P @ low_rank_grad
        return self.scale_factor * (out.T if self.transpose else out)
