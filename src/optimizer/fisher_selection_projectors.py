import torch

from diagnostics import log_fisher_diagnostics
from SVD import get_svd


class _WindowFisherProjector:
    """
    Shared base for projectors that build a rank-``r`` basis by *selecting*
    columns from a pool of ``candidate_rank`` SVD directions, scored by an
    empirical Fisher accumulated over the update window.

    The Fisher accumulation and the per-update bookkeeping are intentionally
    identical to :class:`BlockFisherGaLoreProjector`: every step we project the
    weight gradient onto the (frozen) candidate basis ``U`` and accumulate the
    full block covariance

        C = E[ (Uᵀg)(Uᵀg)ᵀ ]            # c × c, c == candidate_rank

    so that ``diag(C)`` is exactly the per-candidate Fisher importance
    ``E[(uᵢᵀg)²]``. Subclasses differ **only** in how they turn that Fisher
    vector into ``r`` selected candidate indices, via :meth:`_select_indices`.

    Keeping the covariance (not just its diagonal) means all Fisher projectors
    share byte-for-byte the same accumulation path and log the same Fisher
    statistic; the extra ``c×c`` state is ~1 MB for ``c=512`` in fp32.

    Lifecycle, ``project`` / ``reconstruct`` and the diagnostics call match the
    other Fisher projectors, so these are drop-in projectors for
    ``FisherMiniAdam``.
    """

    # Overridden by concrete subclasses for log grouping.
    PROJECTOR_NAME = "window_fisher"
    SVD_CONFIG = {"type": "classic"}

    def __init__(
        self,
        rank: int = 128,
        candidate_rank: int = 512,
        scale_factor: float = 0.25,
    ):
        self.rank = rank
        self.candidate_rank = candidate_rank
        self.scale_factor = scale_factor

        self.transpose: bool | None = None
        self.P: torch.Tensor | None = None
        self.P_prev: torch.Tensor | None = None

        self.U_candidates: torch.Tensor | None = None
        self.S_candidates: torch.Tensor | None = None

        # Block empirical Fisher accumulated in the candidate basis: [c, c].
        self.fisher_cov: torch.Tensor | None = None
        self.fisher_steps: int = 0

    # ------------------------------------------------------------
    # Selection hook (implemented by subclasses)
    # ------------------------------------------------------------
    def _select_indices(self, fisher_diag: torch.Tensor, r: int) -> torch.Tensor:
        """
        Map the per-candidate Fisher importance (CPU float tensor, length ``c``)
        to ``r`` distinct candidate indices (1D LongTensor). Must return exactly
        ``r`` unique indices in ``[0, c)``.
        """
        raise NotImplementedError

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
        # Full block covariance, averaged over the M "samples" (gradient
        # columns); its diagonal is the per-candidate Fisher importance.
        cov = (proj @ proj.T) / proj.shape[1]                # [c, c]
        cov = cov.detach().cpu()

        if self.fisher_cov is None:
            self.fisher_cov = cov
        else:
            self.fisher_cov = self.fisher_cov + cov
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

        bootstrap = self.U_candidates is None or self.fisher_cov is None

        # Snapshot of the OLD candidate spectrum used for selection; needed for
        # diagnostics before we overwrite the pool below.
        prev_S = self.S_candidates
        prev_fisher = None
        fisher_steps_logged = int(self.fisher_steps)

        if bootstrap:
            # First call (or no Fisher yet): seed with top-rank from SVD.
            r = min(self.rank, c)
            self.P = U[:, :r].to(grad.dtype)
            selected_idx = list(range(r))
        else:
            # Per-candidate Fisher importance = diagonal of the block Fisher.
            fisher_diag = torch.diagonal(self.fisher_cov).clone() / max(
                self.fisher_steps, 1
            )                                                # [c], CPU float
            prev_fisher = fisher_diag
            r = min(self.rank, c)
            idx = self._select_indices(fisher_diag, r)       # LongTensor, CPU
            U_prev = self.U_candidates.to(device=grad.device, dtype=grad.dtype)
            self.P = U_prev[:, idx.to(U_prev.device)].to(grad.dtype)
            selected_idx = idx.detach().cpu().tolist()

        # Refresh candidate pool and reset the Fisher accumulator.
        self.U_candidates = U.float().cpu()
        self.S_candidates = S.float().cpu()
        self.fisher_cov = None
        self.fisher_steps = 0

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


class TopKFisherGaLoreProjector(_WindowFisherProjector):
    """
    Selects the ``r`` candidate directions with the **largest** windowed Fisher
    importance: ``idx = topk(diag(C), r)``. This is the greedy, deterministic
    baseline for the Fisher selection family.
    """

    PROJECTOR_NAME = "topk_fisher_galore"

    @torch.no_grad()
    def _select_indices(self, fisher_diag: torch.Tensor, r: int) -> torch.Tensor:
        return torch.topk(fisher_diag, r).indices


class SoftmaxFisherGaLoreProjector(_WindowFisherProjector):
    """
    Stochastic counterpart of :class:`TopKFisherGaLoreProjector`: samples ``r``
    distinct candidates **without replacement** from a categorical distribution
    whose logits are the windowed Fisher importances.

    The raw Fisher values span many orders of magnitude and their scale drifts
    across layers and steps, so we first normalise the logits by their mean
    (a scale-free transform) and then apply a temperature:

        p_i = softmax( (fisher_i / mean(fisher)) / temperature )

    ``temperature = 1.0`` is a neutral default: lower values sharpen toward the
    top-k pick, higher values approach uniform exploration over the candidate
    pool.
    """

    PROJECTOR_NAME = "softmax_fisher_galore"

    def __init__(
        self,
        rank: int = 128,
        candidate_rank: int = 512,
        scale_factor: float = 0.25,
        temperature: float = 1.0,
        logit_eps: float = 1e-12,
    ):
        super().__init__(
            rank=rank,
            candidate_rank=candidate_rank,
            scale_factor=scale_factor,
        )
        self.temperature = temperature
        self.logit_eps = logit_eps

    @torch.no_grad()
    def _select_indices(self, fisher_diag: torch.Tensor, r: int) -> torch.Tensor:
        logits = fisher_diag / fisher_diag.mean().clamp_min(self.logit_eps)
        probs = torch.softmax(logits / self.temperature, dim=0)
        return torch.multinomial(probs, r, replacement=False)
