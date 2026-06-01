import torch

from diagnostics import log_fisher_diagnostics
from SVD import get_svd


class BlockFisherGaLoreProjector:
    """
    GaLore-style projector that selects a rank-``r`` subspace by diagonalising
    a *block* empirical Fisher inside a wider pool of ``candidate_rank`` SVD
    directions.

    Difference from :class:`FisherGaLoreProjector`
    ----------------------------------------------
    ``FisherGaLoreProjector`` keeps only the **diagonal** of the gradient
    covariance in the SVD basis, i.e. ``E[(uᵢᵀg)²]`` per candidate ``i``, and
    then *picks* ``rank`` existing SVD vectors. That implicitly assumes the
    left-singular vectors ``U`` already diagonalise the Fisher — which is
    generally false.

    This projector instead accumulates the full ``c×c`` covariance

        C = E[ (Uᵀg)(Uᵀg)ᵀ ]            # c == candidate_rank

    and eigendecomposes it. The active basis becomes ``P = U · Q[:, :r]`` where
    ``Q`` are the leading eigenvectors of ``C``. Rotating the candidate pool
    into the eigenbasis of the (restricted) empirical Fisher captures the
    *correlations* between SVD directions that the diagonal version discards.

    Relation to K-FAC / EKFAC
    -------------------------
    This is a memory-frugal, single-factor analogue of EKFAC (George et al.,
    NeurIPS 2018, https://arxiv.org/abs/1806.03884). EKFAC's core insight is
    that curvature should be estimated *in the eigenbasis* of the Kronecker
    factors rather than assuming those factors already diagonalise the Fisher.
    We apply the same correction, but restricted to the small ``c``-dimensional
    span of the top SVD directions of the weight gradient, so the extra state
    is a single ``c×c`` matrix instead of K-FAC's full ``[in,in]`` and
    ``[out,out]`` factors — and, crucially, no forward/backward hooks or model
    coupling are required (cf. ``FisherGaLoreOptimizer``).

    The lifecycle (``accumulate_fisher`` every step, ``update_basis`` every
    ``update_gap`` steps) and the ``project`` / ``reconstruct`` contract are
    identical to :class:`FisherGaLoreProjector`, so it is a drop-in projector
    for ``FisherMiniAdam``.
    """

    PROJECTOR_NAME = "block_fisher_galore"
    SVD_CONFIG = {"type": "classic"}

    def __init__(
        self,
        rank: int = 128,
        candidate_rank: int = 512,
        scale_factor: float = 0.25,
        eigen_eps: float = 1e-12,
    ):
        self.rank = rank
        self.candidate_rank = candidate_rank
        self.scale_factor = scale_factor
        self.eigen_eps = eigen_eps

        self.transpose: bool | None = None
        self.P: torch.Tensor | None = None
        self.P_prev: torch.Tensor | None = None

        self.U_candidates: torch.Tensor | None = None
        self.S_candidates: torch.Tensor | None = None

        # Block empirical Fisher accumulated in the candidate basis: [c, c].
        self.fisher_cov: torch.Tensor | None = None
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
        # Full block covariance, averaged over the M "samples" (gradient
        # columns) so its diagonal matches FisherGaLoreProjector's energy.
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
        prev_fisher_diag = None
        fisher_steps_logged = int(self.fisher_steps)

        if bootstrap:
            # First call (or no Fisher yet): seed with top-rank from SVD.
            r = min(self.rank, c)
            self.P = U[:, :r].to(grad.dtype)
            selected_idx = list(range(r))
        else:
            # Diagonalise the block Fisher accumulated against the OLD pool.
            C = self.fisher_cov
            eye = torch.eye(C.shape[0], dtype=C.dtype)
            evals, evecs = torch.linalg.eigh(C + self.eigen_eps * eye)
            r = min(self.rank, evecs.shape[1])
            # eigh returns ascending eigenvalues -> take the last r columns
            # (fp32, CPU); column order is irrelevant for the leverage below.
            top = evecs[:, -r:]                              # [c, r]
            Q = top.flip(dims=(1,)).to(                       # [c, r], desc order
                device=grad.device, dtype=grad.dtype
            )
            U_prev = self.U_candidates.to(device=grad.device, dtype=grad.dtype)
            # Rotate the (old) candidate basis into the Fisher eigenbasis.
            self.P = (U_prev @ Q).to(grad.dtype)             # [m, r], orthonormal

            # For diagnostics we expose the per-candidate Fisher (the diagonal
            # of C), matching what FisherGaLoreProjector logs.
            prev_fisher_diag = torch.diagonal(C).clone() / max(self.fisher_steps, 1)
            # The basis is a rotation, not an index pick, so there is no single
            # selected u_i. We highlight the candidates with the largest
            # leverage in the chosen Fisher eigen-subspace:
            #     leverage_i = Sum_{k<r} Q[i, k]**2   (row energy, sums to r),
            # i.e. the SVD directions that dominate the selected subspace.
            leverage = top.pow(2).sum(dim=1)                 # [c], sums to r
            selected_idx = torch.topk(leverage, r).indices.detach().cpu().tolist()

        # Refresh candidate pool and reset the Fisher accumulator.
        self.U_candidates = U.float().cpu()
        self.S_candidates = S.float().cpu()
        self.fisher_cov = None
        self.fisher_steps = 0

        # The dedicated Fisher chart only makes sense once we have data.
        if prev_fisher_diag is not None and prev_S is not None:
            log_fisher_diagnostics(
                fisher=prev_fisher_diag,
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
