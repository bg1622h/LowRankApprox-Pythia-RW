import torch

from diagnostics import log_fisher_diagnostics
from SVD import get_svd


# Default hyper-parameters shared by all block-Fisher projectors.
_DEFAULT_RANK = 128
_DEFAULT_CANDIDATE_RANK = 64
_DEFAULT_SCALE_FACTOR = 0.25
_DEFAULT_EIGEN_EPS = 1e-12

# Whitened variant only.
_DEFAULT_FISHER_EMA = 0.99       # EMA decay for the windowed block Fisher.
_DEFAULT_WHITEN_FLOOR = 1e-8     # Floor on eigenvalues before inverse sqrt.
_DEFAULT_WHITEN_CLIP = 1e4       # Clip on whitening scale to bound updates.

# Stochastic variant only.
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_MIN_TEMPERATURE = 0.25
_DEFAULT_MAX_TEMPERATURE = 4.0
_DEFAULT_TOPK_DETERMINISTIC_OFFSET = 1   # rank - this -> anchored greedy slots.
_DEFAULT_NOISE_SIGMA_RATIO = 0.01
_DEFAULT_MIN_ENERGY_RATIO = 1e-4

# Adaptive-rank variant only.
_DEFAULT_ENERGY_THRESHOLD = 0.98
_DEFAULT_ENTROPY_SCALE = 1.0
_DEFAULT_MIN_RANK_OFFSET = 2             # rank - this -> floor for adaptive rank.


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
        rank: int = _DEFAULT_RANK,
        candidate_rank: int = _DEFAULT_CANDIDATE_RANK,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
        eigen_eps: float = _DEFAULT_EIGEN_EPS,
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


class WhitenedBlockFisherGaLoreProjector(BlockFisherGaLoreProjector):
    """
    Natural-gradient ("whitened") variant of :class:`BlockFisherGaLoreProjector`.

    Motivation
    ----------
    For a *single* gradient snapshot ``g = U S Vᵀ`` measured against its own
    SVD basis ``U`` we have ``Uᵀg = S Vᵀ`` and hence

        F_i = (1/M) ‖uᵢᵀ g‖² = sᵢ² / M ,

    so the per-candidate empirical Fisher is *identical* (up to a constant) to
    the squared singular spectrum. Selecting top-Fisher candidates therefore
    collapses to GaLore's top-SVD pick and brings no new information. The
    plain :class:`BlockFisherGaLoreProjector` partially escapes this by
    accumulating Fisher across multiple steps, but as long as Fisher is used
    as a *selector* the gain is limited.

    This variant uses Fisher as a **preconditioner** instead of a selector:

      1. *Centring* (Kunstner, Hennig & Balles, NeurIPS 2019) — subtract the
         column mean of the projected gradient before accumulating, so the
         covariance reflects per-token variance rather than the mean drift
         that otherwise dominates the empirical Fisher of weight gradients.
      2. *EMA accumulation* with decay ``fisher_ema`` — extends the effective
         Fisher horizon beyond a single ``update_gap`` window without growing
         memory; the running estimate is reset together with the candidate
         pool to stay aligned with the basis it is expressed in.
      3. *Whitening* — after eigendecomposing the block Fisher ``C = Q Λ Qᵀ``
         in the old candidate basis we keep ``P = U Q[:, :r]`` (same active
         basis as the parent), but additionally store ``w = Λ_r^{-1/2}``.
         ``project`` then returns ``w ⊙ (Pᵀg)`` and ``reconstruct`` applies
         the same ``w`` symmetrically, so the net low-rank update direction
         is ``P Λ_r⁻¹ Pᵀ g`` — a rank-``r`` natural gradient (Amari, 1998).

    Theoretical relation to classic GaLore
    --------------------------------------
    With ``fisher_ema → 0`` and ``center=False`` the running Fisher equals the
    SVD spectrum of the *latest* gradient (see the lemma above), so eigh
    returns the canonical basis and ``Λ_r`` becomes ``Sᵣ²/M``. Whitening then
    multiplies the ``i``-th low-rank coordinate by ``√M / sᵢ`` — the natural
    gradient "inverts" the singular spectrum that GaLore would have followed.
    With non-trivial EMA + centring the off-diagonal of ``C`` is generally
    non-zero, so ``P`` truly differs from the top-SVD pick and ``Λ_r`` is no
    longer ``Sᵣ²``.
    """

    PROJECTOR_NAME = "whitened_block_fisher_galore"

    def __init__(
        self,
        rank: int = _DEFAULT_RANK,
        candidate_rank: int = _DEFAULT_CANDIDATE_RANK,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
        eigen_eps: float = _DEFAULT_EIGEN_EPS,
        fisher_ema: float = _DEFAULT_FISHER_EMA,
        whiten_floor: float = _DEFAULT_WHITEN_FLOOR,
        whiten_clip: float = _DEFAULT_WHITEN_CLIP,
        center: bool = True,
        whiten: bool = True,
    ):
        super().__init__(
            rank=rank,
            candidate_rank=candidate_rank,
            scale_factor=scale_factor,
            eigen_eps=eigen_eps,
        )
        if not 0.0 < fisher_ema < 1.0:
            raise ValueError(f"fisher_ema must be in (0, 1), got {fisher_ema}")
        self.fisher_ema = fisher_ema
        self.whiten_floor = whiten_floor
        self.whiten_clip = whiten_clip
        self.center = center
        self.whiten = whiten

        # Cached natural-gradient scale Λ_r^{-1/2}, [r] in grad device/dtype.
        self.whiten_diag: torch.Tensor | None = None

        # Diagnostic: rank actually used in the last update_basis call. For
        # the base whitened projector this is always ``self.rank``; for the
        # adaptive subclass it is the spectrum-chosen value.
        self.last_active_rank: int = rank

    # ------------------------------------------------------------
    # Selection hooks (overridden by stochastic / adaptive subclasses)
    # ------------------------------------------------------------
    @torch.no_grad()
    def _choose_rank_from_eigs(self, eigs_desc: torch.Tensor, max_rank: int) -> int:
        """Default: keep the configured rank, clipped to what is available."""
        return max(1, min(self.rank, max_rank))

    @torch.no_grad()
    def _select_indices_from_eigs(
        self, eigs_desc: torch.Tensor, r: int
    ) -> torch.Tensor:
        """Default: greedy top-r over the descending eigenvalue spectrum."""
        return torch.arange(r, device=eigs_desc.device)

    # ------------------------------------------------------------
    # Fisher accumulation — centring + EMA
    # ------------------------------------------------------------
    @torch.no_grad()
    def accumulate_fisher(self, grad: torch.Tensor) -> None:
        if self.U_candidates is None or grad is None or grad.dim() != 2:
            return

        g = grad.T if self.transpose else grad
        if g.shape[0] != self.U_candidates.shape[0]:
            return

        g_f = g.float()
        U = self.U_candidates.to(device=g_f.device, dtype=g_f.dtype)
        proj = U.T @ g_f                                      # [c, M]
        if self.center and proj.shape[1] > 1:
            # Subtract the per-row mean over the M gradient columns. This
            # removes the systematic mean-shift component (Kunstner et al.,
            # 2019) that otherwise dominates the empirical Fisher and makes
            # it collapse onto the top singular direction.
            proj = proj - proj.mean(dim=1, keepdim=True)
        cov = (proj @ proj.T) / max(proj.shape[1], 1)         # [c, c]
        cov = cov.detach().cpu()

        if self.fisher_cov is None:
            self.fisher_cov = cov
        else:
            # EMA in the *current* candidate basis. The accumulator is reset
            # whenever ``update_basis`` rotates the basis, so we never mix
            # covariances expressed in different ``U`` snapshots.
            self.fisher_cov = (
                self.fisher_ema * self.fisher_cov + (1.0 - self.fisher_ema) * cov
            )
        self.fisher_steps += 1

    # ------------------------------------------------------------
    # Basis update — eigh + whitening cache
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

        prev_S = self.S_candidates
        prev_fisher_diag = None
        fisher_steps_logged = int(self.fisher_steps)

        if bootstrap:
            r = min(self.rank, c)
            self.P = U[:, :r].to(grad.dtype)
            # No Fisher yet -> disable whitening for this window.
            self.whiten_diag = None
            selected_idx = list(range(r))
            self.last_active_rank = r
        else:
            C = self.fisher_cov
            eye = torch.eye(C.shape[0], dtype=C.dtype)
            evals, evecs = torch.linalg.eigh(C + self.eigen_eps * eye)
            # eigh returns ascending order; reverse so index 0 is the largest.
            eigs_desc = evals.flip(dims=(0,))                 # [c], desc
            evecs_desc = evecs.flip(dims=(1,))                # [c, c], desc cols

            # Hook 1: how many eigen-directions to keep this window.
            r = self._choose_rank_from_eigs(eigs_desc, evecs_desc.shape[1])
            # Hook 2: which r eigen-directions (greedy by default, stochastic
            # in subclasses). Indices reference the desc-sorted spectrum.
            idx = self._select_indices_from_eigs(eigs_desc, r)
            idx = idx.to(device=evecs_desc.device)

            top = evecs_desc[:, idx]                          # [c, r]
            top_eigs = eigs_desc[idx]                         # [r]
            Q = top.to(device=grad.device, dtype=grad.dtype)
            U_prev = self.U_candidates.to(device=grad.device, dtype=grad.dtype)
            self.P = (U_prev @ Q).to(grad.dtype)             # [m, r]

            # Λ_r^{-1/2}, with a floor + clip to avoid blowing up directions
            # whose Fisher mass is effectively zero. Clipping is one-sided:
            # we cap the whitening *amplification*, never the de-amplification
            # of high-curvature directions.
            if self.whiten:
                lam = top_eigs.clamp_min(self.whiten_floor)
                w = lam.rsqrt()
                # Geometric-mean normalisation keeps the overall update scale
                # comparable to the non-whitened projector; otherwise the
                # natural-gradient norm can drift across layers/steps.
                w = w / w.log().mean().exp().clamp_min(self.whiten_floor)
                w = w.clamp(max=self.whiten_clip)
                self.whiten_diag = w.to(device=grad.device, dtype=grad.dtype)
            else:
                self.whiten_diag = None

            prev_fisher_diag = torch.diagonal(C).clone() / max(self.fisher_steps, 1)
            leverage = top.pow(2).sum(dim=1)                 # [c]
            selected_idx = torch.topk(
                leverage, min(r, leverage.numel())
            ).indices.detach().cpu().tolist()
            self.last_active_rank = int(r)

        # Refresh candidate pool and reset Fisher accumulator: the EMA state
        # only makes sense in the basis it was built in, so we drop it when
        # rotating to the new pool.
        self.U_candidates = U.float().cpu()
        self.S_candidates = S.float().cpu()
        self.fisher_cov = None
        self.fisher_steps = 0

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
    # Projection / reconstruction with symmetric Λ_r^{-1/2}
    # ------------------------------------------------------------
    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        z = self.P.T @ (grad.T if self.transpose else grad)
        if self.whiten_diag is not None:
            z = z * self.whiten_diag.view(-1, 1).to(device=z.device, dtype=z.dtype)
        return z

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        z = low_rank_grad
        if self.whiten_diag is not None:
            z = z * self.whiten_diag.view(-1, 1).to(device=z.device, dtype=z.dtype)
        out = self.P @ z
        return self.scale_factor * (out.T if self.transpose else out)


class StochasticWhitenedBlockFisherGaLoreProjector(WhitenedBlockFisherGaLoreProjector):
    """
    Stochastic counterpart of :class:`WhitenedBlockFisherGaLoreProjector`.

    The whitening pipeline (centring + EMA + Λ_r^{-1/2}) is unchanged; only
    the *selection* of ``r`` eigen-directions inside the candidate pool is
    randomised, mirroring :class:`StochasticProjector` but with eigenvalues
    of the block Fisher as the score instead of singular values.

    The strongest ``topk_deterministic`` eigen-directions are kept as anchors;
    the remaining slots are sampled without replacement from a temperature-
    softmax over the eigenvalue spectrum. A noise/energy floor mirrors
    :meth:`StochasticProjector._candidate_mask` and prevents picking pure
    noise eigenvectors (those with effectively zero Fisher mass).
    """

    PROJECTOR_NAME = "stochastic_whitened_block_fisher_galore"

    def __init__(
        self,
        rank: int = _DEFAULT_RANK,
        candidate_rank: int = _DEFAULT_CANDIDATE_RANK,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
        eigen_eps: float = _DEFAULT_EIGEN_EPS,
        fisher_ema: float = _DEFAULT_FISHER_EMA,
        whiten_floor: float = _DEFAULT_WHITEN_FLOOR,
        whiten_clip: float = _DEFAULT_WHITEN_CLIP,
        center: bool = True,
        whiten: bool = True,
        temperature: float = _DEFAULT_TEMPERATURE,
        min_temperature: float = _DEFAULT_MIN_TEMPERATURE,
        max_temperature: float = _DEFAULT_MAX_TEMPERATURE,
        topk_deterministic: int | None = None,
        noise_sigma_ratio: float = _DEFAULT_NOISE_SIGMA_RATIO,
        min_energy_ratio: float = _DEFAULT_MIN_ENERGY_RATIO,
    ):
        super().__init__(
            rank=rank,
            candidate_rank=candidate_rank,
            scale_factor=scale_factor,
            eigen_eps=eigen_eps,
            fisher_ema=fisher_ema,
            whiten_floor=whiten_floor,
            whiten_clip=whiten_clip,
            center=center,
            whiten=whiten,
        )
        self.base_temperature = max(temperature, 1e-6)
        self.min_temperature = max(min_temperature, 1e-6)
        self.max_temperature = max(max_temperature, self.min_temperature)
        if topk_deterministic is None:
            topk_deterministic = max(0, rank - _DEFAULT_TOPK_DETERMINISTIC_OFFSET)
        self.topk_deterministic = max(0, min(int(topk_deterministic), rank))
        self.noise_sigma_ratio = noise_sigma_ratio
        self.min_energy_ratio = min_energy_ratio
        self.last_temperature = self.base_temperature

    @torch.no_grad()
    def _dynamic_temperature(self, weights: torch.Tensor) -> float:
        if weights.numel() <= 1:
            return self.min_temperature
        mean = weights.mean().clamp_min(1e-12)
        spectrum_cv = weights.std(unbiased=False) / mean
        temperature = self.base_temperature / (spectrum_cv.item() + 1e-6)
        return float(
            max(self.min_temperature, min(self.max_temperature, temperature))
        )

    @torch.no_grad()
    def _candidate_mask(self, eigs_desc: torch.Tensor) -> torch.Tensor:
        # Eigenvalues can be slightly negative due to numerical noise / EMA; we
        # rectify with clamp_min(0) before applying noise / energy floors.
        e = eigs_desc.clamp_min(0.0)
        median = e.median()
        sigma_floor = median * self.noise_sigma_ratio
        total = e.sum().clamp_min(1e-12)
        energy_ratio = e / total
        mask = (e >= sigma_floor) & (energy_ratio >= self.min_energy_ratio)
        if not mask.any():
            mask = torch.ones_like(e, dtype=torch.bool)
        return mask

    @torch.no_grad()
    def _select_indices_from_eigs(
        self, eigs_desc: torch.Tensor, r: int
    ) -> torch.Tensor:
        if r <= 0:
            return torch.empty(0, dtype=torch.long, device=eigs_desc.device)

        e = eigs_desc.clamp_min(0.0)
        mask = self._candidate_mask(e)
        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if candidate_indices.numel() < r:
            # Not enough "clean" candidates -> fall back to the top-r block.
            candidate_indices = torch.arange(
                min(r, e.numel()), device=e.device
            )

        deterministic_count = min(self.topk_deterministic, r)
        selected = []
        if deterministic_count:
            # Indices are already sorted desc -> first ``deterministic_count``
            # candidates are the anchors with the largest eigenvalues.
            selected.append(candidate_indices[:deterministic_count])

        remaining = r - deterministic_count
        if remaining > 0:
            stochastic_pool = candidate_indices[deterministic_count:]
            if stochastic_pool.numel() == 0:
                stochastic_pool = candidate_indices[:deterministic_count]

            weights = e[stochastic_pool]
            temperature = self._dynamic_temperature(weights)
            logits = weights / temperature
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)

            replacement = stochastic_pool.numel() < remaining
            sampled_offsets = torch.multinomial(
                probs, num_samples=remaining, replacement=replacement
            )
            selected.append(stochastic_pool[sampled_offsets])
            self.last_temperature = temperature
        else:
            self.last_temperature = self._dynamic_temperature(e[candidate_indices])

        return torch.cat(selected) if selected else candidate_indices[:0]


class AdaptiveWhitenedBlockFisherGaLoreProjector(StochasticWhitenedBlockFisherGaLoreProjector):
    """
    Adaptive-rank stochastic whitened block-Fisher projector.

    Mirrors :class:`AdaptiveStochasticProjector`: ``rank`` becomes an *upper*
    bound and the actual rank is chosen per ``update_basis`` from the block
    Fisher eigenvalue spectrum, combining

      * an energy threshold (rank that captures ``energy_threshold`` of the
        Fisher mass), and
      * a normalised effective rank (``exp(H) * entropy_scale`` where ``H``
        is the Shannon entropy of the eigenvalue distribution).

    The selected rank is clipped to ``[min_rank, rank]`` and then passed to
    the anchored stochastic selector inherited from the parent class. The
    whitening cache is sized to match the *chosen* rank, so cheap windows
    automatically use a smaller projector and a thicker EMA Fisher gets
    properly inverted on its dominant subspace.
    """

    PROJECTOR_NAME = "adaptive_whitened_block_fisher_galore"

    def __init__(
        self,
        rank: int = _DEFAULT_RANK,
        candidate_rank: int = _DEFAULT_CANDIDATE_RANK,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
        eigen_eps: float = _DEFAULT_EIGEN_EPS,
        fisher_ema: float = _DEFAULT_FISHER_EMA,
        whiten_floor: float = _DEFAULT_WHITEN_FLOOR,
        whiten_clip: float = _DEFAULT_WHITEN_CLIP,
        center: bool = True,
        whiten: bool = True,
        temperature: float = _DEFAULT_TEMPERATURE,
        min_temperature: float = _DEFAULT_MIN_TEMPERATURE,
        max_temperature: float = _DEFAULT_MAX_TEMPERATURE,
        topk_deterministic: int | None = None,
        noise_sigma_ratio: float = _DEFAULT_NOISE_SIGMA_RATIO,
        min_energy_ratio: float = _DEFAULT_MIN_ENERGY_RATIO,
        min_rank: int | None = None,
        energy_threshold: float = _DEFAULT_ENERGY_THRESHOLD,
        entropy_scale: float = _DEFAULT_ENTROPY_SCALE,
    ):
        super().__init__(
            rank=rank,
            candidate_rank=candidate_rank,
            scale_factor=scale_factor,
            eigen_eps=eigen_eps,
            fisher_ema=fisher_ema,
            whiten_floor=whiten_floor,
            whiten_clip=whiten_clip,
            center=center,
            whiten=whiten,
            temperature=temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            topk_deterministic=topk_deterministic,
            noise_sigma_ratio=noise_sigma_ratio,
            min_energy_ratio=min_energy_ratio,
        )
        self.max_rank = rank
        if min_rank is None:
            min_rank = max(1, rank - _DEFAULT_MIN_RANK_OFFSET)
        self.min_rank = max(1, min(int(min_rank), rank))
        self.energy_threshold = float(min(max(energy_threshold, 0.0), 1.0))
        self.entropy_scale = float(max(entropy_scale, 0.0))

    @torch.no_grad()
    def _choose_rank_from_eigs(self, eigs_desc: torch.Tensor, max_rank: int) -> int:
        e = eigs_desc.clamp_min(0.0)
        total = e.sum().clamp_min(1e-12)
        cumulative = torch.cumsum(e, dim=0) / total
        # +1 because searchsorted returns the first index where threshold is
        # reached; that index is already the rank in 1-based terms.
        energy_rank = int(
            torch.searchsorted(cumulative, self.energy_threshold).item()
        ) + 1

        probs = e / total
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        effective_rank = int(
            torch.ceil(torch.exp(entropy) * self.entropy_scale).item()
        )

        selected = max(self.min_rank, min(self.max_rank, max(energy_rank, effective_rank)))
        return max(1, min(selected, max_rank))

    @torch.no_grad()
    def _select_indices_from_eigs(
        self, eigs_desc: torch.Tensor, r: int
    ) -> torch.Tensor:
        # Temporarily clip ``topk_deterministic`` to the chosen rank so the
        # anchor count never exceeds the active basis.
        original_topk = self.topk_deterministic
        self.topk_deterministic = min(original_topk, r)
        try:
            return super()._select_indices_from_eigs(eigs_desc, r)
        finally:
            self.topk_deterministic = original_topk
