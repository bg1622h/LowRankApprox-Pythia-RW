import torch

from diagnostics import log_fisher_diagnostics
from SVD import get_svd


# Default hyper-parameters shared by the HVP selection family.
_DEFAULT_RANK = 128
_DEFAULT_CANDIDATE_RANK = 64
_DEFAULT_SCALE_FACTOR = 0.25

# Softmax variant only.
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_LOGIT_EPS = 1e-12

# Adaptive variant only (mirrors AdaptiveStochasticProjector).
_DEFAULT_MIN_TEMPERATURE = 0.25
_DEFAULT_MAX_TEMPERATURE = 4.0
_DEFAULT_TOPK_DETERMINISTIC_OFFSET = 1    # rank - this -> anchored greedy slots.
_DEFAULT_NOISE_SIGMA_RATIO = 0.01
_DEFAULT_MIN_ENERGY_RATIO = 1e-4
_DEFAULT_ENERGY_THRESHOLD = 0.98
_DEFAULT_ENTROPY_SCALE = 1.0
_DEFAULT_MIN_RANK_OFFSET = 2              # rank - this -> floor for adaptive rank.


class _WindowHVPProjector:
    r"""
    Shared base for projectors that build a rank-``r`` basis by *selecting*
    columns from a pool of ``candidate_rank`` SVD directions, scored by the
    **curvature of the loss along each singular component** estimated through a
    Gauss-Newton / Fisher approximation of a Hessian-vector product (HVP).

    Why a curvature score instead of gradient energy
    -------------------------------------------------
    The Fisher selection projectors score a candidate ``u_i`` by the windowed
    gradient energy ``E_t[(u_iᵀ g_t)²]`` — a *one-sided* statistic that only
    looks at the left singular vector. The Hessian, however, acts on directions
    in **parameter space**. For a single SVD component the matching parameter
    direction is the rank-one matrix

        P_i = u_i v_iᵀ ,   p_i = vec(P_i),

    and the curvature of the loss along it is the Rayleigh quotient ``p_iᵀ H p_i``
    (the Pearlmutter HVP contracted back onto ``p_i``). Computing a true HVP
    needs a second backward through the autograd graph, which this optimizer
    deliberately does not expose to projectors (``MiniAdam.step`` runs under
    ``torch.no_grad`` and hands the projector a detached ``grad``).

    The Gauss-Newton shortcut (no second backward)
    ----------------------------------------------
    For the cross-entropy / NLL losses used here the Gauss-Newton matrix equals
    the Fisher and is an exact PSD surrogate for the Hessian:

        H ≈ G = E_s[ vec(∇_W ℓ_s) vec(∇_W ℓ_s)ᵀ ].

    Plugging the rank-one probe ``p_i = vec(u_i v_iᵀ)`` and using the identity
    ``vec(u v ᵀ)ᵀ vec(g) = uᵀ g v`` collapses the curvature to a scalar:

        p_iᵀ H p_i  ≈  E_s[ (u_iᵀ ∇_W ℓ_s v_i)² ].

    So the per-component curvature can be Monte-Carlo estimated **from the
    gradients we already see**, by accumulating ``(u_iᵀ g_t v_i)²`` over the
    update window — one extra elementwise reduction on top of the ``Uᵀg`` that
    the Fisher projectors compute anyway, and strictly cheaper than the block
    projector's ``c×c`` covariance. This is a *two-sided* score (it uses the
    matched pair ``(u_i, v_i)``), which is exactly the curvature information the
    one-sided Fisher diagonal discards.

    Caveat (honest)
    ---------------
    ``g_t`` is the minibatch-averaged gradient, so this is an *empirical* Fisher
    curvature (expectation over minibatches, not per-sample), the same proxy the
    Fisher projectors already rely on. It captures positive GN curvature only;
    genuine negative curvature would require the real Hessian.

    Lifecycle, ``project`` / ``reconstruct`` and the diagnostics call match the
    Fisher projectors, so these are drop-in projectors for ``FisherMiniAdam``
    (the per-step ``accumulate_fisher`` hook is what routes a projector through
    that optimizer).
    """

    # Overridden by concrete subclasses for log grouping.
    PROJECTOR_NAME = "window_hvp"
    SVD_CONFIG = {"type": "classic"}

    def __init__(
        self,
        rank: int = _DEFAULT_RANK,
        candidate_rank: int = _DEFAULT_CANDIDATE_RANK,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
    ):
        self.rank = rank
        self.candidate_rank = candidate_rank
        self.scale_factor = scale_factor

        self.transpose: bool | None = None
        self.P: torch.Tensor | None = None
        self.P_prev: torch.Tensor | None = None

        # Candidate pool: left vectors U [m, c] and right vectors Vᵀ [c, n].
        # We need BOTH because the curvature probe is the matched rank-one pair
        # u_i v_iᵀ, unlike the Fisher projectors that only keep U.
        self.U_candidates: torch.Tensor | None = None
        self.V_candidates: torch.Tensor | None = None
        self.S_candidates: torch.Tensor | None = None

        # Windowed GN curvature per candidate: c_i = Σ_t (u_iᵀ g_t v_i)², [c].
        self.hvp_curv: torch.Tensor | None = None
        self.hvp_steps: int = 0

    # ------------------------------------------------------------
    # Selection hooks (implemented / overridden by subclasses)
    # ------------------------------------------------------------
    def _choose_rank(self, curvature: torch.Tensor, c: int) -> int:
        """Default: keep the configured rank, clipped to what is available."""
        return max(1, min(self.rank, c))

    def _select_indices(self, curvature: torch.Tensor, r: int) -> torch.Tensor:
        """
        Map the per-candidate HVP curvature (CPU float tensor, length ``c``) to
        ``r`` distinct candidate indices (1D LongTensor in ``[0, c)``).
        """
        raise NotImplementedError

    # ------------------------------------------------------------
    # Curvature accumulation (called by the optimizer every step)
    #
    # Named ``accumulate_fisher`` so that ``FisherGaLoreOptimizer`` picks this
    # projector up via its ``getattr(projector, "accumulate_fisher")`` hook;
    # semantically it accumulates the GN/HVP curvature described above.
    # ------------------------------------------------------------
    @torch.no_grad()
    def accumulate_fisher(self, grad: torch.Tensor) -> None:
        # Before the very first update_basis we have no candidate pool yet.
        if self.U_candidates is None or grad is None or grad.dim() != 2:
            return

        g = grad.T if self.transpose else grad
        if (
            g.shape[0] != self.U_candidates.shape[0]
            or g.shape[1] != self.V_candidates.shape[1]
        ):
            # Defensive: skip silently on unexpected shape changes.
            return

        g_f = g.float()
        U = self.U_candidates.to(device=g_f.device, dtype=g_f.dtype)   # [m, c]
        V = self.V_candidates.to(device=g_f.device, dtype=g_f.dtype)   # [c, n]

        # s_i = u_iᵀ g v_i for every candidate at once:
        #   (Uᵀ g)        -> [c, n], row i is u_iᵀ g
        #   row-dot with v_i (= V row i) -> [c]
        bilinear = ((U.T @ g_f) * V).sum(dim=1)                        # [c]
        curv = bilinear.pow(2).detach().cpu()                          # [c]

        if self.hvp_curv is None:
            self.hvp_curv = curv
        else:
            self.hvp_curv = self.hvp_curv + curv
        self.hvp_steps += 1

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
        U, S, Vh = get_svd(g, **self.SVD_CONFIG)

        c = min(self.candidate_rank, U.shape[1])
        U = U[:, :c]
        S = S[:c]
        Vh = Vh[:c, :]

        bootstrap = self.U_candidates is None or self.hvp_curv is None

        # Snapshot of the OLD candidate spectrum used for selection; needed for
        # diagnostics before we overwrite the pool below.
        prev_S = self.S_candidates
        prev_curv = None
        hvp_steps_logged = int(self.hvp_steps)

        if bootstrap:
            # First call (or no curvature yet): seed with top-rank from SVD.
            r = min(self.rank, c)
            self.P = U[:, :r].to(grad.dtype)
            selected_idx = list(range(r))
        else:
            # Mean windowed curvature per candidate (CPU float, length c).
            curvature = self.hvp_curv / max(self.hvp_steps, 1)
            prev_curv = curvature
            r = self._choose_rank(curvature, c)
            idx = self._select_indices(curvature, r)          # LongTensor, CPU
            U_prev = self.U_candidates.to(device=grad.device, dtype=grad.dtype)
            self.P = U_prev[:, idx.to(U_prev.device)].to(grad.dtype)
            selected_idx = idx.detach().cpu().tolist()

        # Refresh candidate pool and reset the curvature accumulator.
        self.U_candidates = U.float().cpu()
        self.V_candidates = Vh.float().cpu()
        self.S_candidates = S.float().cpu()
        self.hvp_curv = None
        self.hvp_steps = 0

        # The dedicated chart only makes sense once we have data. We reuse the
        # Fisher diagnostics: the per-candidate "fisher" series is the HVP
        # curvature, plotted against the SVD energy just like the Fisher ones.
        if prev_curv is not None and prev_S is not None:
            log_fisher_diagnostics(
                fisher=prev_curv,
                singular_values=prev_S,
                selected_idx=selected_idx,
                rank=self.rank,
                fisher_steps=hvp_steps_logged,
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


class TopKHVPProjector(_WindowHVPProjector):
    """
    Selects the ``r`` candidate directions with the **largest** windowed GN/HVP
    curvature: ``idx = topk(p_iᵀ H p_i, r)``. Deterministic curvature-greedy
    baseline of the HVP selection family — the curvature analogue of
    :class:`TopKFisherGaLoreProjector` (which is gradient-energy-greedy).
    """

    PROJECTOR_NAME = "topk_hvp"

    @torch.no_grad()
    def _select_indices(self, curvature: torch.Tensor, r: int) -> torch.Tensor:
        return torch.topk(curvature, r).indices


class SoftmaxHVPProjector(_WindowHVPProjector):
    r"""
    Stochastic counterpart of :class:`TopKHVPProjector`: samples ``r`` distinct
    candidates **without replacement** from a categorical distribution whose
    logits are the windowed HVP curvatures (``k`` drawn from the full
    ``candidate_rank`` pool).

    The raw curvature spans many orders of magnitude and its scale drifts across
    layers and steps, so we normalise the logits by their mean (a scale-free
    transform) before applying a temperature:

        p_i = softmax( (curv_i / mean(curv)) / temperature )

    ``temperature = 1.0`` is neutral: lower sharpens toward the top-k pick,
    higher approaches uniform exploration of the candidate pool.
    """

    PROJECTOR_NAME = "softmax_hvp"

    def __init__(
        self,
        rank: int = _DEFAULT_RANK,
        candidate_rank: int = _DEFAULT_CANDIDATE_RANK,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
        temperature: float = _DEFAULT_TEMPERATURE,
        logit_eps: float = _DEFAULT_LOGIT_EPS,
    ):
        super().__init__(
            rank=rank,
            candidate_rank=candidate_rank,
            scale_factor=scale_factor,
        )
        self.temperature = temperature
        self.logit_eps = logit_eps

    @torch.no_grad()
    def _select_indices(self, curvature: torch.Tensor, r: int) -> torch.Tensor:
        # Curvature is PSD under the GN approximation but can carry tiny
        # negative values from float noise; clamp before forming logits.
        curv = curvature.clamp_min(0.0)
        logits = curv / curv.mean().clamp_min(self.logit_eps)
        probs = torch.softmax(logits / self.temperature, dim=0)
        return torch.multinomial(probs, r, replacement=False)


class AdaptiveHVPProjector(SoftmaxHVPProjector):
    r"""
    Adaptive-rank stochastic HVP projector, mirroring
    :class:`AdaptiveStochasticProjector` but scoring with GN/HVP curvature
    instead of singular values.

    Two additions over :class:`SoftmaxHVPProjector`:

      * **Adaptive rank** — ``rank`` becomes an *upper* bound; the active rank is
        chosen per window from the curvature spectrum, combining an energy
        threshold (rank capturing ``energy_threshold`` of the total curvature)
        and a normalised effective rank (``exp(H)`` of the curvature
        distribution). The result is clipped to ``[min_rank, rank]``.
      * **Anchored stochastic selection** — the strongest
        ``topk_deterministic`` curvature directions are kept as greedy anchors;
        the remaining slots are sampled without replacement from a
        spectrum-adaptive temperature softmax over a noise/energy-filtered
        candidate pool (same recipe as ``StochasticProjector``).
    """

    PROJECTOR_NAME = "adaptive_hvp"

    def __init__(
        self,
        rank: int = _DEFAULT_RANK,
        candidate_rank: int = _DEFAULT_CANDIDATE_RANK,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
        temperature: float = _DEFAULT_TEMPERATURE,
        logit_eps: float = _DEFAULT_LOGIT_EPS,
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
            temperature=temperature,
            logit_eps=logit_eps,
        )
        self.base_temperature = max(temperature, 1e-6)
        self.min_temperature = max(min_temperature, 1e-6)
        self.max_temperature = max(max_temperature, self.min_temperature)
        if topk_deterministic is None:
            topk_deterministic = max(0, rank - _DEFAULT_TOPK_DETERMINISTIC_OFFSET)
        self.topk_deterministic = max(0, min(int(topk_deterministic), rank))
        self.noise_sigma_ratio = noise_sigma_ratio
        self.min_energy_ratio = min_energy_ratio

        self.max_rank = rank
        if min_rank is None:
            min_rank = max(1, rank - _DEFAULT_MIN_RANK_OFFSET)
        self.min_rank = max(1, min(int(min_rank), rank))
        self.energy_threshold = float(min(max(energy_threshold, 0.0), 1.0))
        self.entropy_scale = float(max(entropy_scale, 0.0))

        # Diagnostics: rank / temperature actually used in the last window.
        self.last_active_rank: int = rank
        self.last_temperature: float = self.base_temperature

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
    def _candidate_mask(self, curv: torch.Tensor) -> torch.Tensor:
        c = curv.clamp_min(0.0)
        median = c.median()
        sigma_floor = median * self.noise_sigma_ratio
        total = c.sum().clamp_min(1e-12)
        energy_ratio = c / total
        mask = (c >= sigma_floor) & (energy_ratio >= self.min_energy_ratio)
        if not mask.any():
            mask = torch.ones_like(c, dtype=torch.bool)
        return mask

    @torch.no_grad()
    def _choose_rank(self, curvature: torch.Tensor, c: int) -> int:
        curv = curvature.clamp_min(0.0)
        total = curv.sum().clamp_min(1e-12)
        cumulative = torch.cumsum(curv, dim=0) / total
        energy_rank = int(
            torch.searchsorted(cumulative, self.energy_threshold).item()
        ) + 1

        probs = curv / total
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        effective_rank = int(
            torch.ceil(torch.exp(entropy) * self.entropy_scale).item()
        )

        selected = max(
            self.min_rank, min(self.max_rank, max(energy_rank, effective_rank))
        )
        self.last_active_rank = max(1, min(selected, c))
        return self.last_active_rank

    @torch.no_grad()
    def _select_indices(self, curvature: torch.Tensor, r: int) -> torch.Tensor:
        if r <= 0:
            return torch.empty(0, dtype=torch.long)

        curv = curvature.clamp_min(0.0)
        mask = self._candidate_mask(curv)
        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if candidate_indices.numel() < r:
            # Not enough "clean" candidates -> fall back to the top-r block by
            # curvature so we always return exactly r indices.
            candidate_indices = torch.topk(
                curv, min(r, curv.numel())
            ).indices.sort().values

        deterministic_count = min(self.topk_deterministic, r)
        # Within the surviving pool, anchors are the largest-curvature ones.
        pool_curv = curv[candidate_indices]
        order = torch.argsort(pool_curv, descending=True)
        candidate_indices = candidate_indices[order]

        selected = []
        if deterministic_count:
            selected.append(candidate_indices[:deterministic_count])

        remaining = r - deterministic_count
        if remaining > 0:
            stochastic_pool = candidate_indices[deterministic_count:]
            if stochastic_pool.numel() == 0:
                stochastic_pool = candidate_indices[:deterministic_count]

            weights = curv[stochastic_pool]
            temperature = self._dynamic_temperature(weights)
            logits = weights / temperature
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)

            replacement = stochastic_pool.numel() < remaining
            sampled = torch.multinomial(
                probs, num_samples=remaining, replacement=replacement
            )
            selected.append(stochastic_pool[sampled])
            self.last_temperature = temperature
        else:
            self.last_temperature = self._dynamic_temperature(
                curv[candidate_indices]
            )

        return torch.cat(selected) if selected else candidate_indices[:0]
