import torch
from diagnostics import log_spectrum_diagnostics
from SVD import get_svd
from StochasticLotus import StochasticLotus


class AdaptiveStochasticLotus(StochasticLotus):
    """Layer-wise adaptive-rank variant of StochasticLotus.

    Inherits Lotus's direction-drift refresh schedule (eta / gamma) and
    StochasticLotus's anchored stochastic basis sampling, and adds the
    same energy + entropy rank selection used by
    AdaptiveStochasticGaLore2Projector: sharp spectra receive smaller
    ranks, flat spectra keep more directions.
    """

    projector_name = "adaptive_stochastic_lotus"
    metrics_prefix = "adaptive_stochastic_lotus"

    def __init__(
        self,
        rank: int = 8,
        min_rank: int | None = None,
        energy_threshold: float = 0.98,
        entropy_scale: float = 1.0,
        q: int = 1,
        **kwargs,
    ):
        kwargs.setdefault("topk_deterministic", max(1, rank - 1))
        super().__init__(rank=rank, q=q, **kwargs)

        self.max_rank = rank
        if min_rank is None:
            min_rank = max(1, rank - 2)
        self.min_rank = max(1, min(min_rank, rank))
        self.energy_threshold = min(max(energy_threshold, 0.0), 1.0)
        self.entropy_scale = max(entropy_scale, 0.0)
        self.last_adaptive_rank = rank

    # ------------------------------------------------------------------
    # Rank selection  (identical to AdaptiveStochasticGaLore2Projector)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _choose_rank(self, S: torch.Tensor) -> int:
        sv = S.float()
        if sv.numel() == 0:
            return 0

        energy = sv.square()
        total_energy = energy.sum().clamp_min(1e-12)
        cumulative = torch.cumsum(energy, dim=0) / total_energy
        energy_rank = int(torch.searchsorted(cumulative, self.energy_threshold).item()) + 1

        probs = energy / total_energy
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        effective_rank = int(torch.ceil(torch.exp(entropy) * self.entropy_scale).item())

        selected_rank = max(self.min_rank, min(self.max_rank, max(energy_rank, effective_rank)))
        return min(selected_rank, sv.numel())

    # ------------------------------------------------------------------
    # Basis refresh  (adaptive rank + stochastic sampling + Lotus anchoring)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_basis(
        self,
        grad: torch.Tensor,
        param_name: str | None = None,
        step: int | None = None,
        experiment=None,
        **kwargs,
    ):
        if self.transpose is None:
            self.transpose = grad.shape[0] > grad.shape[1]

        g = grad.T if self.transpose else grad
        U, S, _ = get_svd(g, **self.cfg)

        # --- adaptive rank: temporarily patch self.rank / topk_deterministic ---
        original_rank = self.rank
        original_topk = self.topk_deterministic
        self.rank = self._choose_rank(S)
        self.topk_deterministic = min(self.topk_deterministic, self.rank)

        previous_basis = self.P
        new_P, diagnostics = self._sample_basis(U, S)

        self.rank = original_rank
        self.topk_deterministic = original_topk
        # ----------------------------------------------------------------------

        self.P = new_P.to(grad.dtype)
        self.last_temperature = diagnostics["temperature"]
        self.last_candidate_count = diagnostics["candidate_count"]
        self.last_adaptive_rank = new_P.shape[1]

        # --- Lotus: anchor direction at this refresh point ---
        g_init = self.P.T @ g
        self.d_init = g_init / (torch.norm(g_init) + self.eps)
        self.T = 1
        self.pending_update = False

        if experiment is not None and param_name is not None and step is not None:
            experiment.log_metrics(
                {
                    f"{self.metrics_prefix}/{param_name}/rank": self.last_adaptive_rank,
                    f"{self.metrics_prefix}/{param_name}/temperature": self.last_temperature,
                    f"{self.metrics_prefix}/{param_name}/candidate_count": self.last_candidate_count,
                },
                step=step,
            )

        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.last_adaptive_rank,
            projector_name=self.projector_name,
            param_name=param_name,
            step=step,
            experiment=experiment,
            extra={
                "adaptive_rank": self.last_adaptive_rank,
                "temperature": self.last_temperature,
                "candidate_count": self.last_candidate_count,
                "lotus_eta": self.eta,
                "lotus_gamma": self.gamma,
            },
        )