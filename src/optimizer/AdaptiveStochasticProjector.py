import torch

from diagnostics import log_spectrum_diagnostics
from StochasticProjector import StochasticProjector
from SVD import get_svd


class AdaptiveStochasticProjector(StochasticProjector):
    """Layer-wise adaptive-rank variant of the stochastic sampler.

    The rank is recomputed from the current gradient spectrum at every basis
    refresh. We combine an energy threshold with an entropy-based effective rank:
    sharp spectra receive smaller ranks, while flatter spectra keep more
    directions. The selected rank is then passed to the anchored stochastic
    sampler implemented by ``StochasticProjector``.
    """

    def __init__(
        self,
        rank: int = 8,
        min_rank: int | None = None,
        energy_threshold: float = 0.98,
        entropy_scale: float = 1.0,
        **kwargs,
    ):
        # Keep most directions greedy and reserve one slot for exploration.
        # This makes the method competitive with top-r projectors while still
        # testing whether non-top directions help.
        kwargs.setdefault("topk_deterministic", max(1, rank - 1))
        super().__init__(rank=rank, **kwargs)
        self.max_rank = rank
        if min_rank is None:
            min_rank = max(1, rank - 2)
        self.min_rank = max(1, min(min_rank, rank))
        self.energy_threshold = min(max(energy_threshold, 0.0), 1.0)
        self.entropy_scale = max(entropy_scale, 0.0)
        self.last_adaptive_rank = rank

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

        original_rank = self.rank
        original_topk = self.topk_deterministic
        self.rank = self._choose_rank(S)
        self.topk_deterministic = min(self.topk_deterministic, self.rank)
        new_P, diagnostics = self._sample_basis(U, S)
        self.rank = original_rank
        self.topk_deterministic = original_topk

        new_P = new_P.to(grad.dtype)
        previous_basis = self.P
        self.was_switched = self.P is not None and new_P.shape == self.P.shape
        if self.was_switched and self.P is not None:
            self.was_switched = not torch.allclose(new_P, self.P, atol=1e-4)
        else:
            self.was_switched = self.P is None

        self.P = new_P
        self.last_temperature = diagnostics["temperature"]
        self.last_candidate_count = diagnostics["candidate_count"]
        self.last_adaptive_rank = new_P.shape[1]

        if experiment is not None and param_name is not None and step is not None:
            experiment.log_metrics(
                {
                    f"adaptive_stochastic/{param_name}/rank": self.last_adaptive_rank,
                    f"adaptive_stochastic/{param_name}/temperature": self.last_temperature,
                    f"adaptive_stochastic/{param_name}/candidate_count": self.last_candidate_count,
                },
                step=step,
            )
        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.last_adaptive_rank,
            projector_name="adaptive_stochastic",
            param_name=param_name,
            step=step,
            experiment=experiment,
            extra={
                "adaptive_rank": self.last_adaptive_rank,
                "temperature": self.last_temperature,
                "candidate_count": self.last_candidate_count,
            },
        )
