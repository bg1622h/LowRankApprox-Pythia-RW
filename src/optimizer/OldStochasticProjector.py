import torch

from diagnostics import log_spectrum_diagnostics
from SVD import get_svd


class OldStochasticProjector:
    """Original stochastic sampler kept for ablation comparisons.

    This is the first simple version: compute an SVD candidate basis, filter
    tiny singular values by a median-scaled noise floor, and sample rank-r
    directions from a softmax over singular values. It has no greedy anchor,
    no adaptive rank, and no dynamic temperature.
    """

    def __init__(
        self,
        rank: int = 8,
        temperature: float = 1.0,
        noise_sigma_ratio: float = 0.01,
        scale_factor: float = 0.25,
        candidate_multiplier: int = 4,
    ):
        self.rank = rank
        self.temperature = max(temperature, 1e-6)
        self.noise_sigma_ratio = noise_sigma_ratio
        self.scale_factor = scale_factor
        self.candidate_rank = max(rank, rank * candidate_multiplier)
        self.P = None
        self.transpose = None
        self.cfg = {
            "type": "random",
            "params": {
                "rank": self.candidate_rank,
                "q": 1,
            },
        }
        self.was_switched = False

    @torch.no_grad()
    def _sample_basis(self, U: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        sv = S.float()
        if sv.numel() == 0:
            return U[:, :0]

        median = sv.median()
        noise_floor = median * self.noise_sigma_ratio
        mask = sv >= noise_floor
        if not mask.any():
            mask = torch.ones_like(sv, dtype=torch.bool)

        candidates = U[:, mask]
        weights = sv[mask]
        logits = weights / self.temperature
        logits = logits - logits.max()
        probs = torch.softmax(logits, dim=0)

        r = min(self.rank, candidates.shape[1])
        if r == 0:
            return U[:, :0]

        idx = torch.multinomial(probs, num_samples=r, replacement=False)
        return candidates[:, idx].to(U.dtype)

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
        new_P = self._sample_basis(U, S).to(grad.dtype)
        previous_basis = self.P
        self.was_switched = self.P is not None and new_P.shape == self.P.shape
        if self.was_switched and self.P is not None:
            self.was_switched = not torch.allclose(new_P, self.P, atol=1e-4)
        else:
            self.was_switched = self.P is None
        self.P = new_P
        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.rank,
            projector_name="stochastic_old",
            param_name=param_name,
            step=step,
            experiment=experiment,
            extra={
                "temperature": self.temperature,
                "noise_sigma_ratio": self.noise_sigma_ratio,
            },
        )

    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self.P is None:
            raise RuntimeError("Call update_basis before project")
        if self.transpose:
            return self.P.T @ grad.T
        return self.P.T @ grad

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.scale_factor * (self.P @ low_rank_grad).T
        return self.scale_factor * (self.P @ low_rank_grad)
