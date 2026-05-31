import torch
from diagnostics import log_spectrum_diagnostics
from SVD import get_svd


class StochasticProjector:
    """Stochastic singular-vector sampler for low-rank gradient projection.

    The method keeps the strongest singular direction as an anchor, filters out
    clear noise-floor directions, and samples the remaining basis vectors with a
    temperature adapted to the current spectrum shape. This preserves the usual
    GaLore-style greedy component while allowing controlled exploration of
    non-top directions.
    """

    def __init__(
        self,
        rank: int = 8,
        temperature: float = 1.0,
        min_temperature: float = 0.25,
        max_temperature: float = 4.0,
        noise_sigma_ratio: float = 0.01,
        min_energy_ratio: float = 1e-4,
        topk_deterministic: int = 1,
        scale_factor: float = 0.25,
        candidate_multiplier: int = 4,
    ):
        self.rank = rank
        self.base_temperature = max(temperature, 1e-6)
        self.min_temperature = max(min_temperature, 1e-6)
        self.max_temperature = max(max_temperature, self.min_temperature)
        self.noise_sigma_ratio = noise_sigma_ratio
        self.min_energy_ratio = min_energy_ratio
        self.topk_deterministic = max(0, min(topk_deterministic, rank))
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
        self.last_temperature = self.base_temperature
        self.last_candidate_count = 0

    @torch.no_grad()
    def _dynamic_temperature(self, sv: torch.Tensor) -> float:
        if sv.numel() <= 1:
            return self.min_temperature

        mean = sv.mean().clamp_min(1e-12)
        spectrum_cv = sv.std(unbiased=False) / mean
        temperature = self.base_temperature / (spectrum_cv.item() + 1e-6)
        return float(max(self.min_temperature, min(self.max_temperature, temperature)))

    @torch.no_grad()
    def _candidate_mask(self, sv: torch.Tensor) -> torch.Tensor:
        median = sv.median()
        sigma_floor = median * self.noise_sigma_ratio
        energy = sv.square()
        energy_ratio = energy / energy.sum().clamp_min(1e-12)
        mask = (sv >= sigma_floor) & (energy_ratio >= self.min_energy_ratio)
        if not mask.any():
            mask = torch.ones_like(sv, dtype=torch.bool)
        return mask

    @torch.no_grad()
    def _sample_basis(self, U: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, dict]:
        sv = S.float()
        if sv.numel() == 0:
            return U[:, :0], {
                "temperature": self.base_temperature,
                "candidate_count": 0,
                "deterministic_count": 0,
            }

        mask = self._candidate_mask(sv)
        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if candidate_indices.numel() < self.rank:
            candidate_indices = torch.arange(
                min(self.rank, sv.numel()),
                device=sv.device,
            )

        r = min(self.rank, candidate_indices.numel())
        deterministic_count = min(self.topk_deterministic, r)
        selected = []
        if deterministic_count:
            selected.append(candidate_indices[:deterministic_count])

        remaining_slots = r - deterministic_count
        if remaining_slots:
            stochastic_pool = candidate_indices[deterministic_count:]
            if stochastic_pool.numel() == 0:
                stochastic_pool = candidate_indices[:deterministic_count]

            weights = sv[stochastic_pool]
            temperature = self._dynamic_temperature(weights)
            logits = weights / temperature
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)

            replacement = stochastic_pool.numel() < remaining_slots
            sampled_offsets = torch.multinomial(
                probs,
                num_samples=remaining_slots,
                replacement=replacement,
            )
            selected.append(stochastic_pool[sampled_offsets])
        else:
            temperature = self._dynamic_temperature(sv[candidate_indices])

        idx = torch.cat(selected) if selected else candidate_indices[:0]
        diagnostics = {
            "temperature": temperature,
            "candidate_count": int(candidate_indices.numel()),
            "deterministic_count": int(deterministic_count),
        }
        return U[:, idx].to(U.dtype), diagnostics

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
        new_P, diagnostics = self._sample_basis(U, S)
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

        if experiment is not None and param_name is not None and step is not None:
            experiment.log_metrics(
                {
                    f"stochastic_sampler/{param_name}/temperature": self.last_temperature,
                    f"stochastic_sampler/{param_name}/candidate_count": self.last_candidate_count,
                    f"stochastic_sampler/{param_name}/deterministic_count": diagnostics[
                        "deterministic_count"
                    ],
                },
                step=step,
            )
        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.rank,
            projector_name="stochastic",
            param_name=param_name,
            step=step,
            experiment=experiment,
            extra={
                "temperature": self.last_temperature,
                "candidate_count": self.last_candidate_count,
                "deterministic_count": diagnostics["deterministic_count"],
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
