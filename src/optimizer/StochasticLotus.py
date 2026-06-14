import torch
from diagnostics import log_spectrum_diagnostics
from SVD import get_svd
from StochasticProjector import StochasticProjector


class StochasticLotus(StochasticProjector):
    """Stochastic-basis variant of Lotus.

    Combines Lotus's direction-drift convergence criterion (eta / gamma)
    for deciding *when* to refresh the basis with StochasticProjector's
    anchored, spectrum-weighted stochastic sampling for deciding *which*
    directions to keep.

    The basis is refreshed lazily: on the first call and whenever the
    projected gradient direction drifts by more than ``gamma`` per step
    over a window of ``eta`` steps (same schedule as vanilla Lotus).
    """

    projector_name = "stochastic_lotus"
    metrics_prefix = "stochastic_lotus"

    def __init__(
        self,
        rank: int = 8,
        q: int = 1,
        gamma: float = 0.05,
        eta: int = 200,
        scale_factor: float = 0.25,
        **kwargs,
    ):
        # candidate_multiplier is a StochasticProjector knob that
        # GaLore2-style callers don't use — strip it to avoid confusion.
        kwargs.pop("candidate_multiplier", None)
        super().__init__(rank=rank, **kwargs)

        # Override cfg so get_svd uses the randomized (GaLore2-style) backend.
        self.cfg = {
            "type": "random",
            "params": {
                "rank": rank,
                "q": q,
            },
        }
        self.candidate_rank = self.rank  # mirrors StochasticGaLore2Projector

        # Lotus-specific state
        self.scale_factor = scale_factor
        self.gamma = gamma
        self.eta = eta
        self.eps = 1e-8

        self.d_init = None          # normalised projected gradient at basis refresh
        self.T = None               # steps since last refresh
        self.pending_update = True  # trigger refresh on very first project() call
        self.was_switched = False   # visible to the optimizer

    # ------------------------------------------------------------------
    # Basis refresh  (called lazily from project())
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

        # --- stochastic selection (replaces plain U[:, :r] from vanilla Lotus) ---
        previous_basis = self.P
        new_P, diagnostics = self._sample_basis(U, S)
        self.P = new_P.to(grad.dtype)
        self.last_temperature = diagnostics["temperature"]
        self.last_candidate_count = diagnostics["candidate_count"]

        # --- Lotus: anchor the direction at this refresh point ---
        g_init = self.P.T @ g
        self.d_init = g_init / (torch.norm(g_init) + self.eps)
        self.T = 1
        self.pending_update = False

        if experiment is not None and param_name is not None and step is not None:
            experiment.log_metrics(
                {
                    f"{self.metrics_prefix}/{param_name}/temperature": self.last_temperature,
                    f"{self.metrics_prefix}/{param_name}/candidate_count": self.last_candidate_count,
                },
                step=step,
            )

        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.rank,
            projector_name=self.projector_name,
            param_name=param_name,
            step=step,
            experiment=experiment,
            extra={
                "lotus_eta": self.eta,
                "lotus_gamma": self.gamma,
                "temperature": self.last_temperature,
                "candidate_count": self.last_candidate_count,
            },
        )

    # ------------------------------------------------------------------
    # Project  (lazy refresh + direction-drift check)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        # Refresh basis when explicitly requested or on very first call.
        if self.pending_update or self.P is None:
            self.update_basis(grad)
            self.was_switched = True
        else:
            self.was_switched = False

        target_grad = grad.T if self.transpose else grad
        out = self.P.T @ target_grad

        # --- Lotus direction-drift criterion ---
        d_out = out / (torch.norm(out) + self.eps)
        self.T += 1

        if self.T % self.eta == 0:
            delta_d = d_out - self.d_init
            if torch.norm(delta_d) / self.T < self.gamma:
                self.pending_update = True

        return out

    # ------------------------------------------------------------------
    # Reconstruct  (identical to vanilla Lotus)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.scale_factor * (self.P @ low_rank_grad).T
        else:
            return self.scale_factor * (self.P @ low_rank_grad)