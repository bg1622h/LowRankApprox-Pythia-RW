import torch
from diagnostics import log_spectrum_diagnostics
from SVD import get_svd


# Expected W in RR^(m times n) and m <= n, maybe sometimes we need transpose before
class GaLore2Projector:
    def __init__(
        self, rank: int = 8, q: int = 1, scale_factor: float = 0.25
    ):  # TODO: find real params
        self.rank = rank
        self.P = None
        self.scale_factor = scale_factor
        self.cfg = {
            "type": "random",
            "params": {
                "rank": rank,
                "q": q,
            },
        }
        self.transpose = None

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
            if grad.shape[0] > grad.shape[1]:
                self.transpose = True
            else:
                self.transpose = False

        if self.transpose:
            U, S, _ = get_svd(grad.T, **self.cfg)
            r = min(self.rank, U.shape[1])
            previous_basis = self.P
            self.P = U[:, :r].to(grad.dtype)
        else:
            U, S, _ = get_svd(grad, **self.cfg)
            r = min(self.rank, U.shape[1])
            previous_basis = self.P
            self.P = U[:, :r].to(grad.dtype)
        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.rank,
            projector_name="galore2",
            param_name=param_name,
            step=step,
            experiment=experiment,
        )

    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.P.T @ grad.T
        else:
            return self.P.T @ grad

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.scale_factor * ((self.P @ low_rank_grad).T)
        else:
            return self.scale_factor * (self.P @ low_rank_grad)
