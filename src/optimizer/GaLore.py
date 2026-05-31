import torch
from diagnostics import log_spectrum_diagnostics
from SVD import get_svd

class GaLoreProjector:
    def __init__(self, rank: int = 8, scale_factor: float = 0.25):
        self.rank = rank
        self.P = None
        self.scale_factor = scale_factor
        self.cfg = {"type": "classic"}
        self.transpose = None
        self.P_prev = None

    @torch.no_grad()
    def update_basis(
        self,
        grad: torch.Tensor,
        param_name: str | None = None,
        step: int | None = None,
        experiment=None,
    ):
        if self.transpose is None:
            self.transpose = grad.shape[0] > grad.shape[1]

        g = grad.T if self.transpose else grad
        U, S, _ = get_svd(g, **self.cfg)
        r = min(self.rank, U.shape[1])
        previous_basis = self.P
        self.P = U[:, :r].to(grad.dtype)
        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.rank,
            projector_name="galore",
            param_name=param_name,
            step=step,
            experiment=experiment,
        )
        self.P_prev = self.P.clone()


    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.P.T @ grad.T
        return self.P.T @ grad

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.scale_factor * (self.P @ low_rank_grad).T
        return self.scale_factor * (self.P @ low_rank_grad)