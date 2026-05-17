import torch
from SVD import get_svd


# Expected W in RR^(m times n) and m <= n, maybe sometimes we need transpose before
class GaLoreProjector:
    def __init__(
        self, rank: int = 8, scale_factor: float = 1.0
    ):  # TODO: find real params
        self.rank = rank
        self.P = None
        self.scale_factor = scale_factor
        self.cfg = {
            "type": "classic",
        }
        self.transpose = None

    @torch.no_grad()
    def update_basis(self, grad: torch.Tensor):
        if self.transpose is None:
            if grad.shape[0] > grad.shape[1]:
                self.transpose = True
            else:
                self.transpose = False

        if self.transpose:
            U, _, _ = get_svd(grad.T, **self.cfg)
            r = min(self.rank, U.shape[1])
            self.P = U[:, :r].to(grad.dtype)
        else:
            U, _, _ = get_svd(grad, **self.cfg)
            r = min(self.rank, U.shape[1])
            self.P = U[:, :r].to(grad.dtype)

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
