import torch
from SVD import get_svd


# Expected W in RR^(m times n) and m <= n, maybe sometimes we need transpose before
class Lotus:
    def __init__(
        self, rank: int = 8, q: int = 1, gamma: float = 0.05, eta: int = 100, scale_factor: float = 0.5
    ):  # TODO: find real params
        self.rank = rank
        self.P = None
        self.scale_factor = scale_factor
        self.gamma = gamma
        self.cfg = {
            "type": "random",
            "params": {
                "rank": rank,
                "q": q,
            },
        }
        self.transpose = None
        self.d_init = None
        self.T = None
        self.eps = 1e-8
        self.eta = eta

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
            g_init = self.P.T @ grad.T
            self.d_init = g_init / (torch.norm(g_init) + self.eps)
            self.T = 1
        else:
            U, _, _ = get_svd(grad, **self.cfg)
            r = min(self.rank, U.shape[1])
            self.P = U[:, :r].to(grad.dtype)
            g_init = self.P.T @ grad
            self.d_init = g_init / (torch.norm(g_init) + self.eps)
            self.T = 1

    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            out = self.P.T @ grad.T
            d_out = out / (torch.norm(out) + self.eps)
            self.T += 1
            if self.T % self.eta == 0:
                delta_d = d_out - self.d_init
                if torch.norm(delta_d) < self.gamma * self.T:
                    self.update_basis(grad)
        else:
            out = self.P.T @ grad
            d_out = out / (torch.norm(out) + self.eps)
            self.T += 1
            if self.T % self.eta == 0:
                delta_d = d_out - self.d_init
                if torch.norm(delta_d) < self.gamma * self.T:
                    self.update_basis(grad)
        return out

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.scale_factor * (self.P @ low_rank_grad).T
        else:
            return self.scale_factor * (self.P @ low_rank_grad)
