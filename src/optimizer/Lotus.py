import torch
from diagnostics import log_spectrum_diagnostics
from SVD import get_svd


# Expected W in RR^(m times n) and m <= n, maybe sometimes we need transpose before
class Lotus:
    def __init__(
        self, rank: int = 8, q: int = 1, gamma: float = 0.05, eta: int = 200, scale_factor: float = 0.25
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
        self.pending_update = False
        self.eps = 1e-8
        self.eta = eta
        self.was_switched = False
        self.pending_update = True
        
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
            g_init = self.P.T @ grad.T
            self.d_init = g_init / (torch.norm(g_init) + self.eps)
            self.T = 1
            self.pending_update = False
        else:
            U, S, _ = get_svd(grad, **self.cfg)
            r = min(self.rank, U.shape[1])
            previous_basis = self.P
            self.P = U[:, :r].to(grad.dtype)
            g_init = self.P.T @ grad
            self.d_init = g_init / (torch.norm(g_init) + self.eps)
            self.T = 1
            self.pending_update = False
        log_spectrum_diagnostics(
            singular_values=S,
            basis=self.P,
            previous_basis=previous_basis,
            rank=self.rank,
            projector_name="lotus",
            param_name=param_name,
            step=step,
            experiment=experiment,
            extra={"lotus_eta": self.eta, "lotus_gamma": self.gamma},
        )

    # @torch.no_grad()
    # def project(self, grad: torch.Tensor) -> torch.Tensor:
    #     if self.pending_update:
    #         self.update_basis(grad)

    #     if self.transpose:
    #         out = self.P.T @ grad.T
    #     else:
    #         out = self.P.T @ grad

    #     d_out = out / (torch.norm(out) + self.eps)
    #     self.T += 1
    #     if self.T % self.eta == 0:
    #         delta_d = d_out - self.d_init
    #         if torch.norm(delta_d) < self.gamma * self.T:
    #             self.pending_update = True
    #     return out
    @torch.no_grad()
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        # 1. Если на прошлом шаге решили сменить базис — меняем его СЕЙЧАС
        # Также это сработает при самом первом вызове, если pending_update = True
        if self.pending_update or self.P is None:
            self.update_basis(grad)
            self.pending_update = False
            self.was_switched = True # Флаг для внешнего мира (оптимизатора)
        else:
            self.was_switched = False

        target_grad = grad.T if self.transpose else grad
        out = self.P.T @ target_grad

        d_out = out / (torch.norm(out) + self.eps)
        self.T += 1

        if self.T % self.eta == 0:
            delta_d = d_out - self.d_init
            if torch.norm(delta_d) / self.T < self.gamma:
                self.pending_update = True
        
        return out

    @torch.no_grad()
    def reconstruct(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return self.scale_factor * (self.P @ low_rank_grad).T
        else:
            return self.scale_factor * (self.P @ low_rank_grad)
