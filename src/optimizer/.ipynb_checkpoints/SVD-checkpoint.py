import torch


@torch.no_grad()
def randomized_svd(G: torch.Tensor, rank: int, q: int = 1):
    """
    Randomized SVD from Halko et al. 2011: https://arxiv.org/pdf/0909.4061
    "*" in paper - conjugate matrix (.T)
    """

    m, n = G.shape
    Omega = torch.randn(n, 2 * rank, device=G.device, dtype=G.dtype)
    Y = G @ Omega
    for _ in range(q):
        Y = G.T @ Y
        Y = G @ Y
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ G
    U_tilde, S, V = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U, S, V


@torch.no_grad()
def get_svd(G: torch.Tensor, **kwargs):
    #EPS = 1e-6
    #G_reg = G.float() + EPS * torch.eye(G.size(-1), device=G.device) #not converge without this
    if kwargs.get("type", None) == "classic":
        return torch.linalg.svd(G.float(), full_matrices=False, driver="gesvd")
    elif kwargs.get("type", None) == "random":
        return randomized_svd(
            G.float(), kwargs["params"].get("rank", None), kwargs["params"].get("q", None)
        )
    assert False
