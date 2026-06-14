"""Standalone CPU sanity check for the HVP selection projectors.

Verifies (without CUDA / the full pipeline):
  * the GN curvature identity  s_i = u_iᵀ g v_i  is accumulated correctly,
  * the full lifecycle bootstrap -> accumulate -> update_basis -> project ->
    reconstruct produces finite tensors of the right shapes,
  * top-k / softmax / adaptive all return a valid rank-r orthonormal basis.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("./src/optimizer"))

import torch

from hvp_selection_projectors import (
    TopKHVPProjector,
    SoftmaxHVPProjector,
    AdaptiveHVPProjector,
)

torch.manual_seed(0)

M, N = 16, 32          # grad shape[0] < shape[1] -> transpose = False
RANK, CAND = 4, 8


def check_math():
    proj = TopKHVPProjector(rank=RANK, candidate_rank=CAND)
    g0 = torch.randn(M, N)
    proj.update_basis(g0)                       # bootstrap: fills U/V candidates

    U = proj.U_candidates                       # [M, c]
    V = proj.V_candidates                       # [c, N]
    c = U.shape[1]

    grads = [torch.randn(M, N) for _ in range(5)]
    for g in grads:
        proj.accumulate_fisher(g)

    # Reference curvature: Σ_t (u_iᵀ g_t v_i)² accumulated independently.
    ref = torch.zeros(c)
    for g in grads:
        s = U.T @ g.float() @ V.T               # [c, c]; diagonal is u_iᵀ g v_i
        ref += torch.diagonal(s).pow(2)

    err = (proj.hvp_curv - ref).abs().max().item()
    print(f"[math] max|accumulated - reference| = {err:.3e}  (steps={proj.hvp_steps})")
    assert err < 1e-4, "GN curvature accumulation mismatch"
    assert proj.hvp_steps == len(grads)


def check_lifecycle(cls, **kw):
    proj = cls(rank=RANK, candidate_rank=CAND, **kw)
    g = torch.randn(M, N)
    proj.update_basis(g)                        # bootstrap
    for _ in range(6):
        proj.accumulate_fisher(torch.randn(M, N))
    proj.update_basis(torch.randn(M, N))        # real selection path

    P = proj.P
    r = P.shape[1]
    low = proj.project(g)
    recon = proj.reconstruct(low)

    ortho = (P.T @ P - torch.eye(r)).abs().max().item()
    name = cls.__name__
    print(
        f"[{name}] P={tuple(P.shape)} low={tuple(low.shape)} "
        f"recon={tuple(recon.shape)} ‖PᵀP-I‖∞={ortho:.2e} "
        f"finite={torch.isfinite(recon).all().item()}"
    )
    assert 1 <= r <= RANK
    assert recon.shape == g.shape
    assert torch.isfinite(recon).all()
    assert ortho < 1e-4, "selected basis is not orthonormal"


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    check_math()
    check_lifecycle(TopKHVPProjector)
    check_lifecycle(SoftmaxHVPProjector, temperature=1.0)
    check_lifecycle(AdaptiveHVPProjector, temperature=1.0)
    print("ALL HVP PROJECTOR CHECKS PASSED")
