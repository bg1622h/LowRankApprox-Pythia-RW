from StochasticProjector import StochasticProjector
class StochasticGaLore2Projector(StochasticProjector):
    """Stochastic sampler paired with randomized SVD (GaLore2-style).
    ``randomized_svd`` with target rank ``r`` probes a ``2r``-dimensional
    subspace; we then stochastically sample ``r`` directions from that pool
    using the same anchored, spectrum-weighted scheme as ``StochasticProjector``.
    """
    projector_name = "stochastic_galore2"
    metrics_prefix = "stochastic_galore2_sampler"
    def __init__(self, rank: int = 8, q: int = 1, **kwargs):
        kwargs.pop("candidate_multiplier", None)
        super().__init__(rank=rank, **kwargs)
        self.cfg = {
            "type": "random",
            "params": {
                "rank": self.rank,
                "q": q,
            },
        }
        self.candidate_rank = self.rank