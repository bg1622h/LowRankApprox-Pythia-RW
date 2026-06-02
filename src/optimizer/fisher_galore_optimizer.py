from src.optimizer.MiniAdam import ProjectedMiniAdam


class FisherGaLoreOptimizer(ProjectedMiniAdam):
    """
    ``ProjectedMiniAdam`` variant that drives :class:`FisherGaLoreProjector`.

    Implementation note (важно учитывать как работают вызовы в оптимизаторе):
    ``MiniAdam._get_projector`` lazily deep-copies the per-group projector
    into the parameter ``state`` on first use, and every subsequent call to
    ``_low_rank_update`` operates on that **state** projector — not on the
    one stored in ``param_groups``. We therefore plug Fisher accumulation
    into ``_low_rank_update`` itself, which guarantees that:

      * ``accumulate_fisher`` is called every optimizer step,
      * ``update_basis`` and ``project`` see the same accumulated Fisher,
      * logging inside ``update_basis`` fires on the same projector instance.

    Routing accumulation through autograd hooks would target the *group*
    projector (the wrong instance) and would also break for transposed
    layers where ``grad_output`` does not match the candidate basis dim.
    """

    def _low_rank_update(self, param, grad, state, group, beta1, projector):
        # Accumulate Fisher BEFORE the parent may refresh the basis so the
        # selection inside ``update_basis`` sees the most recent samples.
        # The pipeline routes non-Fisher projectors (e.g. ``adaptive_stochastic``)
        # through ``FisherMiniAdam`` for direct ablation; those projectors
        # legitimately do not implement ``accumulate_fisher`` and we silently
        # fall back to the plain ProjectedMiniAdam path for them.
        accumulate = getattr(projector, "accumulate_fisher", None)
        if callable(accumulate):
            accumulate(grad)
        return super()._low_rank_update(param, grad, state, group, beta1, projector)


# Backwards-compatible alias used by pipeline.py.
FisherMiniAdam = FisherGaLoreOptimizer
