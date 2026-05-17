try:
    from . import refinedweb
except ImportError:
    import refinedweb

CachedDatasetStats = refinedweb.CachedDatasetStats
LocalJsonlTextDataset = refinedweb.LocalJsonlTextDataset
PackedCausalLMDataset = refinedweb.PackedCausalLMDataset
RefinedWebCacheConfig = refinedweb.RefinedWebCacheConfig
RefinedWebStreamingDataset = refinedweb.RefinedWebStreamingDataset
TokenizedCausalLMDataset = refinedweb.TokenizedCausalLMDataset
build_packed_attention_mask = refinedweb.build_packed_attention_mask
build_refinedweb_dataloader = refinedweb.build_refinedweb_dataloader
cache_refinedweb_subset = refinedweb.cache_refinedweb_subset
count_cached_dataset = refinedweb.count_cached_dataset

__all__ = [
    "CachedDatasetStats",
    "LocalJsonlTextDataset",
    "PackedCausalLMDataset",
    "RefinedWebCacheConfig",
    "RefinedWebStreamingDataset",
    "TokenizedCausalLMDataset",
    "build_packed_attention_mask",
    "build_refinedweb_dataloader",
    "cache_refinedweb_subset",
    "count_cached_dataset",
]
