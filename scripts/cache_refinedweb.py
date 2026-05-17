import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor # <-- НОВОЕ

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import RefinedWebCacheConfig, cache_refinedweb_subset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache a streaming RefinedWeb subset into JSONL shards.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="tiiuae/falcon-refinedweb")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--max-gb", type=float, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--shard-gb", type=float, default=0.5)
    parser.add_argument("--shuffle-buffer-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--token", default=None)
    parser.add_argument("--num-workers", type=int, default=8) # <-- НОВОЕ: количество параллельных процессов

    return parser.parse_args()
# <-- НОВОЕ: обертка для запуска в пуле процессов
def run_worker(config):
    return cache_refinedweb_subset(config)


def main():
    args = parse_args()
    max_bytes = None
    if args.max_gb is not None:
        max_bytes = int(args.max_gb * 1024**3)

    # Создаем конфиг для каждого параллельного процесса
    configs = []
    for i in range(args.num_workers):
        config = RefinedWebCacheConfig(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            split=args.split,
            text_column=args.text_column,
            max_bytes=max_bytes,
            max_documents=args.max_documents,
            shard_size_bytes=int(args.shard_gb * 1024**3),
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
            token=args.token,
            num_shards=args.num_workers, # <-- НОВОЕ
            shard_index=i                # <-- НОВОЕ
        )
        configs.append(config)
    # Запускаем параллельно
    print(f"Starting download with {args.num_workers} workers...")
    total_stats = {"documents": 0, "text_bytes": 0, "disk_bytes": 0}
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for stats in executor.map(run_worker, configs):
            total_stats["documents"] += stats.documents
            total_stats["text_bytes"] += stats.text_bytes
            total_stats["disk_bytes"] += stats.disk_bytes

    print("\nDownload finished. Total stats:")
    print(json.dumps(total_stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
