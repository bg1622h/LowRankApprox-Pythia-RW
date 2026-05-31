import argparse
import csv
import itertools
import logging
import os
import sys
import time
from types import SimpleNamespace

logging.basicConfig(
    filename="app.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)


def get_tensor_memory(model, optimizer=None, component="model"):
    mem_bytes = 0
    if component == "model":
        mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    elif component == "gradients":
        mem_bytes = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters()
            if p.grad is not None
        )
    elif component == "optimizer" and optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    mem_bytes += v.numel() * v.element_size()
    return mem_bytes / (1024 ** 3)


def setup_cuda_toolkit_env():
    if os.name != "posix":
        return
    cuda_home = "/usr/local/cuda-13.2"
    if not os.path.isdir(cuda_home):
        return
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("TRITON_PTXAS_PATH", f"{cuda_home}/bin/ptxas")
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    os.environ["CPATH"] = f"{cuda_home}/targets/x86_64-linux/include"
    os.environ["LIBRARY_PATH"] = (
        f"{cuda_home}/targets/x86_64-linux/lib/stubs:"
        + os.environ.get("LIBRARY_PATH", "")
    )
    os.environ["LD_LIBRARY_PATH"] = (
        f"{cuda_home}/targets/x86_64-linux/lib:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )


setup_cuda_toolkit_env()

import comet_ml
import numpy as np
import torch

from config import Config as cfg

sys.path.insert(0, os.path.abspath("./src/optimizer"))
sys.path.insert(0, os.path.abspath("./src/models"))
sys.path.insert(0, os.path.abspath("./src/data"))

from src.data import build_refinedweb_dataloader
from src.models.llama import Llama, LlamaAttentionBackend, LlamaSize
from src.optimizer.MiniAdam import GaLoreMiniAdam
from src.optimizer.schedulers import WarmupScheduler


class NullExperiment:
    def set_name(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_text(self, *args, **kwargs):
        pass

    def log_histogram_3d(self, *args, **kwargs):
        pass

    def log_figure(self, *args, **kwargs):
        pass

    def end(self):
        pass


class SmokeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        # Deterministic byte-level tokenizer for fast local pipeline checks.
        ids = [(byte % 250) + 2 for byte in text.encode("utf-8")]
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [
            int(token)
            for token in token_ids
            if not skip_special_tokens or int(token) > self.eos_token_id
        ]
        return "".join(chr((token - 2) % 250) for token in tokens)


class SmokeCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=256, hidden_size=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = torch.tanh(self.proj(self.embed(input_ids)))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _batch_to_device(batch, valid_keys, device):
    return {k: v.to(device) for k, v in batch.items() if k in valid_keys}


def _build_optimizer(opt_type, model, projector, lr, update_gap, experiment):
    if opt_type == "adammini":
        return GaLoreMiniAdam(
            model.parameters(),
            projector=projector,
            lr=lr,
            update_gap=update_gap,
            experiment=experiment,
            model=model,
            max_projected_params=getattr(experiment, "max_projected_params", None),
        )
    if opt_type == "adam8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for adam8bit; install it or use adamw"
            ) from exc
        return bnb.optim.Adam8bit(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _is_experiment_available(opt_type):
    if opt_type != "adam8bit":
        return True
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def _append_csv_metrics(path, row):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate_val_loss(model, val_loader, valid_keys, device, max_batches=5):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for step, val_batch in enumerate(itertools.islice(val_loader, max_batches)):
        val_batch = _batch_to_device(val_batch, valid_keys, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**val_batch, use_cache=False).loss
        n_tokens = (val_batch["labels"] != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan")
    mean_loss = total_loss / total_tokens
    return mean_loss, float(np.exp(mean_loss))


def train_engine(opt_type, proj_type, args):
    cfg.setup()
    Llama.clear_model_cache()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    run_name = f"{opt_type}-{proj_type}-r{args.rank}"
    if args.no_comet:
        exp = NullExperiment()
    else:
        exp = comet_ml.Experiment(api_key=args.key, project_name="pythia-benchmarks")
    exp.max_projected_params = args.max_projected_params
    exp.spectrum_log_dir = args.spectrum_log_dir
    exp.set_name(run_name)
    logged_args = {k: v for k, v in vars(args).items() if k != "key"}
    exp.log_parameters({
        **logged_args,
        "opt": opt_type,
        "proj": proj_type,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "sequence_length": cfg.sequence_length,
        "tokens_per_step": cfg.batch_size * cfg.sequence_length,
        "scheduler": cfg.scheduler,
        "max_grad_norm": cfg.max_grad_norm,
        "smoke": args.smoke,
    })

    if args.synthetic_smoke:
        attention_backend = LlamaAttentionBackend.SDPA
        model = SmokeCausalLM().to(device)
        tokenizer = SmokeTokenizer()
    else:
        lsize = LlamaSize.from_suffix(args.size)
        attention_backend = LlamaAttentionBackend(args.attention)
        model = Llama.get_model(
            lsize,
            attention_backend=attention_backend,
            torch_dtype=torch.bfloat16,
            gradient_checkpointing=not args.smoke,
        ).to(device)
        tokenizer = Llama.get_tokenizer(lsize)

    train_loader = build_refinedweb_dataloader(
        data_dir=args.data,
        tokenizer=tokenizer,
        seq_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        packed_attention=False,
    )
    val_loader = build_refinedweb_dataloader(
        data_dir=args.data,
        tokenizer=tokenizer,
        seq_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        packed_attention=False,
    )
    test_loader = None
    if args.test_data:
        test_loader = build_refinedweb_dataloader(
            data_dir=args.test_data,
            tokenizer=tokenizer,
            seq_length=cfg.sequence_length,
            batch_size=cfg.batch_size,
            packed_attention=False,
        )
    val_sample = _batch_to_device(
        next(iter(val_loader)),
        ["input_ids", "attention_mask", "labels"],
        device,
    )

    projector = (
        cfg.projector_map[proj_type](rank=args.rank) if proj_type != "none" else None
    )
    if opt_type != "adammini" and projector is not None:
        raise ValueError(
            f"Projector '{proj_type}' requires opt_type=adammini, got {opt_type}"
        )

    optimizer = _build_optimizer(
        opt_type,
        model,
        projector,
        cfg.lr,
        cfg.update_gap,
        exp,
    )

    scheduler = None
    if cfg.scheduler:
        scheduler = WarmupScheduler.create(
            optimizer,
            name=cfg.scheduler["name"],
            num_warmup_steps=cfg.scheduler["num_warmup_steps"],
            num_training_steps=cfg.steps,
            min_lr=cfg.scheduler["min_lr"],
        )

    if attention_backend == LlamaAttentionBackend.FLEX:
        valid_keys = ["input_ids", "labels", "position_ids"]
    else:
        valid_keys = ["input_ids", "attention_mask", "labels"]

    model.train()
    vram_model_gb = get_tensor_memory(model, component="model")
    train_start = time.time()
    tokens_seen = 0
    final_val_loss = float("nan")
    final_val_ppl = float("nan")
    final_test_loss = float("nan")
    final_test_ppl = float("nan")

    train_iter = iter(train_loader)
    for step in range(cfg.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        step_start = time.time()
        batch = _batch_to_device(batch, valid_keys, device)
        tokens_seen += batch["input_ids"].numel()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss

        optimizer.zero_grad()
        loss.backward()

        grad_norms = [
            p.grad.norm(2) for p in model.parameters() if p.grad is not None
        ]
        total_norm = (
            torch.norm(torch.stack(grad_norms), 2) if grad_norms else torch.tensor(0.0)
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
        vram_grads_gb = get_tensor_memory(model, component="gradients")

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        vram_opt_gb = get_tensor_memory(model, optimizer, component="optimizer")
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        vram_activations_gb = max(
            0.0, vram_peak_gb - (vram_model_gb + vram_opt_gb + vram_grads_gb)
        )

        train_loss = loss.item()
        train_ppl = min(float(np.exp(train_loss)), 1000.0)
        elapsed = max(time.time() - train_start, 1e-9)
        tokens_per_sec = tokens_seen / elapsed

        metrics = {
            "loss": train_loss,
            "perplexity_loss": train_ppl,
            "vram_total_peak_gb": vram_peak_gb,
            "vram_model_gb": vram_model_gb,
            "vram_optimizer_gb": vram_opt_gb,
            "vram_gradients_gb": vram_grads_gb,
            "vram_activations_gb": vram_activations_gb,
            "iter_time": time.time() - step_start,
            "lr": scheduler.get_last_lr()[0] if scheduler is not None else cfg.lr,
            "grad_norm": float(total_norm),
            "tokens_per_sec": tokens_per_sec,
        }
        exp.log_metrics(metrics, step=step)

        val_interval = args.val_interval if not args.smoke else 1
        if step % val_interval == 0:
            val_loss, val_ppl = evaluate_val_loss(
                model, val_loader, valid_keys, device, max_batches=args.val_batches
            )
            final_val_loss = val_loss
            final_val_ppl = val_ppl
            if test_loader is not None:
                test_loss, test_ppl = evaluate_val_loss(
                    model,
                    test_loader,
                    valid_keys,
                    device,
                    max_batches=args.test_batches,
                )
                final_test_loss = test_loss
                final_test_ppl = test_ppl
            else:
                test_loss = float("nan")
                test_ppl = float("nan")
            exp.log_metrics(
                {
                    "val_loss": val_loss,
                    "val_perplexity": val_ppl,
                    "test_loss": test_loss,
                    "test_perplexity": test_ppl,
                },
                step=step,
            )

            if (
                not args.no_comet
                and not args.synthetic_smoke
                and not args.smoke
                and step % 20 == 0
            ):
                input_ids = val_sample["input_ids"][:1]
                prompt_len = input_ids.shape[1]
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=32,
                            do_sample=False,
                        )
                prompt_text = tokenizer.decode(
                    input_ids[0], skip_special_tokens=True
                )
                generated_text = tokenizer.decode(
                    output_ids[0][prompt_len:], skip_special_tokens=True
                )
                exp.log_text(
                    f"[step {step}]\nPROMPT:\n{prompt_text}\n\nGENERATED:\n{generated_text}"
                )

            print(
                f"[{run_name}] step {step} | train_loss={train_loss:.4f} "
                f"| val_ppl={val_ppl:.4f} | test_ppl={test_ppl:.4f} "
                f"| vram={vram_peak_gb:.2f}GB"
            )
        else:
            val_loss = float("nan")
            val_ppl = float("nan")
            test_loss = float("nan")
            test_ppl = float("nan")

        csv_row = {
            "run_name": run_name,
            "opt": opt_type,
            "proj": proj_type,
            "rank": args.rank,
            "size": args.size,
            "attention": args.attention,
            "seed": cfg.seed,
            "step": step,
            "tokens_seen": tokens_seen,
            "train_loss": train_loss,
            "train_ppl_capped": train_ppl,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "test_loss": test_loss,
            "test_perplexity": test_ppl,
            "vram_total_peak_gb": vram_peak_gb,
            "vram_model_gb": vram_model_gb,
            "vram_optimizer_gb": vram_opt_gb,
            "vram_gradients_gb": vram_grads_gb,
            "vram_activations_gb": vram_activations_gb,
            "iter_time": metrics["iter_time"],
            "tokens_per_sec": tokens_per_sec,
            "grad_norm": float(total_norm),
            "lr": metrics["lr"],
            "batch_size": cfg.batch_size,
            "sequence_length": cfg.sequence_length,
            "max_projected_params": args.max_projected_params,
            "wall_time_sec": time.time() - train_start,
        }
        _append_csv_metrics(args.csv_log, csv_row)

    exp.log_metrics(
        {
            "final_val_loss": final_val_loss,
            "final_val_perplexity": final_val_ppl,
            "final_test_loss": final_test_loss,
            "final_test_perplexity": final_test_ppl,
            "total_tokens": tokens_seen,
            "wall_time_sec": time.time() - train_start,
        }
    )
    exp.end()
    del model, optimizer
    torch.cuda.empty_cache()


def iter_experiments(opts, projs):
    for opt in opts:
        if opt == "adamw" or opt == "adam8bit":
            yield opt, "none"
            continue
        for proj in projs:
            if proj == "none":
                continue
            yield opt, proj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="./my_dataset_shards",
        help="Path to RefinedWeb JSONL shards",
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help="Optional path to held-out JSONL shards for test loss/perplexity",
    )
    parser.add_argument(
        "--key",
        default="Pd8psXxTfZFpP6RRP2es4y9zs",
        help="Comet ML API Key",
    )
    parser.add_argument("--size", default=cfg.model_size)
    parser.add_argument("--rank", type=int, default=cfg.rank)
    parser.add_argument(
        "--attention",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "flex_attention"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: few steps, small batch/seq, frequent validation",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Use a tiny local model/tokenizer for fast pipeline checks",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Disable Comet logging for local smoke checks",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--update-gap", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=20)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--test-batches", type=int, default=5)
    parser.add_argument(
        "--csv-log",
        default=None,
        help="Append per-step metrics to a local CSV file",
    )
    parser.add_argument(
        "--max-projected-params",
        type=int,
        default=None,
        help="Project only the first N matrix parameters; useful for fast smoke checks",
    )
    parser.add_argument(
        "--spectrum-log-dir",
        default=None,
        help="Write singular-value diagnostics JSONL files to this directory",
    )
    parser.add_argument(
        "--only",
        nargs=2,
        metavar=("OPT", "PROJ"),
        help="Run a single configuration, e.g. adammini lotus",
    )
    args = parser.parse_args()

    if args.smoke or args.synthetic_smoke:
        args.synthetic_smoke = True
        cfg.steps = args.steps or 3
        cfg.batch_size = args.batch_size or 2
        cfg.sequence_length = args.seq_length or 128
        cfg.update_gap = 2
        cfg.scheduler = None
        args.attention = "sdpa"
    else:
        if args.steps is not None:
            cfg.steps = args.steps
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.seq_length is not None:
            cfg.sequence_length = args.seq_length
        if args.lr is not None:
            cfg.lr = args.lr
        if args.update_gap is not None:
            cfg.update_gap = args.update_gap

    opts = cfg.opts
    projs = cfg.projs
    if args.only:
        opts = [args.only[0]]
        projs = [args.only[1]]

    for opt, proj in iter_experiments(opts, projs):
        if not _is_experiment_available(opt):
            print(f"\n>>> SKIPPING: opt={opt}, proj={proj} (bitsandbytes is not installed)")
            continue
        print(f"\n>>> RUNNING: opt={opt}, proj={proj}, rank={args.rank}")
        train_engine(opt, proj, args)
