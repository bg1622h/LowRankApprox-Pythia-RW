import argparse
import time
import os
import logging

logging.basicConfig(
    filename="app.log",  # Имя файла, куда писать логи
    filemode="w",  # 'a' — добавлять в конец, 'w' — перезаписывать файл при старте
    level=logging.DEBUG,  # Уровень важности (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Формат записи
    datefmt="%Y-%m-%d %H:%M:%S",  # Формат даты и времени
    encoding="utf-8",  # Корректная запись кириллицы
)
def get_tensor_memory(model, optimizer=None, component="model"):
    """Считает точный размер памяти тензоров в гигабайтах."""
    mem_bytes = 0
    if component == "model":
        mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    elif component == "gradients":
        mem_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    elif component == "optimizer" and optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    mem_bytes += v.numel() * v.element_size()
    return mem_bytes / (1024**3)

def setup_cuda_toolkit_env():
    cuda_home = "/usr/local/cuda-13.2"
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault(
        "TRITON_PTXAS_PATH",
        f"{cuda_home}/bin/ptxas",
    )
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
import torch
import comet_ml
from config import Config as cfg
import sys
sys.path.insert(0, os.path.abspath('./src/optimizer'))
sys.path.insert(0, os.path.abspath('./src/models'))
sys.path.insert(0, os.path.abspath('./src/data'))

from src.models.pythia import Pythia, PythiaAttentionBackend, PythiaSize
from src.optimizer.MiniAdam import MiniAdam, GaLoreMiniAdam
from src.optimizer.schedulers import WarmupScheduler
from src.data import build_refinedweb_dataloader
from src.models.llama import *

def train_engine(opt_type, proj_type, args):
    cfg.setup()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    exp = comet_ml.Experiment(api_key=args.key, project_name="pythia-benchmarks")
    exp.set_name(f"{opt_type}-{proj_type}-r{args.rank}")
    if cfg.scheduler:
        exp.log_parameters({
            **vars(args),
            "opt": opt_type,
            "proj": proj_type,
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "sequence_length": cfg.sequence_length,
            "tokens_per_step": cfg.batch_size * cfg.sequence_length,
            "scheduler": cfg.scheduler,
            "max_grad_norm": cfg.max_grad_norm,
        })
    else:
        exp.log_parameters({
            **vars(args),
            "opt": opt_type,
            "proj": proj_type,
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "sequence_length": cfg.sequence_length,
            "tokens_per_step": cfg.batch_size * cfg.sequence_length,
            "max_grad_norm": cfg.max_grad_norm,
        })


    # psize = PythiaSize.from_suffix(args.size)
    # attention_backend = PythiaAttentionBackend(args.attention)
    # model = Pythia.get_model(
    #     psize,
    #     attention_backend=attention_backend,
    #     torch_dtype=torch.bfloat16,
    #     gradient_checkpointing=True,
    # ).cuda()
    # tokenizer = Pythia.get_tokenizer(psize)
    lsize = LlamaSize.from_suffix(args.size)
    attention_backend = LlamaAttentionBackend(args.attention)
    model = Llama.get_model(
        lsize,
        attention_backend=attention_backend,
        torch_dtype=torch.bfloat16,
        gradient_checkpointing=True,
    ).cuda()
    tokenizer = Llama.get_tokenizer(lsize)

    loader = build_refinedweb_dataloader(
        data_dir=args.data,
        tokenizer=tokenizer,
        seq_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        packed_attention=False,
    )

    projector = cfg.projector_map[proj_type](rank=args.rank) if proj_type != "none" else None
    if opt_type == "adammini":
        optimizer = GaLoreMiniAdam(
            model.parameters(),
            projector=projector,
            lr=cfg.lr,
            update_gap=cfg.update_gap
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    scheduler = None
    # if cfg.scheduler:
    #     scheduler = WarmupScheduler.create(
    #         optimizer,
    #         name = cfg.scheduler['name'],
    #         num_warmup_steps = cfg.scheduler['num_warmup_steps'],
    #         num_training_steps = cfg.steps,
    #         min_lr = cfg.scheduler['min_lr'],
    #     )

    model.train()
    start_time = time.time()
    vram_model_gb = get_tensor_memory(model, component="model")

    for step, batch in enumerate(loader):
        #print(tokenizer.decode(batch['input_ids'][0]))
        #break
        if step >= cfg.steps: break
        
        step_start = time.time()
        if attention_backend == PythiaAttentionBackend.FLEX:
            # FlexAttention builds the same packed causal BlockMask from reset
            # position_ids, without materializing the old 3D attention_mask.
            valid_keys = ["input_ids", "labels", "position_ids"]
        else:
            valid_keys = ["input_ids", "attention_mask", "labels"]
        batch = {k: v.cuda() for k, v in batch.items() if k in valid_keys}
        logging.debug({k: tuple(v.shape) for k, v in batch.items()})

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss

        optimizer.zero_grad()
        loss.backward()
        # 1. Получаем градиенты (loss.backward())
        # 2. Проходим по всем параметрам, которые обучаем через Lotus:
        
        for p in model.parameters():
            if hasattr(p, 'lotus_engine'): # Допустим, ты сохранил объект Lotus в параметре
                engine = p.lotus_engine
                
                # Проецируем градиент
                # Внутри project() может сработать смена базиса
                low_rank_grad = engine.project(p.grad)
                
                # ИСПОЛЬЗУЕМ ФЛАГ ЗДЕСЬ:
                if engine.was_switched:
                    # Очищаем состояния оптимизатора (m и v у Adam) именно для этого веса
                    # Это предотвращает "взрыв" весов из-за старой инерции
                    if p in optimizer.state:
                        optimizer.state[p].clear() 
                
                # Заменяем полный градиент на низкоранговый для шага оптимизатора
                p.grad = engine.reconstruct(low_rank_grad)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.max_grad_norm,
        )
        total_norm = torch.norm(
        torch.stack([
                p.grad.norm(2)
                for p in model.parameters()
                if p.grad is not None
            ]),
            2
        )
        first_grad_norm = next(model.parameters()).grad.norm()
        last_grad_norm =  list(model.parameters())[-1].grad.norm()
        
        vram_grads_gb = get_tensor_memory(model, component="gradients")

        optimizer.step()
        if scheduler:
            scheduler.step()
        
        vram_opt_gb = get_tensor_memory(model, optimizer, component="optimizer")
        
    
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        vram_activations_gb = vram_peak_gb - (vram_model_gb + vram_opt_gb + vram_grads_gb)
        vram_activations_gb = max(0, vram_activations_gb)

        vram = torch.cuda.max_memory_allocated() / (1024**3)
        perplexity_loss = torch.exp(loss).item()
        
        exp.log_metrics({
            "loss": loss.item(),
            "perplexity_loss": perplexity_loss if perplexity_loss < 1000 else 1000,
            "vram_total_peak_gb": vram_peak_gb,
            "vram_model_gb": vram_model_gb,
            "vram_optimizer_gb": vram_opt_gb,
            "vram_gradients_gb": vram_grads_gb,
            "vram_activations_gb": vram_activations_gb, # Зависит от batch size и sequence length
            "iter_time": time.time() - step_start,
            'lr': scheduler.get_last_lr()[0] if scheduler is not None else cfg.lr,
            'grad_norm': total_norm,
            'first_grad_norm': first_grad_norm,
            'last_grad_norm': last_grad_norm,
        }, step=step)

        if step % 20 == 0:
            print(f"[{opt_type}-{proj_type}] Step {step} | Loss: {loss.item():.4f} | "
                  f"VRAM Peak: {vram_peak_gb:.2f}GB (Model: {vram_model_gb:.2f}, "
                  f"Opt: {vram_opt_gb:.2f}, Grads: {vram_grads_gb:.2f}, Acts: {vram_activations_gb:.2f})")
    
    exp.end()
    del model, optimizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, help="Path to RefinedWeb shards", default = './my_dataset_shards')
    parser.add_argument("--key", required=False, help="Comet ML API Key", default = 'Pd8psXxTfZFpP6RRP2es4y9zs')
    parser.add_argument("--size", default=cfg.model_size)
    parser.add_argument("--rank", type=int, default=cfg.rank)
    parser.add_argument(
        "--attention",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "flex_attention"],
    )
    args = parser.parse_args()

    for opt in cfg.opts:
        if opt == "adamw":
            proj = "none"
            print(f"\n>>> RUNNING EXPERIMENT: OPT={opt.upper()}, PROJ={proj}")
            print(torch.backends.cuda.flash_sdp_enabled())
            print(torch.backends.cuda.mem_efficient_sdp_enabled())
            print(torch.backends.cuda.math_sdp_enabled())
            train_engine(opt, proj, args)
        else:
            for proj in cfg.projs:
                if proj != "none" and proj !=  "galore":
                    print(proj, "deniwnfiknfeifn")
                    print(f"\n>>> RUNNING EXPERIMENT: OPT={opt.upper()}, PROJ={proj.upper()}")
                    print(torch.backends.cuda.flash_sdp_enabled())
                    print(torch.backends.cuda.mem_efficient_sdp_enabled())
                    print(torch.backends.cuda.math_sdp_enabled())
                    train_engine(opt, proj, args)