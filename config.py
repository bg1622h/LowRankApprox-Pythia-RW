import os
import sys

import torch

sys.path.insert(0, os.path.abspath("./src/optimizer"))
sys.path.insert(0, os.path.abspath("./src/models"))
sys.path.insert(0, os.path.abspath("./src/data"))

from src.optimizer.GaLore import GaLoreProjector
from src.optimizer.AdaptiveStochasticProjector import AdaptiveStochasticProjector
from src.optimizer.GaLore2 import GaLore2Projector
from src.optimizer.Lotus import Lotus
from src.optimizer.OldStochasticProjector import OldStochasticProjector
from src.optimizer.StochasticProjector import StochasticProjector
from src.optimizer.fisher_galore_projector import FisherGaLoreProjector
from src.optimizer.block_fisher_galore_projector import BlockFisherGaLoreProjector
from src.optimizer.fisher_selection_projectors import (
    TopKFisherGaLoreProjector,
    SoftmaxFisherGaLoreProjector,
)


class Config:
    seed = 666
    opts = ["adamw", "adam8bit", "adammini",]
    projs = [
        "galore",
        "galore2",
        "lotus",
        "stochastic_old",
        "stochastic",
        "adaptive_stochastic",
    ]

    projector_map = {
        "galore": GaLoreProjector,
        "galore2": GaLore2Projector,
        "lotus": Lotus,
        "stochastic_old": OldStochasticProjector,
        "stochastic": StochasticProjector,
        "adaptive_stochastic": AdaptiveStochasticProjector,
        'fisher_projector': FisherGaLoreProjector,
        'block_fisher_projector': BlockFisherGaLoreProjector,
        'topk_fisher_projector': TopKFisherGaLoreProjector,
        'softmax_fisher_projector': SoftmaxFisherGaLoreProjector,
    }

    model_size = "1.1B"
    rank = 8
    lr = 5e-4
    steps = 1500
    batch_size = 8
    sequence_length = 512
    update_gap = 300
    scheduler = {
        "name": "linear",
        "num_warmup_steps": 250,
        "min_lr": 1e-8,
    }
    max_grad_norm = 1.0

    @staticmethod
    def setup():
        torch.manual_seed(Config.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training")
