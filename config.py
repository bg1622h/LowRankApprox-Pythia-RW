import os
import sys

import torch

sys.path.insert(0, os.path.abspath("./src/optimizer"))
sys.path.insert(0, os.path.abspath("./src/models"))
sys.path.insert(0, os.path.abspath("./src/data"))

from src.optimizer.GaLore import GaLoreProjector
from src.optimizer.AdaptiveStochasticProjector import AdaptiveStochasticProjector
from src.optimizer.AdaptiveStochasticGaLore2Projector import AdaptiveStochasticGaLore2Projector
from src.optimizer.GaLore2 import GaLore2Projector
from src.optimizer.Lotus import Lotus
from src.optimizer.OldStochasticProjector import OldStochasticProjector
from src.optimizer.StochasticProjector import StochasticProjector
from src.optimizer.StochasticGaLore2Projector import StochasticGaLore2Projector
from src.optimizer.StochasticLotus import StochasticLotus
from src.optimizer.AdaptiveStochasticLotus import AdaptiveStochasticLotus
from src.optimizer.fisher_galore_projector import FisherGaLoreProjector
from src.optimizer.block_fisher_galore_projector import (
    BlockFisherGaLoreProjector,
    WhitenedBlockFisherGaLoreProjector,
    StochasticWhitenedBlockFisherGaLoreProjector,
    AdaptiveWhitenedBlockFisherGaLoreProjector,
)
from src.optimizer.fisher_selection_projectors import (
    TopKFisherGaLoreProjector,
    SoftmaxFisherGaLoreProjector,
)
from src.optimizer.hvp_selection_projectors import (
    TopKHVPProjector,
    SoftmaxHVPProjector,
    AdaptiveHVPProjector,
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
        "adaptive_stochastic_galore2",
        'stochastic_galore2',
        'StochasticLotus',
        'AdaptiveStochasticLotus',
    ]

    projector_map = {
        "galore": GaLoreProjector,
        "galore2": GaLore2Projector,
        "lotus": Lotus,
        "stochastic_old": OldStochasticProjector,
        "stochastic": StochasticProjector,
        "adaptive_stochastic": AdaptiveStochasticProjector,
        "adaptive_stochastic_galore2": AdaptiveStochasticGaLore2Projector,
        'fisher_projector': FisherGaLoreProjector,
        'block_fisher_projector': BlockFisherGaLoreProjector,
        'whitened_block_fisher_projector': WhitenedBlockFisherGaLoreProjector,
        'stochastic_whitened_block_fisher_projector': StochasticWhitenedBlockFisherGaLoreProjector,
        'adaptive_whitened_block_fisher_projector': AdaptiveWhitenedBlockFisherGaLoreProjector,
        'topk_fisher_projector': TopKFisherGaLoreProjector,
        'softmax_fisher_projector': SoftmaxFisherGaLoreProjector,
        'topk_hvp_projector': TopKHVPProjector,
        'softmax_hvp_projector': SoftmaxHVPProjector,
        'adaptive_hvp_projector': AdaptiveHVPProjector,
        'stochastic_galore2': StochasticGaLore2Projector,
        'StochasticLotus': StochasticLotus,
        'AdaptiveStochasticLotus': AdaptiveStochasticLotus,
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
