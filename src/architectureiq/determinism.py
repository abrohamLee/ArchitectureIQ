import os
import random

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    """播种所有 RNG 并开启确定性算法。训练前必须调用。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
