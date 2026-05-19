import random

import torch
import numpy as np

def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Integer seed used to initialize all random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
