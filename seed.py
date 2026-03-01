import torch
import numpy as np
import random
import os
from torch import Generator, cuda

# Default seed value
DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for Python's random module, NumPy, and PyTorch
    (including CUDA) to ensure deterministic behavior across runs.
    
    Args:
        seed: Integer seed value (default: 42)
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
