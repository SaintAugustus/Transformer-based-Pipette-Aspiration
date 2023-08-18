import random

import numpy as np
import pandas as pd
import torch

# split data into train, valid, test
def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    same_seed(10)
