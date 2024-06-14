
def seed_everything(seed: int, seed_cudnn=False):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if seed_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

