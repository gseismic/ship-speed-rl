
import torch
import numpy as np
import random

def seed_all(seed):
    # 1. 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 确保 CUDA 操作确定性（可能降低性能）
        torch.backends.cudnn.benchmark = False

    # 2. 设置 NumPy 随机种子（环境内部可能依赖 NumPy）
    np.random.seed(seed)
    random.seed(seed)