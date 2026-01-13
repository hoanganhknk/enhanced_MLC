# utils.py
import os
import random
import numpy as np
import torch

class DataIterator:
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    return batch.to(device, non_blocking=True)

@torch.no_grad()
def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += int(n)
    @property
    def avg(self):
        return self.sum / max(self.cnt, 1)

def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
