# CIFAR/load_corrupted_data_mlg.py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

# CIFAR-10 asymmetric mapping (Appendix E.1) :contentReference[oaicite:5]{index=5}
_C10_ASYM = {
    9: 1,  # truck -> automobile
    2: 0,  # bird -> airplane
    4: 7,  # deer -> horse
    3: 5,  # cat -> dog
    5: 3,  # dog -> cat
}

# CIFAR-100 coarse mapping: list of 100 fine labels -> 20 coarse labels (standard mapping)
_C100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]

def _rng(seed: int):
    return np.random.RandomState(seed)

class NumpyCIFAR(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        self.data = data
        self.targets = targets.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
        y = int(self.targets[idx])
        # torchvision transforms expect PIL or ndarray; CIFAR data is uint8 HWC
        if self.transform is not None:
            img = self.transform(img)
        return img, y

def _class_balanced_indices(y: np.ndarray, n_per_class: int, seed: int, num_classes: int):
    r = _rng(seed)
    idxs = []
    for c in range(num_classes):
        c_idx = np.where(y == c)[0]
        r.shuffle(c_idx)
        idxs.append(c_idx[:n_per_class])
    return np.concatenate(idxs)

def _apply_unif(y: np.ndarray, p: float, seed: int, num_classes: int):
    r = _rng(seed)
    y2 = y.copy()
    m = r.rand(len(y2)) < p
    y2[m] = r.randint(0, num_classes, size=m.sum())
    return y2

def _apply_flip(y: np.ndarray, p: float, seed: int, num_classes: int):
    r = _rng(seed)
    y2 = y.copy()
    # fixed mapping per class
    mapping = np.arange(num_classes)
    for c in range(num_classes):
        choices = [j for j in range(num_classes) if j != c]
        mapping[c] = r.choice(choices)
    m = r.rand(len(y2)) < p
    y2[m] = mapping[y2[m]]
    return y2

def _apply_flip2(y: np.ndarray, p: float, seed: int, num_classes: int):
    r = _rng(seed)
    y2 = y.copy()
    map1 = np.arange(num_classes)
    map2 = np.arange(num_classes)
    for c in range(num_classes):
        choices = [j for j in range(num_classes) if j != c]
        a, b = r.choice(choices, size=2, replace=False)
        map1[c], map2[c] = a, b
    u = r.rand(len(y2))
    m1 = u < (p / 2)
    m2 = (u >= (p / 2)) & (u < p)
    y2[m1] = map1[y2[m1]]
    y2[m2] = map2[y2[m2]]
    return y2

def _apply_asym_c10(y: np.ndarray, p: float, seed: int):
    r = _rng(seed)
    y2 = y.copy()
    m = r.rand(len(y2)) < p
    for i in np.where(m)[0]:
        y2[i] = _C10_ASYM.get(int(y2[i]), int(y2[i]))
    return y2

def _apply_asym_c100_successor_in_coarse(y: np.ndarray, p: float, seed: int):
    r = _rng(seed)
    y2 = y.copy()
    fine_to_coarse = np.array(_C100_FINE_TO_COARSE)
    coarse_to_fines = {k: np.where(fine_to_coarse == k)[0].tolist() for k in range(20)}
    m = r.rand(len(y2)) < p
    for i in np.where(m)[0]:
        fine = int(y2[i])
        coarse = int(fine_to_coarse[fine])
        fines = coarse_to_fines[coarse]
        pos = fines.index(fine)
        y2[i] = fines[(pos + 1) % len(fines)]
    return y2

def _apply_hierarchical_c100(y: np.ndarray, p: float, seed: int):
    r = _rng(seed)
    y2 = y.copy()
    fine_to_coarse = np.array(_C100_FINE_TO_COARSE)
    coarse_to_fines = {k: np.where(fine_to_coarse == k)[0] for k in range(20)}
    m = r.rand(len(y2)) < p
    for i in np.where(m)[0]:
        fine = int(y2[i])
        coarse = int(fine_to_coarse[fine])
        candidates = coarse_to_fines[coarse]
        candidates = candidates[candidates != fine]
        y2[i] = int(r.choice(candidates))
    return y2

def load_cifar_splits(
    dataset: str,
    root: str,
    gold_fraction: float,
    corruption_type: str,
    corruption_level: float,
    seed: int,
    download: bool = True
):
    dataset = dataset.lower()
    assert dataset in ("cifar10", "cifar100")
    is_c10 = dataset == "cifar10"
    num_classes = 10 if is_c10 else 100

    ds = datasets.CIFAR10(root=root, train=True, download=download) if is_c10 else datasets.CIFAR100(root=root, train=True, download=download)
    data = ds.data                      # (50000,32,32,3) uint8
    targets = np.array(ds.targets)

    n_total = len(targets)
    n_gold = int(round(n_total * gold_fraction))
    n_gold = max(n_gold, num_classes)   # at least 1 per class in extreme cases
    n_per = n_gold // num_classes
    gold_idx = _class_balanced_indices(targets, n_per_class=n_per, seed=seed, num_classes=num_classes)

    # if remainder exists, add more random indices (still from remaining pool)
    remaining = np.setdiff1d(np.arange(n_total), gold_idx)
    rem = n_gold - len(gold_idx)
    if rem > 0:
        r = _rng(seed + 13)
        r.shuffle(remaining)
        gold_idx = np.concatenate([gold_idx, remaining[:rem]])

    silver_idx = np.setdiff1d(np.arange(n_total), gold_idx)

    gold_data = data[gold_idx]
    gold_y = targets[gold_idx]          # clean

    silver_data = data[silver_idx]
    silver_clean_y = targets[silver_idx]
    p = float(corruption_level)

    ct = corruption_type.lower()
    if ct == "unif":
        silver_noisy_y = _apply_unif(silver_clean_y, p, seed + 1, num_classes)
    elif ct == "flip":
        silver_noisy_y = _apply_flip(silver_clean_y, p, seed + 2, num_classes)
    elif ct == "flip2":
        silver_noisy_y = _apply_flip2(silver_clean_y, p, seed + 3, num_classes)
    elif ct == "asym":
        if is_c10:
            silver_noisy_y = _apply_asym_c10(silver_clean_y, p, seed + 4)
        else:
            silver_noisy_y = _apply_asym_c100_successor_in_coarse(silver_clean_y, p, seed + 4)
    elif ct == "hierarchical":
        assert not is_c10, "hierarchical only supported for CIFAR-100"
        silver_noisy_y = _apply_hierarchical_c100(silver_clean_y, p, seed + 5)
    else:
        raise ValueError(f"Unknown corruption_type={corruption_type}")

    out = dict(
        num_classes=num_classes,
        gold=(gold_data, gold_y),
        silver=(silver_data, silver_noisy_y),
        test_dataset="cifar10" if is_c10 else "cifar100",
    )
    return out
