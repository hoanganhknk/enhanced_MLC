import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets.utils import download_url, check_integrity

def uniform_mix_C(mixing_ratio, num_classes):
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

class CIFARBase(data.Dataset):
    base_folder = None
    url = None
    filename = None
    tgz_md5 = None
    train_list = None
    test_list = None

    def __init__(self, root, train=True, meta=True, num_meta=1000,
                 corruption_prob=0.0, corruption_type='unif',
                 transform=None, target_transform=None,
                 download=False, seed=1):
        self.root = root
        self.train = train
        self.meta = meta
        self.num_meta = int(num_meta)
        self.corruption_prob = float(corruption_prob)
        self.corruption_type = corruption_type
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. Use download=True.")

        if self.train:
            self._load_train()
        else:
            self._load_test()

    def _load_train(self):
        train_data = []
        train_labels = []
        coarse_labels = []

        for fentry in self.train_list:
            f = fentry[0]
            path = os.path.join(self.root, self.base_folder, f)
            with open(path, "rb") as fo:
                import pickle, sys
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding="latin1")

            train_data.append(entry["data"])
            if "labels" in entry:
                train_labels += entry["labels"]
                num_classes = 10
            else:
                train_labels += entry["fine_labels"]
                coarse_labels += entry["coarse_labels"]
                num_classes = 100

        train_data = np.concatenate(train_data).reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
        train_labels = np.array(train_labels, dtype=np.int64)

        if num_classes == 10:
            per_cls = int(self.num_meta / 10)
        else:
            per_cls = int(self.num_meta / 100)

        # balanced meta split
        idx_meta = []
        idx_train = []
        rng = np.random.RandomState(self.seed)
        for c in range(num_classes):
            idx_c = np.where(train_labels == c)[0]
            rng.shuffle(idx_c)
            idx_meta.extend(idx_c[:per_cls].tolist())
            idx_train.extend(idx_c[per_cls:].tolist())

        idx_meta = np.array(idx_meta, dtype=np.int64)
        idx_train = np.array(idx_train, dtype=np.int64)

        if self.meta:
            self.data = train_data[idx_meta]
            self.targets = train_labels[idx_meta].tolist()
            self.original_targets = self.targets.copy()
            self.coarse_targets = None
        else:
            self.data = train_data[idx_train]
            self.targets = train_labels[idx_train].tolist()
            self.original_targets = self.targets.copy()
            self.coarse_targets = None

            if self.corruption_prob > 0:
                if self.corruption_type == "unif":
                    C = uniform_mix_C(self.corruption_prob, num_classes)
                elif self.corruption_type == "flip":
                    C = flip_labels_C(self.corruption_prob, num_classes, seed=self.seed)
                elif self.corruption_type == "flip2":
                    C = flip_labels_C_two(self.corruption_prob, num_classes, seed=self.seed)
                elif self.corruption_type == "hierarchical":
                    assert num_classes == 100, "hierarchical chỉ dùng cho CIFAR-100"
                    coarse_labels = np.array(coarse_labels, dtype=np.int64)[idx_train]
                    self.coarse_targets = coarse_labels.tolist()

                    # build C: trong cùng coarse superclass
                    coarse_fine = [set() for _ in range(20)]
                    for y, yc in zip(self.targets, self.coarse_targets):
                        coarse_fine[yc].add(y)
                    coarse_fine = [list(s) for s in coarse_fine]

                    C = np.eye(num_classes) * (1 - self.corruption_prob)
                    for i in range(20):
                        fine = coarse_fine[i]
                        for j, y in enumerate(fine):
                            others = [t for t in fine if t != y]
                            if len(others) > 0:
                                C[y, others] += self.corruption_prob * (1 / len(others))
                else:
                    raise ValueError(f"Invalid corruption_type={self.corruption_type}")

                rng = np.random.RandomState(self.seed)
                noisy_targets = []
                for y in self.targets:
                    noisy_targets.append(int(rng.choice(num_classes, p=C[y])))
                self.targets = noisy_targets
                self.corruption_matrix = C

        self.num_classes = num_classes

    def _load_test(self):
        f = self.test_list[0][0]
        path = os.path.join(self.root, self.base_folder, f)
        with open(path, "rb") as fo:
            import pickle, sys
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding="latin1")

        test_data = entry["data"].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
        if "labels" in entry:
            test_labels = entry["labels"]
        else:
            test_labels = entry["fine_labels"]

        self.data = test_data
        self.targets = test_labels
        self.num_classes = 10 if "labels" in entry else 100

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)

    def _check_integrity(self):
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_url(self.url, self.root, self.filename, self.tgz_md5)
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

class CIFAR10Corrupted(CIFARBase):
    base_folder = "cifar-10-batches-py"
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [["test_batch", "40351d587109b95175f43aff81a1287e"]]

class CIFAR100Corrupted(CIFARBase):
    base_folder = "cifar-100-python"
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]

def build_transforms(loader_style: str):
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    normalize = transforms.Normalize(mean=mean, std=std)

    if loader_style == "mwnet":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4), mode="reflect").squeeze(0)),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, test_tf

def make_cifar_loaders(dataset: str, data_root: str, gold_fraction: float,
                      corruption_prob: float, corruption_type: str,
                      batch_size: int, num_workers: int,
                      loader_style: str, seed: int):
    train_tf, test_tf = build_transforms(loader_style)

    num_meta = int(50000 * gold_fraction)

    if dataset == "cifar10":
        Gold = CIFAR10Corrupted
        Silv = CIFAR10Corrupted
        Test = CIFAR10Corrupted
    elif dataset == "cifar100":
        Gold = CIFAR100Corrupted
        Silv = CIFAR100Corrupted
        Test = CIFAR100Corrupted
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    gold_ds = Gold(root=data_root, train=True, meta=True, num_meta=num_meta,
                   corruption_prob=0.0, corruption_type="unif",
                   transform=train_tf, download=True, seed=seed)

    silver_ds = Silv(root=data_root, train=True, meta=False, num_meta=num_meta,
                     corruption_prob=corruption_prob, corruption_type=corruption_type,
                     transform=train_tf, download=True, seed=seed)

    test_ds = Test(root=data_root, train=False, meta=False, num_meta=num_meta,
                   corruption_prob=0.0, corruption_type="unif",
                   transform=test_tf, download=True, seed=seed)

    gold_loader = torch.utils.data.DataLoader(
        gold_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    silver_loader = torch.utils.data.DataLoader(
        silver_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return gold_loader, silver_loader, test_loader, gold_ds.num_classes
