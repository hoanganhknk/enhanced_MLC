# CIFAR/data_helper_cifar.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

from CIFAR.load_corrupted_data_mlg import load_cifar_splits, NumpyCIFAR

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def build_transforms():
    tf_noisy = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    # Clean augmentations include AutoAugment (Table 5) :contentReference[oaicite:6]{index=6}
    tf_clean = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return tf_noisy, tf_clean, tf_test

def prepare_cifar_loaders(args):
    tf_noisy, tf_clean, tf_test = build_transforms()

    splits = load_cifar_splits(
        dataset=args.dataset,
        root=args.data_path,
        gold_fraction=args.gold_fraction,
        corruption_type=args.corruption_type,
        corruption_level=args.corruption_level,
        seed=args.data_seed,
        download=True,
    )
    num_classes = splits["num_classes"]

    gold_data, gold_y = splits["gold"]
    silver_data, silver_y = splits["silver"]

    # split gold into gold_train and gold_meta (50/50 by default)
    r = np.random.RandomState(args.data_seed + 777)
    idx = np.arange(len(gold_y))
    r.shuffle(idx)
    split = int(round(len(idx) * args.gold_train_fraction))
    idx_train = idx[:split]
    idx_meta  = idx[split:]

    gold_train = NumpyCIFAR(gold_data[idx_train], gold_y[idx_train], transform=tf_clean)
    gold_meta  = NumpyCIFAR(gold_data[idx_meta],  gold_y[idx_meta],  transform=tf_test)

    silver_train = NumpyCIFAR(silver_data, silver_y, transform=tf_noisy)

    # torchvision test set
    if args.dataset.lower() == "cifar10":
        testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=tf_test)
    else:
        testset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=tf_test)

    loader_s = DataLoader(silver_train, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    loader_g_train = DataLoader(gold_train, batch_size=args.gold_bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    loader_g_meta  = DataLoader(gold_meta, batch_size=args.gold_bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    loader_test = DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    return num_classes, loader_s, loader_g_train, loader_g_meta, loader_test
