# main_cifar.py
import os
import argparse
import torch
import torch.optim as optim

from CIFAR.data_helper_cifar import prepare_cifar_loaders
from models import build_resnet32
from meta_models import Teacher
from utils import set_seed

def build_args():
    p = argparse.ArgumentParser("EMLC CIFAR (Kaggle-stable)")

    # data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--gold_fraction", type=float, default=0.02)  # paper uses 1000 clean (~0.02) :contentReference[oaicite:11]{index=11}
    p.add_argument("--gold_train_fraction", type=float, default=0.5)
    p.add_argument("--corruption_type", type=str, default="unif",
                   choices=["unif", "flip", "flip2", "asym", "hierarchical"])
    p.add_argument("--corruption_level", type=float, default=0.9)
    p.add_argument("--data_seed", type=int, default=1)

    # train
    p.add_argument("--epochs", type=int, default=15)             # Table 5 :contentReference[oaicite:12]{index=12}
    p.add_argument("--bs", type=int, default=128)                # noisy batch size :contentReference[oaicite:13]{index=13}
    p.add_argument("--gold_bs", type=int, default=32)            # clean batch size :contentReference[oaicite:14]{index=14}
    p.add_argument("--test_bs", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)

    # optimizer (paper CIFAR: lr=0.02, wd=0, mom=0.9, no scheduler) :contentReference[oaicite:15]{index=15}
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # emlc
    p.add_argument("--k", type=int, default=5)                   # paper finds k=5 best on CIFAR :contentReference[oaicite:16]{index=16}
    p.add_argument("--meta_interval", type=int, default=5)        # usually = k
    p.add_argument("--lambda_gold", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)

    # teacher objectives
    p.add_argument("--teacher_bce", type=str, default="adv", choices=["none", "rand", "adv"])
    p.add_argument("--w_ce", type=float, default=1.0)
    p.add_argument("--w_bce", type=float, default=1.0)

    # misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ssl_path", type=str, default="")           # optional
    p.add_argument("--out_dir", type=str, default="./out")
    p.add_argument("--clip_grad", type=float, default=5.0)

    return p.parse_args()

def main():
    args = build_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes, loader_s, loader_g_train, loader_g_meta, loader_test = prepare_cifar_loaders(args)

    # student model: resnet32
    student = build_resnet32(num_classes=num_classes, ssl_path=args.ssl_path)

    # teacher backbone: independent resnet32 (no shared weights unless bạn muốn)
    teacher_backbone = build_resnet32(num_classes=num_classes, ssl_path="")
    teacher = Teacher(teacher_backbone, num_classes=num_classes, feat_dim=64, emb_dim=64, hidden=128)

    student.to(device)
    teacher.to(device)

    opt_s = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    opt_t = optim.SGD(teacher.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    from trainer import Trainer
    tr = Trainer(student, teacher, opt_s, opt_t,
                 loader_s, loader_g_train, loader_g_meta, loader_test,
                 args, device, num_classes)
    tr.train()

if __name__ == "__main__":
    main()
