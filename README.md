## 1. Overview

This is the Pytorch implementation for "Efficient Bilevel Optimization for Noisy Labels Learning using Meta Label Correction Framework". This repo is based on several MLC implementations.

Authors: Ba Hoang Anh Nguyen, Cuong Ta

<p align="center">
  <img src="figures/EBOMLC.png" alt="Framework of EBOMLC">
</p>

## 2. Setup

### 2.1 Environment
`pip install -r requirement.txt`

### 2.2 Computational Resources
All experiments run in Kaggle notebook.

## 3. Experiments
### 3.1 This repository currently supports three methods:

- `mlc` — Baseline Meta Label Correction  
- `mlcbome` — MLC with BOME updates  
- `ebomlc` — Efficient bi-level optimization for MLC

### 3.2 To run these methods:
```
python main.py --method ebomlc --dataset cifar100 --optimizer sgd --bs 100 \
  --corruption_type unif --corruption_level 0.8 --gold_fraction 0.02 \
  --epochs 120 --main_lr 0.1 --meta_lr 3e-4 --runid ebomlc_c100_u80 --cls_dim 128 \
  --rho 0.2 --xi 0.5 --delta 0.25
```
| Argument                 | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| `--method`               | Training method (options: `ebomlc`, `mlc`, `mlcbome`).      |
| `--dataset`              | Dataset to use (`cifar10` or `cifar100`).                   |
| `--optimizer`            | Optimizer for the main network.                             |
| `--bs`                   | Batch size for training data.                               |
| `--corruption_type`      | Noise type (`unif`, `flip`).                                |
| `--corruption_level`     | Noise ratio (e.g., `0.4` = 40% noisy labels).               |
| `--gold_fraction`        | Fraction of clean (gold) data.                              |
| `--epochs`               | Number of training epochs.                                  |
| `--main_lr`              | Learning rate for the main network.                         |
| `--meta_lr`              | Learning rate for the meta network.                         |
| `--runid`                | Identifier for logs and checkpoints.                        |
| `--cls_dim`              | Dimension of label embedding in the meta model.             |


## 4. Baselines
For the baselines, please follow these repos and papers:
- [MW-net](https://github.com/xjtushujun/meta-weight-net)
- [L2RW](https://github.com/uber-research/learning-to-reweight-examples)
- [GLC](https://github.com/mmazeika/glc)
