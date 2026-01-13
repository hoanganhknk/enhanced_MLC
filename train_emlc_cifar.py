import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from enhanced_MLC.datasets.cifar_corrupted import make_cifar_loaders
from enhanced_MLC.models.resnet_cifar import resnet32
from enhanced_MLC.models.teacher_emlc import EMLCTeacher
from enhanced_MLC.models.metanet import MetaNet
from enhanced_MLC.utils.seed import set_seed
from enhanced_MLC.utils.data_iter import DataIterator
from enhanced_MLC.utils.meters import AverageMeter, accuracy_top1

# -------------------------
# Loss helpers
# -------------------------
def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor):
    # CE(q, p) = - sum q * log p
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=1).mean()

def hard_cross_entropy(logits: torch.Tensor, y: torch.Tensor):
    return F.cross_entropy(logits, y)

# -------------------------
# Functional utilities for torch.func.jvp
# -------------------------
def _get_named_params(model) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.named_parameters()}

def _get_named_buffers(model) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.named_buffers()}

def _clone_state(model) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    return params, buffers

def _grads_to_named_dict(model, grads_list) -> Dict[str, torch.Tensor]:
    out = {}
    for (name, p), g in zip(model.named_parameters(), grads_list):
        out[name] = torch.zeros_like(p) if g is None else g.detach()
    return out

@dataclass
class Snapshot:
    x: torch.Tensor
    y: torch.Tensor
    params: Dict[str, torch.Tensor]
    buffers: Dict[str, torch.Tensor]

# -------------------------
# Meta-gradient (FPMG)
# -------------------------
def compute_meta_grads_fpmg(
    student: nn.Module,
    teacher,
    snapshots: List[Snapshot],
    x_clean: torch.Tensor,
    y_clean: torch.Tensor,
    lr_student: float,
    meta_interval: int,
    meta_grad_impl: str,
):
    """
    Trả về list grads tương ứng teacher.parameters() (chưa nhân meta_lr).
    Meta-gradient theo paper: dùng g_w (grad clean wrt student),
    và JVP log p theo direction g_w để tạo proxy sum(q * jvp).
    """
    # 1) g_w = ∇_w L_clean(student(w_cur))
    student.zero_grad(set_to_none=True)
    logits_c = student(x_clean)
    loss_c = F.cross_entropy(logits_c, y_clean)
    g_list = torch.autograd.grad(loss_c, list(student.parameters()), retain_graph=False, create_graph=False)
    g_named = _grads_to_named_dict(student, g_list)

    if meta_grad_impl == "autograd":
        # fallback chậm: (khuyên chỉ dùng k=1)
        # meta_proxy = sum over snapshots: weight * dot(∇_w Le, g_w) rồi grad wrt teacher
        meta_proxy = 0.0
        gamma = 1.0 - lr_student
        w_factor = 1.0
        # recent -> old
        for snap in reversed(snapshots):
            # Le(w,α) = CE(qα, p_w)
            # dot(∇_w Le, g_w) -> grad wrt α (2nd order)
            # đây là bản chậm, nhưng portable.
            logits_s = student(snap.x)  # NOTE: dùng w_cur, không đúng tuyệt đối với w(τ); fallback thôi
            if isinstance(teacher, MetaNet):
                _, hx = student(snap.x, return_h=True)
                q = teacher(hx.detach(), snap.y)
            else:
                q, _, _ = teacher(snap.x, snap.y)
            le = soft_cross_entropy(logits_s, q.detach())
            gw_dot = 0.0
            grads_le = torch.autograd.grad(le, list(student.parameters()), create_graph=True)
            for (name, _), gle in zip(student.named_parameters(), grads_le):
                if gle is None:
                    continue
                gw_dot = gw_dot + (gle * g_named[name]).sum()
            meta_proxy = meta_proxy + w_factor * gw_dot
            w_factor = w_factor * gamma

        meta_grads = torch.autograd.grad(meta_proxy, list(teacher.parameters()), retain_graph=False, allow_unused=True)
        # sign: meta_grad = -η * g_w^T H_wα; với H=-Jw^T Jα => +η * ...
        # autograd fallback đang làm gần đúng; giữ scale η_w để consistent
        meta_grads = [None if g is None else (lr_student * g) for g in meta_grads]
        return meta_grads

    # meta_grad_impl == "func" (khuyến nghị)
    try:
        from torch.func import functional_call, jvp
    except Exception as e:
        raise RuntimeError("torch.func not available. Install PyTorch>=2.0 or use --meta_grad_impl autograd") from e

    # function: params -> log_probs
    def f_logp(params, buffers, x):
        logits = functional_call(student, (params, buffers), (x,))
        return F.log_softmax(logits, dim=-1)

    meta_proxy = 0.0
    gamma = 1.0 - lr_student
    w_factor = 1.0

    # recent -> old (t, t-1, ...)
    for snap in reversed(snapshots):
        # JVP of logp wrt params in direction g_w
        # jvp returns (primals_out, tangents_out)
        _, jvp_out = jvp(lambda p: f_logp(p, snap.buffers, snap.x), (snap.params,), (g_named,))
        jvp_out = jvp_out.detach()  # treat as constant wrt teacher

        if isinstance(teacher, MetaNet):
            # MetaNet expects hx + y_tilde
            _, hx = student(snap.x, return_h=True)
            q = teacher(hx.detach(), snap.y)  # q: (B,C)
        else:
            q, _, _ = teacher(snap.x, snap.y)  # q: (B,C)

        # proxy scalar: mean over batch of sum_c q_c * jvp_c
        proxy_tau = (q * jvp_out).sum(dim=1).mean()
        meta_proxy = meta_proxy + w_factor * proxy_tau
        w_factor = w_factor * gamma

    meta_grads = torch.autograd.grad(meta_proxy, list(teacher.parameters()), retain_graph=False, allow_unused=True)
    meta_grads = [None if g is None else (lr_student * g) for g in meta_grads]
    return meta_grads

# -------------------------
# Teacher auxiliary losses (paper eq. (12): CE + BCE + META) :contentReference[oaicite:1]{index=1}
# -------------------------
def teacher_aux_losses_emlc(
    teacher: EMLCTeacher,
    x_clean: torch.Tensor,
    y_clean: torch.Tensor,
    bce_mode: str,
):
    # L_CE: train feature extractor + classifier on clean
    logits_t = teacher.forward_classifier(x_clean)
    loss_ce = F.cross_entropy(logits_t, y_clean)

    if bce_mode == "none":
        return loss_ce, torch.tensor(0.0, device=x_clean.device)

    # L_BCE: corrupt half batch, predict clean/corrupt via gate w
    B = x_clean.size(0)
    perm = torch.randperm(B, device=x_clean.device)
    half = B // 2
    idx_corrupt = perm[:half]
    idx_clean = perm[half:]

    y_in = y_clean.clone()
    target = torch.ones(B, device=x_clean.device)

    if bce_mode == "rand":
        # random wrong labels
        num_classes = teacher.num_classes
        r = torch.randint(0, num_classes, size=(half,), device=x_clean.device)
        # ensure != true
        r = torch.where(r == y_in[idx_corrupt], (r + 1) % num_classes, r)
        y_in[idx_corrupt] = r
        target[idx_corrupt] = 0.0

    elif bce_mode == "adv":
        # adversarial: strongest incorrect prediction from teacher classifier
        with torch.no_grad():
            p = F.softmax(logits_t, dim=-1)  # (B,C)
            p2 = p.clone()
            p2[torch.arange(B, device=x_clean.device), y_clean] = -1.0
            adv = p2.argmax(dim=-1)
        y_in[idx_corrupt] = adv[idx_corrupt]
        target[idx_corrupt] = 0.0
    else:
        raise ValueError("bce_mode must be none/rand/adv")

    w = teacher.forward_gate(x_clean, y_in)  # (B,)
    loss_bce = F.binary_cross_entropy(w, target)
    return loss_ce, loss_bce

# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    accm = AverageMeter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = student(x)
        accm.update(accuracy_top1(logits, y), n=x.size(0))
    student.train()
    return accm.avg

def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--gold_fraction", type=float, default=0.02)  # 1000/50000
    p.add_argument("--corruption_type", type=str, default="unif", choices=["unif", "flip", "flip2", "hierarchical"])
    p.add_argument("--corruption_prob", type=float, default=0.4)
    p.add_argument("--loader_style", type=str, default="mlc", choices=["mlc", "mwnet"])
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)

    # models
    p.add_argument("--teacher_type", type=str, default="emlc", choices=["emlc", "metanet"])
    # metanet args
    p.add_argument("--cls_dim", type=int, default=32)
    p.add_argument("--meta_hdim", type=int, default=128)
    p.add_argument("--skip", action="store_true")
    p.add_argument("--tie", action="store_true")
    p.add_argument("--sparsemax", action="store_true")

    # optimization
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr_student", type=float, default=0.1)
    p.add_argument("--lr_teacher", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # meta
    p.add_argument("--meta_interval", type=int, default=1)  # k
    p.add_argument("--meta_lambda", type=float, default=1.0)  # scale meta-grad
    p.add_argument("--meta_grad_impl", type=str, default="func", choices=["func", "autograd"])

    # teacher aux losses
    p.add_argument("--ce_lambda", type=float, default=1.0)
    p.add_argument("--bce_lambda", type=float, default=1.0)
    p.add_argument("--bce_mode", type=str, default="adv", choices=["none", "rand", "adv"])

    # misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.corruption_type == "hierarchical" and args.dataset != "cifar100":
        raise ValueError("hierarchical noise chỉ hỗ trợ cifar100")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gold_loader, silver_loader, test_loader, num_classes = make_cifar_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        gold_fraction=args.gold_fraction,
        corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type,
        batch_size=args.bs,
        num_workers=args.num_workers,
        loader_style=args.loader_style,
        seed=args.seed,
    )
    gold_iter = DataIterator(gold_loader)

    student = resnet32(num_classes=num_classes).to(device)

    if args.teacher_type == "emlc":
        teacher = EMLCTeacher(num_classes=num_classes, label_emb_dim=args.cls_dim, gate_hidden=args.meta_hdim).to(device)
    else:
        # MetaNet uses hx from student (64-d)
        teacher = MetaNet(hx_dim=64, cls_dim=args.cls_dim, h_dim=args.meta_hdim, num_classes=num_classes, args=args).to(device)

    opt_s = torch.optim.SGD(student.parameters(), lr=args.lr_student,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    opt_t = torch.optim.SGD(teacher.parameters(), lr=args.lr_teacher,
                            momentum=args.momentum, weight_decay=0.0)

    # simple cosine schedule for student
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=args.epochs)

    best_acc = 0.0
    snapshots: List[Snapshot] = []

    for epoch in range(args.epochs):
        student.train()
        teacher.train()

        loss_s_meter = AverageMeter()
        loss_t_meter = AverageMeter()
        acc_silver_meter = AverageMeter()

        for step, (x_s, y_s) in enumerate(silver_loader):
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)

            # lấy batch clean (gold/meta)
            x_g, y_g = next(gold_iter)
            x_g = x_g.to(device, non_blocking=True)
            y_g = y_g.to(device, non_blocking=True)

            # snapshot student state BEFORE update (w(t))
            p_state, b_state = _clone_state(student)
            snapshots.append(Snapshot(x=x_s, y=y_s, params=p_state, buffers=b_state))

            # -------- Student update: minimize Le(w, α) on silver --------
            with torch.no_grad():
                if args.teacher_type == "emlc":
                    q, _, _ = teacher(x_s, y_s)
                else:
                    _, hx = student(x_s, return_h=True)
                    q = teacher(hx.detach(), y_s)
                q = q.detach()

            logits_s = student(x_s)
            loss_s = soft_cross_entropy(logits_s, q)

            opt_s.zero_grad(set_to_none=True)
            loss_s.backward()
            opt_s.step()

            loss_s_meter.update(loss_s.item(), n=x_s.size(0))
            acc_silver_meter.update(accuracy_top1(logits_s.detach(), y_s), n=x_s.size(0))

            # -------- Teacher meta update every k steps --------
            if len(snapshots) == args.meta_interval:
                lr_now = opt_s.param_groups[0]["lr"]

                # meta grads
                meta_grads = compute_meta_grads_fpmg(
                    student=student,
                    teacher=teacher,
                    snapshots=snapshots,
                    x_clean=x_g,
                    y_clean=y_g,
                    lr_student=lr_now,
                    meta_interval=args.meta_interval,
                    meta_grad_impl=args.meta_grad_impl,
                )

                opt_t.zero_grad(set_to_none=True)

                # aux losses for teacher
                if args.teacher_type == "emlc":
                    loss_ce, loss_bce = teacher_aux_losses_emlc(teacher, x_g, y_g, args.bce_mode)
                    aux = args.ce_lambda * loss_ce + args.bce_lambda * loss_bce
                else:
                    # MetaNet không có x -> chỉ dùng meta-grad (hoặc bạn có thể tự thêm regularizer)
                    aux = torch.tensor(0.0, device=device)

                if aux.requires_grad:
                    aux.backward()

                # add meta grads
                with torch.no_grad():
                    for p_t, g_t in zip(teacher.parameters(), meta_grads):
                        if g_t is None:
                            continue
                        if p_t.grad is None:
                            p_t.grad = args.meta_lambda * g_t
                        else:
                            p_t.grad.add_(args.meta_lambda * g_t)

                opt_t.step()

                loss_t_meter.update(float(aux.item()), n=1)
                snapshots.clear()

        sch_s.step()

        test_acc = evaluate(student, test_loader, device)
        best_acc = max(best_acc, test_acc)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss_s={loss_s_meter.avg:.4f} "
            f"loss_t={loss_t_meter.avg:.4f} "
            f"acc_silver={acc_silver_meter.avg:.2f} "
            f"test_acc={test_acc:.2f} "
            f"best={best_acc:.2f} "
            f"lr={opt_s.param_groups[0]['lr']:.5f}"
        )

if __name__ == "__main__":
    main()
