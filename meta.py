# meta.py
import copy
import torch
import torch.nn.functional as F

def soft_cross_entropy(logits, soft_targets):
    # CE(q, p) = -sum q * log softmax(p)
    logp = F.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()

def teacher_ce_loss(teacher, x_clean, y_clean):
    # CE trains backbone+classifier (embedding+MLP not used here) :contentReference[oaicite:7]{index=7}
    logits = teacher.backbone(x_clean, return_h=False)
    return F.cross_entropy(logits, y_clean)

def teacher_bce_loss(teacher, x_clean, y_clean, strategy: str):
    """
    Corrupt half of batch; BCE teaches enhancer to predict clean vs corrupted label. :contentReference[oaicite:8]{index=8}
    strategy: 'none'|'rand'|'adv'
    """
    strategy = strategy.lower()
    if strategy == "none":
        return torch.tensor(0.0, device=x_clean.device)

    b = x_clean.size(0)
    half = b // 2
    idx = torch.randperm(b, device=x_clean.device)
    corrupt_idx = idx[:half]
    clean_idx   = idx[half:]

    y_in = y_clean.clone()
    is_clean = torch.ones((b, 1), device=x_clean.device)
    is_clean[corrupt_idx] = 0.0

    if strategy == "rand":
        # random labels
        num_classes = teacher.num_classes
        y_in[corrupt_idx] = torch.randint(0, num_classes, (half,), device=x_clean.device)
    elif strategy == "adv":
        adv = teacher.adversarial_corrupt_labels(x_clean[corrupt_idx], y_clean[corrupt_idx])
        y_in[corrupt_idx] = adv
    else:
        raise ValueError(f"Unknown bce strategy: {strategy}")

    w, _, _ = teacher.retain_conf(x_clean, y_in)
    return F.binary_cross_entropy(w, is_clean)

def compute_teacher_meta_grads(
    student,
    teacher,
    x_meta,
    y_meta,
    stored_steps,
    eta_w: float,
    gamma_w: float,
    temperature: float,
    device
):
    """
    Implements paper-style k-step approximation (Prop.2) but in a clear, robust way.
    We keep teacher fixed across the last k steps (buffer cleared after each teacher update).
    """
    # g_w = ∇_w L(meta) evaluated at current student weights w^{t+1}
    meta_logits = student(x_meta)
    L_meta = F.cross_entropy(meta_logits, y_meta)
    g_w = torch.autograd.grad(L_meta, student.parameters(), retain_graph=False, create_graph=False)

    # snapshot current student weights to restore after loading old states
    cur_sd = copy.deepcopy(student.state_dict())

    meta_grads = [torch.zeros_like(p, device=device) for p in teacher.parameters()]
    dw = 1.0

    # iterate τ = t, t-1, ..., t-k+1  (stored_steps already ordered oldest->newest)
    for step in reversed(stored_steps):
        student.load_state_dict(step["w"], strict=True)

        x_s = step["x"].to(device, non_blocking=True)
        y_s = step["y"].to(device, non_blocking=True)

        q, _, _, _ = teacher.corrected_targets(x_s, y_s, temperature=temperature)
        logits_s = student(x_s)
        Le = soft_cross_entropy(logits_s, q)

        # grad_w Le (create_graph=True because we differentiate it w.r.t teacher params)
        grad_w_Le = torch.autograd.grad(Le, student.parameters(), create_graph=True, retain_graph=True)

        dot = 0.0
        for gw_i, gLe_i in zip(g_w, grad_w_Le):
            dot = dot + (gw_i * gLe_i).sum()

        grads_alpha = torch.autograd.grad(dot, teacher.parameters(), retain_graph=True, create_graph=False)

        # Proposition 2: ∇α ≈ -ηw * Σ γ^{t-τ} * g_w^T H_{wα}(τ)
        for i, ga in enumerate(grads_alpha):
            meta_grads[i] = meta_grads[i] + (-eta_w * dw) * ga

        dw *= gamma_w

    # restore student weights
    student.load_state_dict(cur_sd, strict=True)
    return meta_grads, float(L_meta.detach().cpu().item())
