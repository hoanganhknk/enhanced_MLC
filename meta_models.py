# meta_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherEnhancer(nn.Module):
    """
    label embedding + feature -> MLP -> sigmoid weight w in (0,1)
    Paper: 1 hidden layer, 128 units (Table 5). :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, num_classes: int, feat_dim: int = 64, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor, noisy_y: torch.Tensor):
        z = self.emb(noisy_y)               # (B, emb_dim)
        h = torch.cat([feat, z], dim=1)     # (B, feat_dim+emb_dim)
        w = torch.sigmoid(self.mlp(h))      # (B, 1)
        return w

class Teacher(nn.Module):
    """
    Teacher = independent backbone+classifier + enhancer producing w.
    Corrected target: q = w * one_hot(noisy_y) + (1-w) * softmax(teacher_logits)
    Matches Fig.3 and Appendix D. :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, backbone: nn.Module, num_classes: int, feat_dim: int = 64, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.enhancer = TeacherEnhancer(num_classes, feat_dim=feat_dim, emb_dim=emb_dim, hidden=hidden)

    @torch.no_grad()
    def _one_hot(self, y: torch.Tensor):
        return F.one_hot(y, num_classes=self.num_classes).float()

    def forward(self, x, return_h: bool = False):
        # backbone MUST support return_h
        return self.backbone(x, return_h=return_h)

    def retain_conf(self, x: torch.Tensor, noisy_y: torch.Tensor):
        logits, feat = self.backbone(x, return_h=True)
        w = self.enhancer(feat, noisy_y)
        return w, logits, feat

    def corrected_targets(self, x: torch.Tensor, noisy_y: torch.Tensor, temperature: float = 1.0):
        w, logits, feat = self.retain_conf(x, noisy_y)
        p = F.softmax(logits / max(temperature, 1e-8), dim=1)  # (B,C)
        y_oh = F.one_hot(noisy_y, num_classes=self.num_classes).float()
        q = w * y_oh + (1.0 - w) * p
        q = q / (q.sum(dim=1, keepdim=True) + 1e-12)
        return q, w, logits, feat

    @torch.no_grad()
    def adversarial_corrupt_labels(self, x: torch.Tensor, true_y: torch.Tensor):
        """
        Adversarial corruption for BCE objective:
        replace label with strongest incorrect prediction of teacher classifier.
        """
        logits = self.backbone(x, return_h=False)
        probs = F.softmax(logits, dim=1)
        probs[torch.arange(true_y.size(0), device=true_y.device), true_y] = -1.0
        adv = probs.argmax(dim=1)
        return adv
