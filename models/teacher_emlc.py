import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cifar import resnet32

class EMLCTeacher(nn.Module):
    """
    Teacher theo EMLC:
      - feature extractor + classifier => p(y|x)
      - embedding(y_tilde) + hidden(x) => gate w in (0,1)
      - q(y|x,y_tilde) = w * onehot(y_tilde) + (1-w) * p(y|x)
    """
    def __init__(self, num_classes: int, label_emb_dim=32, gate_hidden=128):
        super().__init__()
        self.num_classes = num_classes

        # Teacher feature extractor/classifier (độc lập với student)
        self.backbone = resnet32(num_classes=num_classes)  # trả logits + hidden 64

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        self.gate = nn.Sequential(
            nn.Linear(64 + label_emb_dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
        )

    def forward_classifier(self, x):
        logits = self.backbone(x, return_h=False)
        return logits

    def forward_gate(self, x, y_tilde):
        _, h = self.backbone(x, return_h=True)  # h: (B,64)
        z = self.label_emb(y_tilde)
        g_in = torch.cat([h, z], dim=-1)
        w = torch.sigmoid(self.gate(g_in)).squeeze(1)  # (B,)
        return w

    def forward(self, x, y_tilde):
        logits, h = self.backbone(x, return_h=True)
        p = F.softmax(logits, dim=-1)  # p(y|x)

        z = self.label_emb(y_tilde)
        g_in = torch.cat([h, z], dim=-1)
        w = torch.sigmoid(self.gate(g_in))  # (B,1)

        y_onehot = F.one_hot(y_tilde, self.num_classes).type_as(p)  # (B,C)
        q = w * y_onehot + (1.0 - w) * p
        return q, w.squeeze(1), p
