import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaNet(nn.Module):
    """
    Teacher kiểu MLC: q_alpha(y | hx, y_tilde)
    """
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes, args):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.in_class = num_classes
        self.cls_emb = nn.Embedding(self.in_class, cls_dim)

        in_dim = hx_dim + cls_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, num_classes + int(args.skip), bias=(not args.tie)),
        )

        if args.sparsemax:
            from sparsemax import Sparsemax
            self.sparsemax = Sparsemax(-1)

        self._init_weights()

        if args.tie:
            print("Tying cls emb to output cls weight")
            self.net[-1].weight = self.cls_emb.weight

        self.alpha = torch.zeros(1)

    def _init_weights(self):
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)
        nn.init.xavier_normal_(self.net[6].weight)
        self.net[0].bias.data.zero_()
        self.net[3].bias.data.zero_()
        if not self.args.tie:
            self.net[6].bias.data.zero_()

    def get_alpha(self):
        return self.alpha if self.args.skip else torch.zeros(1, device=self.alpha.device)

    def forward(self, hx, y_tilde):
        y_emb = self.cls_emb(y_tilde)
        hin = torch.cat([hx, y_emb], dim=-1)
        logit = self.net(hin)

        if self.args.skip:
            alpha = torch.sigmoid(logit[:, self.num_classes:])
            self.alpha = alpha.mean().detach()
            logit = logit[:, :self.num_classes]

        if self.args.sparsemax:
            out = self.sparsemax(logit)
        else:
            out = F.softmax(logit, dim=-1)

        if self.args.skip:
            out = (1.0 - alpha) * out + alpha * F.one_hot(y_tilde, self.num_classes).type_as(out)

        return out
