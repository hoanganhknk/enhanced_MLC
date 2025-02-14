import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from gensim.models import KeyedVectors
label_names = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
embedding_model = KeyedVectors.load_word2vec_format("/kaggle/input/word2vec/pytorch/default/1/GoogleNews-vectors-negative300.bin", binary=True)
def embedding_label(label_text):
    return torch.tensor(embedding_model[label_text], device = 'cuda', requires_grad=True)
class MetaNet(nn.Module):
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes, args):
        super().__init__()

        self.args = args
        self.num_classes = num_classes        
        self.in_class = self.num_classes 
        self.hdim = h_dim
        in_dim = hx_dim + cls_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Dropout(p = 0.3),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.BatchNorm1d(self.hdim), 
            nn.Linear(self.hdim, num_classes + int(self.args.skip), bias=(not self.args.tie)),
        )

        if self.args.sparsemax:
            from sparsemax import Sparsemax
            self.sparsemax = Sparsemax(-1)

        self.init_weights()

        
    def init_weights(self):
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)
        nn.init.xavier_normal_(self.net[6].weight)

        self.net[0].bias.data.zero_()
        self.net[3].bias.data.zero_()

        if not self.args.tie:
            assert self.in_class == self.num_classes, 'In and out classes conflict!'
            self.net[6].bias.data.zero_()

    def get_alpha(self):
        return self.alpha if self.args.skip else torch.zeros(1)

    def forward(self, hx, y):
        bs = hx.size(0)
        # Chuyển label số thành chữ
        label_texts = [label_names[i.item()] for i in y]
        
        y_emb = embedding_label(label_texts)
        hin = torch.cat([hx, y_emb], dim=-1)

        logit = self.net(hin)

        if self.args.skip:
            alpha = torch.sigmoid(logit[:, self.num_classes:])
            self.alpha = alpha.mean()
            logit = logit[:, :self.num_classes]

        if self.args.sparsemax:
            out = self.sparsemax(logit) # test sparsemax
        else:
            out = F.softmax(logit, -1)

        if self.args.skip:
            out = (1.-alpha) * out + alpha * F.one_hot(y, self.num_classes).type_as(out)

        return out