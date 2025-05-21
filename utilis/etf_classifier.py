import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

        self.LWS = LWS
        self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x