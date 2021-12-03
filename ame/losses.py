import torch
import torch.nn as nn
import torch.nn.functional as F

import ame.functional as af

from loguru import logger


class JaccardLoss(nn.Module):

    def __init__(self, eps=1., ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.ignore_channels = ignore_channels

    def forward(self, inputs, targets):
        return 1 - af.jaccard(
            inputs, targets,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

