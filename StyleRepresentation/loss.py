# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negatives sample
    """

    def __init__(self, mode='cosine', margin=None):
        super(TripletLoss, self).__init__()
        self.l2 = nn.PairwiseDistance(p=2)
        self.mode = mode
        if margin is None:
            if mode == 'cosine':
                self.margin = 0.4
            elif mode == 'l2':
                self.margin = 1.0
        else:
            self.margin = margin

    def forward(self, anchor, positive, negatives, size_average=True):
        if anchor.size()[1] == 1 and positive.size()[1] == 1 and negatives.size()[1] == 1:
            anchor = anchor.squeeze(1)
            positive = positive.squeeze(1)
            negatives = negatives.squeeze(1)
        
        if self.mode == 'l2':
            distance = self.l2(anchor, positive) - self.l2(anchor, negatives) + self.margin
            losses = F.relu(distance)
        elif self.mode == 'cosine':
            negatives_num = negatives.size(0)
            distance = torch.zeros(negatives_num)
            for i in range(negatives_num):
                distance[i] = -torch.dot(anchor[0], negatives[i]) + self.margin
            distance = torch.dot(anchor[0], positive[0]) + distance
            losses = F.relu(distance)
        
        return losses.mean() if size_average else losses.sum() 