import torch.nn as nn
import math
import torch
from torch.autograd import Variable
import numpy as np
import pickle
import torch.nn.functional as F

class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        # self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_dim, n_classes, bias=True)

    def forward(self, x):
        # x = self.dropout(x)
        return self.linear(x)

class LinearRotateHead(nn.Module):
    def __init__(self, in_dim=512, n_classes=100):
        super(LinearRotateHead, self).__init__()

        self.rotate_classifier = nn.Sequential(
            nn.Linear(in_dim, 4)
        )
        self.cls_classifier = nn.Sequential(
            nn.Linear(in_dim, n_classes)
        )


    def forward(self, x, use_cls=True):
        if use_cls:
            out = self.cls_classifier(x)
        else:
            out = self.rotate_classifier(x)
        return out


