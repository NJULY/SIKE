from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import random


class BertAdapter(nn.Module):
    def __init__(self, input_size=768, intermediate_size=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, input_size)
        self.activation = torch.nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.xavier_normal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))

        return x + h

    def squash(self, input_tensor, dim=-1, epsilon=1e-16):
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        squared_norm = squared_norm + epsilon
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)

