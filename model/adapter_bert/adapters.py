from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import random


class BottleNeckAdapter(nn.Module):
    def __init__(self, input_size=768, intermediate_size=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, input_size)
        self.activation = torch.nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x))) + x

