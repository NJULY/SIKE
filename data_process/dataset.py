import os
import copy
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def read_triples(file_path):
    triples = list()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = line.strip().split()
            triples.append((h, r, t))
    return triples

class KGCDataset(Dataset):
    def __init__(self, data: list):
        super(KGCDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

