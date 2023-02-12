import os
import json
import random
import copy
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

"""
codes in this file are used for entity alignment
"""

def degree(data_folder: str):
    ent2degree = dict()
    triples = list()
    for file_name in ['train.txt', 'valid.txt', 'test.txt']:
        with open(os.path.join(data_folder, file_name), 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((h, r, t))
                if h not in ent2degree:
                    ent2degree[h] = 0
                if t not in ent2degree:
                    ent2degree[t] = 0
                ent2degree[h] += 1
                ent2degree[t] += 1
    print(len(triples), len(ent2degree))
    print(sum(ent2degree.values()) / len(ent2degree))

def read_aligns(data_path):
    align_dict1 = dict()
    align_dict2 = dict()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tokens = line.strip().split('\t')
            assert len(tokens) == 2 or len(tokens) == 3
            ent1, ent2 = tokens[0], tokens[1]
            align_dict1[ent1] = ent2
            align_dict2[ent2] = ent1
    return align_dict1, align_dict2

def read_ents(json_path: str):
    ents = json.load(open(json_path, 'r', encoding='utf-8'))
    id2ent = {i: e for i, e in enumerate(ents)}
    dataloader = get_dataloader(ents)
    return ents, id2ent, dataloader

class MyDataset(Dataset):
    def __init__(self, data: list):
        super(MyDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(ents: dict):
    prompts = [ents[ent]['name'] for ent in ents]
    dataset = MyDataset(prompts)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=128, shuffle=False)
    return dataloader

def collate_fn(batch_data):
    encoded_data = tokenizer(batch_data, padding=True, truncation=True, max_length=128)
    input_ids = torch.tensor(encoded_data['input_ids'])
    token_type_ids = torch.tensor(encoded_data['token_type_ids'])
    attention_mask = torch.tensor(encoded_data['attention_mask'])

    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}

def ent_align(model, dataloader1, dataloader2, id2ent1, id2ent2, threshold: float):
    def get_cls_embeds(dataloader):
        cls = []
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cls_embeds = output.last_hidden_state[:, 0, :]  # pooler_output
            cls.append(cls_embeds)
        return torch.cat(cls, dim=0)

    cls1 = get_cls_embeds(dataloader1)  # (ent_num, hidden_size)
    cls2 = get_cls_embeds(dataloader2)  # (ent_num, hidden_size)
    cos = torch.nn.CosineSimilarity(dim=1)

    sim1 = torch.stack([cos(cls, cls2) for cls in tqdm(cls1)], dim=0)
    sim2 = torch.stack([cos(cls, cls1) for cls in tqdm(cls2)], dim=0)

    match1 = {ent_idx: idx.item() for ent_idx, idx in enumerate(torch.argmax(sim1, dim=-1))
              if sim1[ent_idx, idx.item()] >= threshold}
    match2 = {idx.item(): ent_idx for ent_idx, idx in enumerate(torch.argmax(sim2, dim=-1))
              if sim2[ent_idx, idx.item()] >= threshold}

    align_dit = {ent: match1[ent] for ent in match1 if ent in match2 and match1[ent] == match2[ent]}
    res = list()
    for id1 in align_dit:
        id2 = align_dit[id1]
        ent1, ent2 = id2ent1[id1], id2ent2[id2]
        sim = sim1[id1, id2].item()
        res.append((ent1, ent2, sim))
    return res

def get_score(dit1: dict, dit2: dict):
    tp_cnt = 0
    for ent1 in dit1:
        if ent1 in dit2:
            if dit1[ent1] == dit2[ent1]:
                tp_cnt += 1

    p = tp_cnt / len(dit2)
    r = tp_cnt / len(dit1)
    f1 = 2 * p * r / (p + r)
    print(f'预测正确: {tp_cnt}; 实际对齐数量: {len(dit1)}; 预测对齐数量: {len(dit2)}')
    print(f'P: {round(p, 4)}; R: {round(r, 4)}; F1: {round(f1, 4)}')

def save_links(data_path, align_list):
    with open(data_path, 'w', encoding='utf-8') as f:
        for ent1, ent2, sim in align_list:
            f.write(f'{ent1}\t{ent2}\t{sim}\n')


if __name__ == '__main__':
    device = 'cuda:0'
    bert_path = '../../bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BertModel.from_pretrained(bert_path).to(device)
    for p in bert.parameters():
        p.requires_grad = False

    dbp_ents, dbp_id2ent, dbp_dataloader = read_ents('./DWY15K/DBpedia15K/entity.json')
    wiki_ents, wiki_id2ent, wiki_dataloader = read_ents('./DWY15K/Wikidata15K/entity.json')
    yg_ents, yg_id2ent, yg_dataloader = read_ents('./DWY15K/Yago15K/entity.json')
    fb_ents, fb_id2ent, fb_dataloader = read_ents('./DWY15K/FB15K237/entity.json')
    
    t = 0.999

    dbp2wiki_test = ent_align(bert, dbp_dataloader, wiki_dataloader, dbp_id2ent, wiki_id2ent, t)
    save_links('dbp_wd_links.txt', dbp2wiki_test)

    wiki2yg_test = ent_align(bert, wiki_dataloader, yg_dataloader, wiki_id2ent, yg_id2ent, t)
    save_links('wd_yg_links.txt', wiki2yg_test)

    dbp2yg_test = ent_align(bert, dbp_dataloader, yg_dataloader, dbp_id2ent, yg_id2ent, t)
    save_links('dbp_yg_links.txt', dbp2yg_test)

    dbp2fb_test = ent_align(bert, dbp_dataloader, fb_dataloader, dbp_id2ent, fb_id2ent, t)
    save_links('dbp_fb_links.txt', dbp2fb_test)

    wiki2fb_test = ent_align(bert, wiki_dataloader, fb_dataloader, wiki_id2ent, fb_id2ent, t)
    save_links('wiki_fb_links.txt', wiki2fb_test)

    yg2fb_test = ent_align(bert, yg_dataloader, fb_dataloader, yg_id2ent, fb_id2ent, t)
    save_links('yg_fb_links.txt', yg2fb_test)

