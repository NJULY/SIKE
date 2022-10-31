import os
import numpy as np

import torch
from models import Knowformer


def read_all_align_dicts(data_path):
    dbp2wiki, wiki2dbp = _read_align_dict(os.path.join(data_path, 'dbp_wd_links.txt'))
    wiki2yg, yg2wiki = _read_align_dict(os.path.join(data_path, 'wd_yg_links.txt'))
    dbp2yg, yg2dbp = _read_align_dict(os.path.join(data_path, 'dbp_yg_links.txt'))
    return {
        'DBpedia15K': {
            'Wikidata15K': dbp2wiki,
            'Yago15K': dbp2yg,
        },
        'Wikidata15K': {
            'DBpedia15K': wiki2dbp,
            'Yago15K': wiki2yg,
        },
        'Yago15K': {
            'DBpedia15K': yg2dbp,
            'Wikidata15K': yg2wiki,
        }
    }

def _read_align_dict(data_path):
    A2B, B2A = dict(), dict()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ent1, ent2 = line.strip().split('\t')
            A2B[ent1] = ent2
            B2A[ent2] = ent1
    return A2B, B2A


def save_model(config: dict, model: torch.nn.Module, model_path: str):
    torch.save({'config': config, 'model': model.state_dict()}, model_path)


def load_model(model_path: str, device: str):
    print(f'Loading Knowformer from {model_path}')
    state_dict = torch.load(model_path, map_location=device)
    model_config = state_dict['config']
    model = Knowformer(model_config)
    model.load_state_dict(state_dict['model'])
    return model_config, model


def swa(output_path, device):
    files = os.listdir(output_path)
    files = [file_name for file_name in files if file_name.startswith('epoch_')]

    model_config = None
    model_dicts = list()
    for file_name in files:
        state_dict = torch.load(os.path.join(output_path, file_name), map_location=device)
        model_config = state_dict['config']
        model_dicts.append(state_dict['model'])

    avg_model_dict = dict()
    for k in model_dicts[0]:
        sum_param = None
        for dit in model_dicts:
            if sum_param is None:
                sum_param = dit[k]
            else:
                sum_param += dit[k]
        avg_param = sum_param / len(model_dicts)
        avg_model_dict[k] = avg_param
    model = Knowformer(model_config)
    model.load_state_dict(avg_model_dict)

    save_model(model_config, model, os.path.join(output_path, 'avg.bin'))


def get_scores(rank: list, loss=None):
    rank = np.array(rank)
    hits1 = round(np.mean(rank <= 1) * 100, 2)
    hits3 = round(np.mean(rank <= 3) * 100, 2)
    hits10 = round(np.mean(rank <= 10) * 100, 2)
    mrr = round(float(np.mean(1. / rank)), 4)
    loss = round(loss, 2)
    return {'loss': loss, 'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10, 'MRR': mrr}


def score2str(score):
    loss = score['loss']
    hits1 = score['hits@1']
    hits3 = score['hits@3']
    hits10 = score['hits@10']
    mrr = score['MRR']
    return f'loss: {loss}, hits@1: {hits1}, hits@3: {hits3}, hits@10: {hits10}, MRR: {mrr}'


def save_results(triples, ranks):
    results = list()
    batch_size = len(triples)
    for i in range(batch_size):
        h, r, t = triples[i]
        rank = ranks[i]
        results.append((h, r, t, rank))
    return results

