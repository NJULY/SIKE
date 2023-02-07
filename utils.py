import os
import numpy as np

import torch
from model import Knowformer


def read_all_align_dicts(data_path: str):
    if data_path.endswith('DWY15K'):
        return read_dwy_align_dicts(data_path)
    else:
        assert 0

def read_dwy_align_dicts(data_path):
    dbp2wiki, wiki2dbp = _read_align_dict(os.path.join(data_path, 'links', 'dbp_wd_links.txt'))
    wiki2yg, yg2wiki = _read_align_dict(os.path.join(data_path, 'links', 'wd_yg_links.txt'))
    dbp2yg, yg2dbp = _read_align_dict(os.path.join(data_path, 'links', 'dbp_yg_links.txt'))
    return {
        'DBpedia15K': {'Wikidata15K': dbp2wiki, 'Yago15K': dbp2yg},
        'Wikidata15K': {'DBpedia15K': wiki2dbp, 'Yago15K': wiki2yg},
        'Yago15K': {'DBpedia15K': yg2dbp, 'Wikidata15K': yg2wiki},
    }


def _read_align_dict(data_path):
    src1_to_src2, src2_to_src1 = dict(), dict()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tokens = line.strip().split('\t')
            if len(tokens) == 2:
                ent1, ent2 = tokens[0], tokens[1]
                src1_to_src2[ent1] = {'ent': ent2, 'cos': 1.0}
                src2_to_src1[ent2] = {'ent': ent1, 'cos': 1.0}
            elif len(tokens) == 3:
                ent1, ent2, cos = tokens[0], tokens[1], float(tokens[2])
                src1_to_src2[ent1] = {'ent': ent2, 'cos': cos}
                src2_to_src1[ent2] = {'ent': ent1, 'cos': cos}
            else:
                assert 0

    return src1_to_src2, src2_to_src1


def save_model(config: dict, model: torch.nn.Module, model_path: str):
    torch.save({'config': config, 'model': model.state_dict()}, model_path)


def load_model(model_path: str, device: str):
    # load the trained Knowformer model
    print(f'Loading Knowformer from {model_path}')
    state_dict = torch.load(model_path, map_location=device)
    model_config = state_dict['config']
    model = Knowformer(model_config)
    model.load_state_dict(state_dict['model'])
    return model_config, model


def get_scores(rank: list, loss=None):
    rank = np.array(rank)
    hits1 = round(np.mean(rank <= 1) * 100, 2)
    hits3 = round(np.mean(rank <= 3) * 100, 2)
    hits10 = round(np.mean(rank <= 10) * 100, 2)
    mrr = round(float(np.mean(1. / rank)), 4)
    # mr = round(float(np.mean(rank)), 2)
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

