import os
import numpy as np
import torch

def read_all_align_dicts(data_path):
    dbp2wiki, wiki2dbp = _read_align_dict(os.path.join(data_path, 'dbp_wd_links.txt'))
    wiki2yg, yg2wiki = _read_align_dict(os.path.join(data_path, 'wd_yg_links.txt'))
    dbp2yg, yg2dbp = _read_align_dict(os.path.join(data_path, 'dbp_yg_links.txt'))
    yg2fb, fb2yg = _read_align_dict(os.path.join(data_path, 'yg_fb_links.txt'))
    dbp2fb, fb2dbp = _read_align_dict(os.path.join(data_path, 'dbp_fb_links.txt'))
    wiki2fb, fb2wiki = _read_align_dict(os.path.join(data_path, 'wk_fb_links.txt'))
    return {
        'DBpedia15K': {'Wikidata15K': dbp2wiki, 'Yago15K': dbp2yg, 'FB15K237': dbp2fb},
        'Wikidata15K': {'DBpedia15K': wiki2dbp, 'Yago15K': wiki2yg, 'FB15K237': wiki2fb},
        'Yago15K': {'DBpedia15K': yg2dbp, 'Wikidata15K': yg2wiki, 'FB15K237': yg2fb},
        'FB15K237': {'DBpedia15K': fb2dbp, 'Wikidata15K': fb2wiki, 'Yago15K': fb2yg}
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

