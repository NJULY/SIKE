import os
import numpy as np

import torch
from model import Knowformer


def read_all_align_dicts(data_path: str):
    if data_path.endswith('DWY15K'):
        return read_dwy_align_dicts(data_path)
    elif data_path.endswith('YFW'):
        return read_yfw_align_dicts(data_path)
    else:
        assert 0

def read_yfw_align_dicts(data_path: str):
    fb2wn, wn2fb = _read_align_dict(os.path.join(data_path, 'links', 'fb_wn_links.txt'))
    fb2yg, yg2fb = _read_align_dict(os.path.join(data_path, 'links', 'fb_yg_links.txt'))
    yg2wn, wn2yg = _read_align_dict(os.path.join(data_path, 'links', 'yg_wn_links.txt'))
    return {
        'YAGO3': {'FB15K237': yg2fb, 'WN18RR': yg2wn},
        'FB15K237': {'YAGO3': fb2yg, 'WN18RR': fb2wn},
        'WN18RR': {'YAGO3': wn2yg, 'FB15K237': wn2fb},
    }

def read_dwy_align_dicts(data_path):
    dbp2wiki, wiki2dbp = _read_align_dict(os.path.join(data_path, 'links', 'dbp_wd_links.txt'))
    wiki2yg, yg2wiki = _read_align_dict(os.path.join(data_path, 'links', 'wd_yg_links.txt'))
    dbp2yg, yg2dbp = _read_align_dict(os.path.join(data_path, 'links', 'dbp_yg_links.txt'))
    yg2fb, fb2yg = _read_align_dict(os.path.join(data_path, 'links', 'yg_fb_links.txt'))
    dbp2fb, fb2dbp = _read_align_dict(os.path.join(data_path, 'links', 'dbp_fb_links.txt'))
    wiki2fb, fb2wiki = _read_align_dict(os.path.join(data_path, 'links', 'wk_fb_links.txt'))
    return {
        'DBpedia15K': {'Wikidata15K': dbp2wiki, 'Yago15K': dbp2yg, 'FB15K237': dbp2fb},
        'Wikidata15K': {'DBpedia15K': wiki2dbp, 'Yago15K': wiki2yg, 'FB15K237': wiki2fb},
        'Yago15K': {'DBpedia15K': yg2dbp, 'Wikidata15K': yg2wiki, 'FB15K237': yg2fb},
        'FB15K237': {'DBpedia15K': fb2dbp, 'Wikidata15K': fb2wiki, 'Yago15K': fb2yg}
    }


def _read_align_dict(data_path):
    """
    读取实体对齐的数据, 每一行格式为: ent1 \t ent2 \t 余弦相似度
    :param data_path:
    :return:
    """
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


def swa(output_path, device):
    """
    we save the best 20 models, load these model and average parameters
    :param output_path:
    :param device:
    :return:
    """
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

