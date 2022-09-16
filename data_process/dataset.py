import os
import copy
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


# nothing special in this class
class KGCDataset(Dataset):
    def __init__(self, data: list):
        super(KGCDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


class StructureDataModule:
    def __init__(self, args: dict):
        # 0. some variables used in this class
        self.task = args['task']
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']

        self.add_neighbors = args['add_neighbors']
        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.no_relation_token = args['no_relation_token']

        # 1. read entities and relations from files
        self.entities, self.relations = self._read_entities_and_relations()
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}')

        # 2 construct the vocab
        self.vocab, struc_offset = self._get_vocab()
        args.update(struc_offset)
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = struc_offset['struc_relation_end_idx'] - struc_offset['struc_relation_begin_idx']

        # 3 read dataset
        self.lines = self._read_lines()  # {'train': [(h,r,t),...], 'dev': [], 'test': []}
        self.neighbors = self._get_neighbors()  # {ent: {text_prompt: [], struc_prompt: []}, ...}
        self.entity_filter = self._get_entity_filter()

        examples = self._create_examples()
        self.train_ds = KGCDataset(examples['train'])
        self.dev_ds = KGCDataset(examples['dev'])
        self.test_ds = KGCDataset(examples['test'])

    def _read_entities_and_relations(self):
        ent_path = os.path.join(self.data_path, 'entities.txt')
        ents = dict()
        with open(ent_path, 'r', encoding='utf-8') as f:
            for idx, ent in enumerate(f.readlines()):
                ents[ent.strip()] = {'token_id': idx}

        rel_path = os.path.join(self.data_path, 'relations.txt')
        rels = list()
        with open(rel_path, 'r', encoding='utf-8') as f:
            for idx, rel in enumerate(f.readlines()):
                rels.append(rel.strip())

        return ents, rels

    def _get_vocab(self):
        tokens = ['[PAD]', '[MASK]', '[SEP]', self.no_relation_token]
        ent_names = [e for e in self.entities]
        rel_names = []
        for r in self.relations:
            rel_names += [r, f'{r}_reverse']

        entity_begin_idx = len(tokens)
        entity_end_idx = len(tokens) + len(ent_names)
        relation_begin_idx = len(tokens) + len(ent_names)
        relation_end_idx = len(tokens) + len(ent_names) + len(rel_names)

        tokens = tokens + ent_names + rel_names
        vocab = {token: idx for idx, token in enumerate(tokens)}

        return vocab, {
            'struc_entity_begin_idx': entity_begin_idx,
            'struc_entity_end_idx': entity_end_idx,
            'struc_relation_begin_idx': relation_begin_idx,
            'struc_relation_end_idx': relation_end_idx,
        }

    def _read_lines(self):
        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'valid.txt'),
            'test': os.path.join(self.data_path, 'test.txt')
        }

        lines = dict()
        for mode in data_paths:
            data = list()
            with open(data_paths[mode], 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = str(line).strip().split('\t')
                    data.append((h, r, t))

            lines[mode] = data

        return lines

    def _get_neighbors(self):
        mask_token = '[MASK]'

        lines = self.lines['train']
        data = {e: [] for e in self.entities}
        for h, r, t in lines:
            data[h].append([self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]])
            data[t].append([self.vocab[h], self.vocab[r], self.vocab[mask_token]])

        # add a fake neighbor if there is no neighbor for the entity
        for ent in data:
            if len(data[ent]) == 0:
                data[ent].append([self.vocab[ent], self.vocab[self.no_relation_token], self.vocab[mask_token]])

        return data

    def _get_entity_filter(self):
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)
        for h, r, t in lines:
            entity_filter[h, r].add(self.entities[t]['token_id'])
            entity_filter[t, r].add(self.entities[h]['token_id'])
        return entity_filter

    def _create_examples(self):
        examples = dict()
        for mode in self.lines:
            data = list()
            lines = self.lines[mode]
            for h, r, t in tqdm(lines, desc=f'[{mode}]构建examples'):
                head_example, tail_example = self._create_one_example(h, r, t)
                data.append(head_example)
                data.append(tail_example)
            examples[mode] = data
        return examples

    def _create_one_example(self, h, r, t):
        mask_token = '[MASK]'
        head, tail = self.entities[h], self.entities[t]

        struc_head_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
        head_filters = list(self.entity_filter[t, r] - {head['token_id']})
        head_example = {
            'data_triple': (t, r, h),
            'struc_prompt': struc_head_prompt,
            'neighbors_label': tail['token_id'],
            'label': head["token_id"],
            'filters': head_filters,
        }

        struc_tail_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
        tail_filters = list(self.entity_filter[h, r] - {tail['token_id']})
        tail_example = {
            'data_triple': (h, r, t),
            'struc_prompt': struc_tail_prompt,
            'neighbors_label': head['token_id'],
            'label': tail["token_id"],
            'filters': tail_filters,
        }

        return head_example, tail_example

    def collate_fn(self, batch_data):

        data_triple = [data_dit['data_triple'] for data_dit in batch_data]  # [(h, r, t), ...]

        inputs_ids = [copy.deepcopy(data_dit['struc_prompt']) for data_dit in batch_data]
        struc_data = {'input_ids': torch.tensor(inputs_ids)}

        if self.add_neighbors:
            batch_struc_neighbors = [[] for _ in range(self.neighbor_num)]
            for ent, _, _ in data_triple:
                struc_neighbors = self.neighbors[ent]
                idxs = list(range(len(struc_neighbors)))
                if len(idxs) >= self.neighbor_num:
                    idxs = random.sample(idxs, self.neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.neighbor_num
                for i, idx in enumerate(idxs):
                    batch_struc_neighbors[i].append(struc_neighbors[idx])
            struc_neighbors = [{'input_ids': torch.tensor(batch_struc_neighbors[i])} for i in range(self.neighbor_num)]
        else:
            struc_neighbors = None

        neighbors_labels = torch.tensor([data_dit['neighbors_label'] for data_dit in batch_data]) \
            if self.add_neighbors else None
        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'data': data_triple,
            'struc_data': struc_data, 'struc_neighbors': struc_neighbors,
            'labels': labels, 'filters': filters, 'neighbors_labels': neighbors_labels,
        }

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_dev_dataloader(self):
        dataloader = DataLoader(self.dev_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_test_dataloader(self):
        dataloader = DataLoader(self.test_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader


# 使用nformer模块进行多个数据集的联合训练时, 使用此模块加载数据
class StructureJointDataModule:
    def __init__(self, args: dict):
        # 0. some variables used in this class
        self.data_path = args['data_path']
        self.datasets = ['DBpedia15K', 'Wikidata15K', 'Yago15K']
        alignment_files = ['dbp_wd_links.txt', 'wd_yg_links.txt']

        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']

        self.add_neighbors = args['add_neighbors']
        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.no_relation_token = args['no_relation_token']

        # 1. 读取实体 关系 对齐信息
        self.entities, self.relations, self.align = self._read_support_information(alignment_files)

        # 2 构建统一的词表
        self.vocab, self.entity_range, num_ents, num_rels = self._get_vocab()
        with open('vocab.txt', 'w', encoding='utf-8') as f:
            for k in self.vocab:
                f.write(f'{k}\t{self.vocab[k]}\n')
        with open('entity_range.json', 'w', encoding='utf-8') as f:
            json.dump(self.entity_range, f)
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = num_rels

        # 3 read dataset
        train_triples, valid_triples, test_triples = self._read_lines()
        self.neighbors = self._get_neighbors(train_triples)
        self.entity_filter = self._get_entity_filter(train_triples, valid_triples, test_triples)

        self.train_dataset = KGCDataset(self._create_examples(train_triples))
        self.valid_dateset = dict()
        for dataset in valid_triples:
            self.valid_dateset[dataset] = KGCDataset(self._create_examples(valid_triples[dataset], dataset))
        self.test_dataset = dict()
        for dataset in test_triples:
            self.test_dataset[dataset] = KGCDataset(self._create_examples(test_triples[dataset], dataset))

    def _read_support_information(self, entity_alignment_files):
        # 读取各个数据集的实体和关系信息
        entities, relations = dict(), dict()
        for dataset in self.datasets:
            ent_path = os.path.join(self.data_path, dataset, 'entities.txt')
            ents = list()
            with open(ent_path, 'r', encoding='utf-8') as f:
                for idx, ent in enumerate(f.readlines()):
                    ents.append(ent.strip())

            rel_path = os.path.join(self.data_path, dataset, 'relations.txt')
            rels = list()
            with open(rel_path, 'r', encoding='utf-8') as f:
                for idx, rel in enumerate(f.readlines()):
                    rels.append(rel.strip())
            entities[dataset] = ents
            relations[dataset] = rels

        # 全局的实体对齐信息, 对齐到最靠前的数据集上面
        align_dict = dict()
        for file_path in entity_alignment_files:
            with open(os.path.join(self.data_path, file_path), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    # 读取一对对齐的实体
                    ent1, ent2 = line.strip().split('\t')
                    # 将后面的实体对齐到前面的实体上
                    if ent2 in align_dict:
                        assert 0
                    if ent1 in align_dict:
                        # 说明ent1已经对齐到了之前的数据集的实体上, 将ent2也对齐到之前的实体上
                        align_dict[ent2] = align_dict[ent1]
                    else:
                        # 说明ent1没有对齐到之前的数据集上, ent2直接对齐到ent1即可
                        align_dict[ent2] = ent1

        return entities, relations, align_dict

    def _get_vocab(self):
        tokens = ['[PAD]', '[MASK]', '[SEP]', self.no_relation_token]

        # 所有的实体
        ent_names = set()
        for dataset in self.entities:  # 遍历每个数据集的实体
            ents = self.entities[dataset]
            for ent in ents:
                if ent in self.align:
                    ent = self.align[ent]

                if ent not in ent_names:
                    ent_names.add(ent)
        ent_names = sorted(list(ent_names))

        # 所有的关系
        rel_names = []
        for dataset in self.relations:
            rels = self.relations[dataset]
            for r in rels:
                rel_names += [r, f'{r}_reverse']

        # 构建词表
        tokens = tokens + ent_names + rel_names
        vocab = {token: idx for idx, token in enumerate(tokens)}

        # 记录每个数据集词表的范围, 用于最后计算rank
        ent_range = {}
        for dataset in self.entities:
            ents = self.entities[dataset]
            ent_ids = list()
            for ent in ents:
                if ent not in self.align:
                    ent_ids.append(vocab[ent])
                else:
                    ent_ids.append(vocab[self.align[ent]])
            ent_range[dataset] = ent_ids
        return vocab, ent_range, len(ent_names), len(rel_names)

    def _read_lines(self):
        # 读取txt文件中的三元组
        def read_triples(data_path):
            triples = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    h, r, t = line.strip().split('\t')
                    # 根据对齐信息, 对所有三元组中对齐实体的id进行修改
                    if h in self.align:
                        h = self.align[h]
                    if t in self.align:
                        t = self.align[t]
                    triples.append((h, r, t))
            return triples

        # 训练集合并成一个大的集合, 验证集和测试集分开保存
        train_triples = list()
        valid_triples = dict()
        test_triples = dict()
        for dataset in self.datasets:
            train_triples += read_triples(os.path.join(self.data_path, dataset, 'train.txt'))
            valid_triples[dataset] = read_triples(os.path.join(self.data_path, dataset, 'valid.txt'))
            test_triples[dataset] = read_triples(os.path.join(self.data_path, dataset, 'test.txt'))

        return train_triples, valid_triples, test_triples

    def _get_neighbors(self, triples):
        mask_token = '[MASK]'

        ents = set()
        for h, _, t in triples:
            ents.add(h)
            ents.add(t)
        ents = list(ents)

        neighbors = {e: [] for e in ents}
        for h, r, t in triples:
            neighbors[h].append([self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]])
            neighbors[t].append([self.vocab[h], self.vocab[r], self.vocab[mask_token]])

        # add a fake neighbor if there is no neighbor for the entity
        for ent in neighbors:
            if len(neighbors[ent]) == 0:
                neighbors[ent].append([self.vocab[ent], self.vocab[self.no_relation_token], self.vocab[mask_token]])

        return neighbors

    def _get_entity_filter(self, train_triples, valid_triples, test_triples):
        total_triples = list()
        total_triples += train_triples
        for dataset in valid_triples:
            total_triples += valid_triples[dataset]
        for dataset in test_triples:
            total_triples += test_triples[dataset]

        entity_filter = defaultdict(set)
        for h, r, t in total_triples:
            entity_filter[h, r].add(self.vocab[t])
            entity_filter[t, r].add(self.vocab[h])
        return entity_filter

    def _create_examples(self, triples, dataset=None):
        # 把抽象的三元组转换成模型的输入
        data = list()
        for h, r, t in triples:
            head_example, tail_example = self._create_one_example(h, r, t)
            head_example['dataset'] = dataset
            tail_example['dataset'] = dataset
            data.append(head_example)
            data.append(tail_example)
        return data

    def _create_one_example(self, h, r, t):
        mask_token = '[MASK]'
        h_id, t_id = self.vocab[h], self.vocab[t]
        r_id, r_reverse_id = self.vocab[r], self.vocab[f'{r}_reverse']

        # 预测h
        struc_head_prompt = [self.vocab[t], r_reverse_id, self.vocab[mask_token]]
        head_filters = list(self.entity_filter[t, r] - {h_id})
        head_example = {
            'data_triple': (t, f'{r}_reverse', h),
            'struc_prompt': struc_head_prompt,
            'filters': head_filters,
        }

        # 预测t
        struc_tail_prompt = [self.vocab[h], r_id, self.vocab[mask_token]]
        tail_filters = list(self.entity_filter[h, r] - {t_id})
        tail_example = {
            'data_triple': (h, r, t),
            'struc_prompt': struc_tail_prompt,
            'filters': tail_filters,
        }

        return head_example, tail_example

    def collate_fn(self, batch_data):
        # 检查当前batch来自哪个数据集
        dataset = list(set([data_dit['dataset'] for data_dit in batch_data]))
        if len(dataset) > 1:
            assert 0
        dataset = dataset[0]

        # 收集最原始的数据集
        data_triples = [data_dit['data_triple'] for data_dit in batch_data]

        inputs_ids = [copy.deepcopy(data_dit['struc_prompt']) for data_dit in batch_data]
        struc_data = {'input_ids': torch.tensor(inputs_ids)}

        if self.add_neighbors:
            batch_struc_neighbors = [[] for _ in range(self.neighbor_num)]
            for ent, _, _ in data_triples:
                struc_neighbors = self.neighbors[ent]
                idxs = list(range(len(struc_neighbors)))
                if len(idxs) >= self.neighbor_num:
                    idxs = random.sample(idxs, self.neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.neighbor_num
                for i, idx in enumerate(idxs):
                    batch_struc_neighbors[i].append(struc_neighbors[idx])

            struc_neighbors = [{'input_ids': torch.tensor(batch_struc_neighbors[i])} for i in range(self.neighbor_num)]
            neighbors_labels = torch.tensor([self.vocab[h] for h, _, _ in data_triples])
        else:
            struc_neighbors = None
            neighbors_labels = None

        labels = torch.tensor([self.vocab[t] for _, _, t in data_triples])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])
        entity_range = None if dataset is None else self.entity_range[dataset]

        return {
            'data': data_triples,
            'struc_data': struc_data, 'labels': labels, 'filters': filters, 'entity_range': entity_range,
            'struc_neighbors': struc_neighbors, 'neighbors_labels': neighbors_labels,
        }

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_valid_dataloader(self):
        data_loader = dict()
        for dataset in self.valid_dateset:
            data_loader[dataset] = DataLoader(self.valid_dateset[dataset], collate_fn=self.collate_fn,
                                              batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                              pin_memory=self.pin_memory, shuffle=False)
        return data_loader

    def get_test_dataloader(self):
        data_loader = dict()
        for dataset in self.test_dataset:
            data_loader[dataset] = DataLoader(self.test_dataset[dataset], collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return data_loader


if __name__ == '__main__':
    pass

