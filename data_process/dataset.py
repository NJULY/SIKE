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

class DataModule:
    def __init__(
            self, kg_name, kg_path, tokenizer_path, num_soft_prompt=4, max_seq_length=64,
            batch_size=128, num_workers=32, pin_memory=True,
    ):
        self.kg_name = kg_name
        self.kg_path = kg_path
        self.num_soft_prompt = num_soft_prompt
        self.max_seq_length = max_seq_length

        entity_path = os.path.join(self.kg_path, 'entity.json')
        relation_path = os.path.join(self.kg_path, 'relation.json')
        self.entities = self.get_entities(entity_path)
        self.relations = self.get_relations(relation_path)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)
        self.offset = len(self.tokenizer)
        self.vocab = self.get_vocab()

        train_triples = read_triples(os.path.join(self.kg_path, 'train.txt'))
        valid_triples = read_triples(os.path.join(self.kg_path, 'valid.txt'))
        test_triples = read_triples(os.path.join(self.kg_path, 'test.txt'))
        self.hr2t = self.get_hr2t(train_triples + valid_triples + test_triples, self.entities)

        self.train_ds = KGCDataset(self.create_examples(train_triples))
        self.valid_ds = KGCDataset(self.create_examples(valid_triples))
        self.test_ds = KGCDataset(self.create_examples(test_triples))

        self.train_dl = DataLoader(self.train_ds, collate_fn=self.collate_fn, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        self.valid_dl = DataLoader(self.valid_ds, collate_fn=self.collate_fn, batch_size=2 * batch_size,
                                   num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        self.test_dl = DataLoader(self.test_ds, collate_fn=self.collate_fn, batch_size=2 * batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    def get_entities(self, entity_path):
        entities = json.load(open(entity_path, 'r', encoding='utf-8'))
        for idx, e_id in enumerate(entities):
            ent = entities[e_id]
            raw_name, desc = ent['name'], ent['desc']
            name = f'[{self.kg_name}_ENT{idx}]'
            entities[e_id] = {'token_id': idx, 'name': name, 'desc': desc, 'raw_name': raw_name}
        return entities

    def get_relations(self, relation_path):
        relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        for idx, r_id in enumerate(relations):
            raw_name = relations[r_id]['name']
            relations[r_id] = {f'sp{i + 1}': f'[{self.kg_name}_REL{idx}_SP{i + 1}]' for i in
                               range(self.num_soft_prompt)}
            relations[r_id].update(
                {'name': f'[{self.kg_name}_REL{idx}]', 'reverse_name': f'[{self.kg_name}_REL{idx}_reverse]',
                 'raw_name': raw_name})
        return relations

    def get_vocab(self):
        entity_names = [self.entities[e_id]['name'] for e_id in self.entities]
        relation_sp_names = []
        for r_id in self.relations:
            relation_sp_names += [self.relations[r_id][f'sp{i + 1}'] for i in range(self.num_soft_prompt)]

        num_added_tokens = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': entity_names + relation_sp_names})
        assert num_added_tokens == len(self.entities) + len(self.relations) * self.num_soft_prompt

        relation_names = []
        for r_id in self.relations:
            relation_names += [self.relations[r_id]['name'], self.relations[r_id]['reverse_name']]
        vocab = {token: idx for idx, token in
                 enumerate(entity_names + relation_sp_names + relation_names + ['[MASK]', '[SEP]'])}
        return vocab

    def get_hr2t(self, triples, entities):
        hr2t = defaultdict(set)
        for h, r, t in triples:
            hr2t[h, r].add(entities[t]['token_id'])
            hr2t[t, r].add(entities[h]['token_id'])
        return hr2t

    def create_examples(self, triples):
        data = list()
        for h, r, t in triples:
            head_example, tail_example = self.create_one_example(h, r, t)
            data.append(head_example)
            data.append(tail_example)
        return data

    def create_one_example(self, h_id, r_id, t_id):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token

        head, relation, tail = self.entities[h_id], self.relations[r_id], self.entities[t_id]

        h_name, h_desc, h_raw_name, h_token_id = head['name'], head['desc'], head['raw_name'], head['token_id']
        h_desc = f'{h_name} is also known as {h_raw_name}, {h_desc}.' if h_desc != '' else f'{h_name} is also known as {h_raw_name}.'
        r_name, r_reverse_name, r_raw_name = relation['name'], relation['reverse_name'], relation['raw_name']
        sp = [relation[f'sp{i + 1}'] for i in range(self.num_soft_prompt)]
        t_name, t_desc, t_raw_name, t_token_id = tail['name'], tail['desc'], tail['raw_name'], tail['token_id']
        t_desc = f'{t_name} is also known as {t_raw_name}, {t_desc}.' if t_desc != '' else f'{t_name} is also known as {t_raw_name}.'

        head_text = ' '.join([sp[0], mask_token, sp[1], r_raw_name, sp[2], t_name, sp[3], t_desc])
        tail_text = ' '.join([sp[0], h_name, sp[1], r_raw_name, sp[2], mask_token, sp[3], h_desc])

        head_struct = [self.vocab[t_name], self.vocab[r_reverse_name], self.vocab['[MASK]']]
        tail_struct = [self.vocab[h_name], self.vocab[r_name], self.vocab['[MASK]']]

        head_filters = list(self.hr2t[t_id, r_id] - {h_token_id})
        tail_filters = list(self.hr2t[h_id, r_id] - {t_token_id})

        head_example = {
            'triple': (t_id, r_id, h_id), 'raw_text': (t_raw_name, r_name, h_raw_name, 'first is tail'),
            'text': head_text, 'struct': head_struct, 'label': h_token_id, 'filters': head_filters,
        }
        tail_example = {
            'triple': (h_id, r_id, t_id), 'raw_text': (h_raw_name, r_name, t_raw_name, 'first is head'),
            'text': tail_text, 'struct': tail_struct, 'label': t_token_id, 'filters': tail_filters,
        }
        return head_example, tail_example

    def batch_encoding(self, texts):
        encoded_data = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])

        new_pos = torch.nonzero(torch.ge(input_ids, self.offset))
        new_ids = input_ids[new_pos[:, 0], new_pos[:, 1]] - self.offset
        ent_pos = new_pos[torch.less(new_ids, len(self.entities))]
        ent_ids = new_ids[torch.less(new_ids, len(self.entities))]
        sp_pos = new_pos[torch.ge(new_ids, len(self.entities))]
        sp_ids = new_ids[torch.ge(new_ids, len(self.entities))]
        input_ids[ent_pos[:, 0], ent_pos[:, 1]] = self.tokenizer.pad_token_id
        input_ids[sp_pos[:, 0], sp_pos[:, 1]] = self.tokenizer.sep_token_id  # 填充为PAD
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))

        return {
            'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
            'new_pos': new_pos, 'new_ids': new_ids, 'mask_pos': mask_pos,
            'ent_pos': ent_pos, 'ent_ids': ent_ids, 'sp_pos': sp_pos, 'sp_ids': sp_ids,
        }

    def collate_fn(self, batch_data):
        triple = [data_dit['triple'] for data_dit in batch_data]
        raw_text = [data_dit['raw_text'] for data_dit in batch_data]
        text = self.batch_encoding([data_dit['text'] for data_dit in batch_data])
        struct = torch.tensor([data_dit['struct'] for data_dit in batch_data])
        label = torch.tensor([data_dit['label'] for data_dit in batch_data])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'triple': triple, 'raw_text': raw_text, 'text': text, 'struct': struct, 'label': label, 'filters': filters,
        }