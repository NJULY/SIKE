import os
import copy
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .utils import KGCDataset, read_triples


class SIKEDataModule:
    def __init__(self, args: dict, dataset):
        # save hyperparameters
        self.args = args
        self.data_path = os.path.join(args['data_path'], dataset)
        print(f'Load dataset {dataset} from path {self.data_path}')
        self.dataset = dataset

        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.max_seq_length = args['max_seq_length']

        self.ent2text, self.rel2text = self.read_ents_and_rels()
        self.tokenizer = AutoTokenizer.from_pretrained(args['bert_path'])
        self.init_tokenizer_size = len(self.tokenizer)  # the offset for new special tokens
        self.num_added_tokens = self.vocab_expand()
        assert self.num_added_tokens == len(self.ent2text) + 4 * len(self.rel2text)

        # read triples from dataset
        train_triples = read_triples(os.path.join(self.data_path, 'train.txt'))
        valid_triples = read_triples(os.path.join(self.data_path, 'valid.txt'))
        test_triples = read_triples(os.path.join(self.data_path, 'test.txt'))

        self.train_hr2t, self.train_tr2h = self.get_hr2t_and_tr2h(train_triples)
        self.all_hr2t, self.all_tr2h = self.get_hr2t_and_tr2h(train_triples + valid_triples + test_triples)

        # construct Dataset for training
        self.train_ds = KGCDataset(self.create_examples(train_triples, is_train=True))
        self.valid_ds = KGCDataset(self.create_examples(valid_triples))
        self.test_ds = KGCDataset(self.create_examples(test_triples))

        self.train_loader = self.get_train_dataloader()
        self.valid_loader = self.get_valid_dataloader()
        self.test_loader = self.get_test_dataloader()

    def read_ents_and_rels(self):
        entity_path = os.path.join(self.data_path, 'entity.json')
        entities = json.load(open(entity_path, 'r', encoding='utf-8'))
        for idx, e in enumerate(entities):  
            new_name = f'[{self.dataset}_E_{idx}]'
            raw_name = entities[e]['name']
            desc = entities[e]['desc']
            entities[e] = {'idx': idx, 'name': new_name, 'desc': desc, 'raw_name': raw_name}

        relation_path = os.path.join(self.data_path, 'relation.json')
        relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        for idx, r in enumerate(relations):
            sep1, sep2, sep3, sep4 = f'[{self.dataset}_R_{idx}_SEP1]', f'[{self.dataset}_R_{idx}_SEP2]', f'[{self.dataset}_R_{idx}_SEP3]', f'[{self.dataset}_R_{idx}_SEP4]'
            name = relations[r]['name']
            relations[r] = {'sep1': sep1, 'sep2': sep2, 'sep3': sep3, 'sep4': sep4,'name': name}

        return entities, relations

    def vocab_expand(self):
        ent_names = [self.ent2text[e]['name'] for e in self.ent2text]
        rel_names = []
        for rel in self.rel2text:
            rel_names += [self.rel2text[rel]['sep1'], self.rel2text[rel]['sep2'], self.rel2text[rel]['sep3'], self.rel2text[rel]['sep4']]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': ent_names+rel_names})

        return num_added_tokens

    def get_hr2t_and_tr2h(self, triples):
        hr2t = defaultdict(set)
        tr2h = defaultdict(set)
        for h, r, t in triples:
            hr2t[h, r].add(self.ent2text[t]['idx'])
            tr2h[t, r].add(self.ent2text[h]['idx'])
        return hr2t, tr2h

    def create_examples(self, triples, is_train=False):
        data = list()
        for h, r, t in triples:
            head_example, tail_example = self.create_one_example(h, r, t, is_train)
            data.append(head_example)
            data.append(tail_example)
        return data

    def create_one_example(self, h, r, t, is_train):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token

        head, rel, tail = self.ent2text[h], self.rel2text[r], self.ent2text[t]
        
        h_name, h_desc, h_raw_name, h_idx = head['name'], head['desc'], head['raw_name'], head['idx']
        h_desc = f'Also known as {h_raw_name}, {h_desc}'
        
        r_name = rel['name']
        sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']
        
        t_name, t_desc, t_raw_name, t_idx = tail['name'], tail['desc'], tail['raw_name'], tail['idx']
        t_desc = f'Also known as {t_raw_name}, {t_desc}'

        text_head_prompt = ' '.join([sep1, mask_token, sep2, r_name, sep3, t_name, sep4, t_desc])
        text_tail_prompt = ' '.join([sep1, h_name, sep2, r_name, sep3, mask_token, sep4, h_desc])

        if is_train:
            head_filters = list(self.train_tr2h[t, r] - {h_idx})
            tail_filters = list(self.train_hr2t[h, r] - {t_idx})
        else:
            head_filters = list(self.all_tr2h[t, r] - {h_idx})
            tail_filters = list(self.all_hr2t[h, r] - {t_idx})

        # prepare examples
        head_example = {
            'triple': (t, r, h), 'text': (t_raw_name, r_name, h_raw_name, True),
            'prompt': text_head_prompt, 
            'label': h_idx, 'filters': head_filters,
        }
        tail_example = {
            'triple': (h, r, t), 'text': (h_raw_name, r_name, t_raw_name, False),
            'prompt': text_tail_prompt, 
            'label': t_idx, 'filters': tail_filters,
        }

        return head_example, tail_example

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])

        new_pos = torch.nonzero(torch.ge(input_ids, self.init_tokenizer_size))
        new_ids = input_ids[new_pos[:, 0], new_pos[:, 1]] - self.init_tokenizer_size
        ent_pos = new_pos[torch.less(new_ids, len(self.ent2text))]
        ent_ids = new_ids[torch.less(new_ids, len(self.ent2text))]
        sp_pos = new_pos[torch.ge(new_ids, len(self.ent2text))]
        sp_ids = new_ids[torch.ge(new_ids, len(self.ent2text))]
        input_ids[ent_pos[:, 0], ent_pos[:, 1]] = self.tokenizer.pad_token_id
        input_ids[sp_pos[:, 0], sp_pos[:, 1]] = self.tokenizer.sep_token_id

        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))

        return {
            'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
            'new_pos': new_pos, 'new_ids': new_ids, 'mask_pos': mask_pos,
            'ent_pos': ent_pos, 'ent_ids': ent_ids, 'sp_pos': sp_pos, 'sp_ids': sp_ids,
        }

    def collate_fn(self, batch_data):
        triples = [data_dit['triple'] for data_dit in batch_data]
        texts = [data_dit['text'] for data_dit in batch_data]

        prompts = self.text_batch_encoding([data_dit['prompt'] for data_dit in batch_data])

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'triples': triples, 'texts': texts,
            'prompts': prompts, 
            'labels': labels, 'filters': filters, 'ent_num': len(self.ent2text),
        }

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_valid_dataloader(self):
        dataloader = DataLoader(self.valid_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_test_dataloader(self):
        dataloader = DataLoader(self.test_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

