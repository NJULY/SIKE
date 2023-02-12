import os
import copy
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from .dataset import KGCDataset, read_triples

class ContinualDataModule:
    def __init__(
            self, data_path, tokenizer: BertTokenizer,
            encode_text=False, encode_struc=False, is_pretrain=False,
            max_seq_length=64, batch_size=128, num_workers=32, pin_memory=True,
    ):
        # some hyper-parameters
        self.is_pretrain = is_pretrain
        self.data_path = data_path
        self.dataset = self.data_path.split('/')[-1]
        self.encode_text = encode_text
        self.encode_struc = encode_struc

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_length = max_seq_length

        # 1. read entities and relations from dataset
        self.ents, self.rels = self.read_ents_and_rels()

        # 2. two vocabs, one for text, another for struct
        self.tokenizer = tokenizer
        self.init_tokenizer_size = len(self.tokenizer)  # the offset for new special tokens
        # construct two vocabs
        self.text_vocab_size, self.struc_vocab = self.get_vocab()
        # knowformer need vocab size and the number of relations, respectively
        self.struc_vocab_size = len(self.struc_vocab)
        self.struc_rels_num = 2 * len(self.rels)
        assert self.text_vocab_size == len(self.ents) + 4 * len(self.rels)
        assert self.struc_vocab_size == len(self.ents) + 2 * len(self.rels) + 3

        # 3.1 read triples from dataset
        train_triples = read_triples(os.path.join(self.data_path, 'train.txt'))
        valid_triples = read_triples(os.path.join(self.data_path, 'valid.txt'))
        test_triples = read_triples(os.path.join(self.data_path, 'test.txt'))
        # 3.2 get all possible t for (h, r, ?), it's a dict like (e_id, r_id) -> {e_token_id, ...}
        # the token ids start at 0, same for text and struct
        self.candidates = self.get_entity_candidates(train_triples + valid_triples + test_triples)

        # 4. construct Dataset for training
        if is_pretrain:  # based on entity desc, deprecated
            train_examples, valid_examples, test_examples = self.create_pretrain_examples()
            self.train_ds = KGCDataset(train_examples)
            self.valid_ds = KGCDataset(valid_examples)
            self.test_ds = KGCDataset(test_examples)
        else:
            self.train_ds = KGCDataset(self.create_examples(train_triples))
            self.valid_ds = KGCDataset(self.create_examples(valid_triples))
            self.test_ds = KGCDataset(self.create_examples(test_triples))

    def read_ents_and_rels(self):
        entity_path = os.path.join(self.data_path, 'entity.json')
        entities = json.load(open(entity_path, 'r', encoding='utf-8'))
        for idx, e in enumerate(entities):  # 14541
            new_name = f'[{self.dataset}_E_{idx}]'
            raw_name = entities[e]['name']
            desc = entities[e]['desc']
            entities[e] = {'token_id': idx, 'name': new_name, 'desc': desc, 'raw_name': raw_name}

        relation_path = os.path.join(self.data_path, 'relation.json')
        relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        for idx, r in enumerate(relations):
            sep1, sep2, sep3, sep4 = f'[{self.dataset}_R_{idx}_SEP1]', f'[{self.dataset}_R_{idx}_SEP2]', f'[{self.dataset}_R_{idx}_SEP3]', f'[{self.dataset}_R_{idx}_SEP4]'
            name = relations[r]['name']
            relations[r] = {'sep1': sep1, 'sep2': sep2, 'sep3': sep3, 'sep4': sep4, 'name': name}

        return entities, relations

    def get_vocab(self):
        # expand vocab for BERT
        ent_names = [self.ents[e]['name'] for e in self.ents]
        rel_names = []
        for rel in self.rels:
            rel_names += [self.rels[rel]['sep1'], self.rels[rel]['sep2'], self.rels[rel]['sep3'], self.rels[rel]['sep4']]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': ent_names+rel_names})

        # vocab for struct
        tokens = ['[PAD]', '[MASK]', '[SEP]']
        ents = [e for e in self.ents]
        rels = []
        for r in self.rels:
            rels += [r, f'{r}_reverse']
        vocab = {token: idx for idx, token in enumerate(ents + rels + tokens)}

        return num_added_tokens, vocab

    def get_entity_candidates(self, triples):
        entity_candidates = defaultdict(set)
        for h, r, t in triples:
            entity_candidates[h, r].add(self.ents[t]['token_id'])
            entity_candidates[t, r].add(self.ents[h]['token_id'])
        return entity_candidates

    # useless in this project
    def create_pretrain_examples(self):
        mask_token = self.tokenizer.mask_token
        train_examples, valid_examples, test_examples = list(), list(), list()

        for ent in self.ents.keys():
            ent_token_id = self.ents[ent]['token_id']
            raw_name = self.ents[ent]['raw_name']
            desc = f'Also known as {raw_name}, ' + self.ents[ent]['desc']
            desc_tokens = desc.split()

            prompts = [f'The description of {mask_token} is : {desc}']
            begins = random.sample(range(0, len(desc_tokens)), min(10, len(desc_tokens)))
            for begin in begins:
                end = min(begin + self.max_seq_length, len(desc_tokens))
                new_desc = ' '.join(desc_tokens[begin: end])
                prompts.append(f'The description of {mask_token} is : {new_desc}')

            for i in range(len(prompts)):
                example = {'triple': None, 'text': None, 'prompt': prompts[i], 'label': ent_token_id, 'filters': None}
                if i == len(prompts) - 1:
                    test_examples.append(example)
                elif i == len(prompts) - 2:
                    valid_examples.append(example)
                else:
                    train_examples.append(example)
        return train_examples, valid_examples, test_examples

    def create_examples(self, triples):
        data = list()
        for h, r, t in triples:
            head_example, tail_example = self.create_one_example(h, r, t)
            data.append(head_example)
            data.append(tail_example)
        return data

    def create_one_example(self, h, r, t):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token

        head, rel, tail = self.ents[h], self.rels[r], self.ents[t]
        # head
        h_name, h_desc, h_raw_name, h_token_id = head['name'], head['desc'], head['raw_name'], head['token_id']
        h_desc = f'Also known as {h_raw_name}, {h_desc}'
        # relation
        r_name = rel['name']
        sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']
        # tail
        t_name, t_desc, t_raw_name, t_token_id = tail['name'], tail['desc'], tail['raw_name'], tail['token_id']
        t_desc = f'Also known as {t_raw_name}, {t_desc}'

        text_head_prompt = ' '.join([sep1, mask_token, sep2, r_name, sep3, t_name, sep4, t_desc])
        text_tail_prompt = ' '.join([sep1, h_name, sep2, r_name, sep3, mask_token, sep4, h_desc])

        struc_head_prompt = [self.struc_vocab[t], self.struc_vocab[f'{r}_reverse'], self.struc_vocab[mask_token]]
        struc_tail_prompt = [self.struc_vocab[h], self.struc_vocab[r], self.struc_vocab[mask_token]]

        head_filters = list(self.candidates[t, r] - {h_token_id})
        tail_filters = list(self.candidates[h, r] - {t_token_id})

        # prepare examples
        head_example = {
            'triple': (t, r, h), 'text': (t_raw_name, r_name, h_raw_name, 0),
            'prompt': text_head_prompt, 'struc_input': struc_head_prompt,
            'label': h_token_id, 'filters': head_filters,
        }
        tail_example = {
            'triple': (h, r, t), 'text': (h_raw_name, r_name, t_raw_name, 1),
            'prompt': text_tail_prompt, 'struc_input': struc_tail_prompt,
            'label': t_token_id, 'filters': tail_filters,
        }

        return head_example, tail_example

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])

        # words to be replaced
        new_pos = torch.nonzero(torch.ge(input_ids, self.init_tokenizer_size))
        new_ids = input_ids[new_pos[:, 0], new_pos[:, 1]] - self.init_tokenizer_size
        ent_pos = new_pos[torch.less(new_ids, len(self.ents))]
        ent_ids = new_ids[torch.less(new_ids, len(self.ents))]
        sp_pos = new_pos[torch.ge(new_ids, len(self.ents))]
        sp_ids = new_ids[torch.ge(new_ids, len(self.ents))]
        input_ids[ent_pos[:, 0], ent_pos[:, 1]] = self.tokenizer.pad_token_id
        input_ids[sp_pos[:, 0], sp_pos[:, 1]] = self.tokenizer.sep_token_id

        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))

        return {
            'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
            'new_pos': new_pos, 'new_ids': new_ids, 'mask_pos': mask_pos,
            'ent_pos': ent_pos, 'ent_ids': ent_ids, 'sp_pos': sp_pos, 'sp_ids': sp_ids,
        }

    def struc_batch_encoding(self, inputs):
        input_ids = torch.tensor(inputs)
        return input_ids

    def collate_fn(self, batch_data):
        triples = [data_dit['triple'] for data_dit in batch_data]
        texts = [data_dit['text'] for data_dit in batch_data]

        if self.encode_text:
            prompts = self.text_batch_encoding([data_dit['prompt'] for data_dit in batch_data])
        else:
            prompts = None

        if self.encode_struc:
            struc_inputs = self.struc_batch_encoding([data_dit['struc_input'] for data_dit in batch_data])
        else:
            struc_inputs = None

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        if self.is_pretrain:
            filters = None
        else:
            filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'triples': triples, 'texts': texts,
            'prompts': prompts, 'struc_inputs': struc_inputs,
            'labels': labels, 'filters': filters, 'ent_num': len(self.ents),
        }

    # following 3 are used to get data loaders
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
