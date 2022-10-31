import os
import copy
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from .dataset import KGCDataset


class JointDataModule:
    def __init__(
            self, args: dict, tokenizer: BertTokenizer,
            encode_text=False, encode_struc=False, use_align=False, joint_train=False,
    ):
        assert args['task'] in ['pretrain', 'train', 'validate']
        self.is_pretrain = True if args['task'] == 'pretrain' else False
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.max_seq_length = args['max_seq_length'] if encode_text else -1
        self.encode_text = encode_text
        self.encode_struc = encode_struc

        self.datasets = ['DBpedia15K', 'Wikidata15K', 'Yago15K']
        if use_align:
            align_files = ['dbp_wd_links.txt', 'wd_yg_links.txt']
        else:
            align_files = []

        self.dataset2ents, self.dataset2rels, self.align, self.ent2text, self.rel2text = self.read_support(align_files)

        self.tokenizer = tokenizer
        self.text_ent_range = self.resize_tokenizer()
        self.vocab, self.struc_ent_range, num_rels = self.get_vocab()
        self.struc_vocab_size = len(self.vocab)
        self.struc_num_relations = num_rels

        train_triples, valid_triples, test_triples = self.read_lines()
        self.text_ent_filter, self.struc_ent_filter = self.get_entity_filter(train_triples, valid_triples, test_triples)

        if self.is_pretrain:
            if joint_train:
                self.train_dataset = {'joint': KGCDataset(self._create_pretrain_examples(self.ent2text, None))}
            else:
                self.train_dataset = {
                    dataset: KGCDataset(self._create_pretrain_examples(self.dataset2ents[dataset], None))
                    for dataset in self.datasets
                }

            self.valid_dateset = dict()
            self.test_dataset = dict()
            for dataset in self.datasets:
                self.valid_dateset[dataset] = KGCDataset(
                    self._create_pretrain_examples(self.dataset2ents[dataset], dataset))
                self.test_dataset[dataset] = KGCDataset(
                    self._create_pretrain_examples(self.dataset2ents[dataset], dataset))
        else:
            if joint_train:
                total_train_triples = []
                for dataset in train_triples:
                    total_train_triples += train_triples[dataset]
                self.train_dataset = {'joint': KGCDataset(self._create_examples(total_train_triples, None))}
            else:
                self.train_dataset = {
                    dataset: KGCDataset(self._create_examples(train_triples[dataset], dataset))
                    for dataset in self.datasets
                }

            self.valid_dateset = dict()
            self.test_dataset = dict()
            for dataset in self.datasets:
                self.valid_dateset[dataset] = KGCDataset(self._create_examples(valid_triples[dataset], dataset))
                self.test_dataset[dataset] = KGCDataset(self._create_examples(test_triples[dataset], dataset))

    def read_support(self, entity_alignment_files):
        dataset2ents, dataset2rels = dict(), dict()
        for dataset in self.datasets:
            dataset_folder = os.path.join(self.data_path, dataset)

            entity_path = os.path.join(dataset_folder, 'entity.json')
            ent_dit = json.load(open(entity_path, 'r', encoding='utf-8'))
            for idx, e in enumerate(ent_dit):
                new_name = f'[{dataset}_E_{idx}]'
                raw_name = ent_dit[e]['name']
                desc = ent_dit[e]['desc']
                ent_dit[e] = {
                    'name': new_name,  
                    'desc': desc,  
                    'raw_name': raw_name, 
                }

            relation_path = os.path.join(dataset_folder, 'relation.json')
            rel_dit = json.load(open(relation_path, 'r', encoding='utf-8'))
            for idx, r in enumerate(rel_dit):  
                sep1, sep2, sep3, sep4 = f'[{dataset}_R_{idx}_SEP1]', f'[{dataset}_R_{idx}_SEP2]', f'[{dataset}_R_{idx}_SEP3]', f'[{dataset}_R_{idx}_SEP4]'
                name = rel_dit[r]['name']
                rel_dit[r] = {
                    'sep1': sep1,  
                    'sep3': sep3,
                    'sep4': sep4,
                    'name': name,  
                }

            dataset2ents[dataset] = ent_dit
            dataset2rels[dataset] = rel_dit


        align_dict = dict()
        for file_path in entity_alignment_files:
            with open(os.path.join(self.data_path, file_path), 'r', encoding='utf-8') as f:
                for line in f.readlines():

                    ent1, ent2 = line.strip().split('\t')
                    if ent1 in align_dict:  
                        align_dict[ent2] = align_dict[ent1]
                    else: 
                        align_dict[ent2] = ent1

        ent2text = dict()
        for dataset in dataset2ents:
            ents = dataset2ents[dataset]
            for ent in ents:
                ent2text[ent] = ents[ent]
        for ent in ent2text:
            if ent in align_dict:
                ent2text[ent] = ent2text[align_dict[ent]]

        rel2text = dict()
        for dataset in dataset2rels:
            rels = dataset2rels[dataset]
            for rel in rels:
                rel2text[rel] = rels[rel]

        return dataset2ents, dataset2rels, align_dict, ent2text, rel2text

    def resize_tokenizer(self):
        ent_names = sorted(set([self.ent2text[ent]['name'] for ent in self.ent2text]))

        rel_names = list()
        for rel in self.rel2text:
            rel_names += [self.rel2text[rel]['sep1'], self.rel2text[rel]['sep2'], self.rel2text[rel]['sep3'],
                          self.rel2text[rel]['sep4']]


        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': ent_names + rel_names})
        assert len(ent_names) + len(rel_names) == num_added_tokens

        ent_range = dict()
        for dataset in self.dataset2ents:
            ents = self.dataset2ents[dataset]
            ent_tokens = [self.ent2text[ent]['name'] for ent in ents]
            ent_token_ids = sorted(self.tokenizer.convert_tokens_to_ids(ent_tokens))
            ent_range[dataset] = sorted(ent_token_ids)

        return ent_range

    def get_vocab(self):
        tokens = ['[PAD]', '[MASK]', '[SEP]']

        ent_ids = set()
        for dataset in self.dataset2ents:  
            ents = self.dataset2ents[dataset]
            for ent in ents:
                if ent in self.align:
                    ent = self.align[ent]
                ent_ids.add(ent)
        ent_ids = sorted(list(ent_ids))

        rel_ids = []
        for r in self.rel2text:
            rel_ids += [r, f'{r}_reverse']


        tokens = ent_ids + rel_ids + tokens
        vocab = {token: idx for idx, token in enumerate(tokens)}


        ent_range = {}
        for dataset in self.dataset2ents:
            ents = self.dataset2ents[dataset]
            ent_token_ids = list()
            for ent in ents:
                if ent not in self.align:
                    ent_token_ids.append(vocab[ent])
                else:
                    ent_token_ids.append(vocab[self.align[ent]])
            ent_range[dataset] = sorted(ent_token_ids)
        return vocab, ent_range, len(rel_ids)


    def read_lines(self):

        def _read_triples(data_path):
            triples = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    h, r, t = line.strip().split('\t')

                    if h in self.align:
                        h = self.align[h]
                    if t in self.align:
                        t = self.align[t]
                    triples.append((h, r, t))
            return triples

        train_triples = dict()
        valid_triples = dict()
        test_triples = dict()
        for dataset in self.datasets:
            train_triples[dataset] = _read_triples(os.path.join(self.data_path, dataset, 'train.txt'))
            valid_triples[dataset] = _read_triples(os.path.join(self.data_path, dataset, 'valid.txt'))
            test_triples[dataset] = _read_triples(os.path.join(self.data_path, dataset, 'test.txt'))

        return train_triples, valid_triples, test_triples

    def get_entity_filter(self, train_triples, valid_triples, test_triples):
        total_triples = list()
        for dataset in self.datasets:
            total_triples += train_triples[dataset]
            total_triples += valid_triples[dataset]
            total_triples += test_triples[dataset]

        text_ent_filter = defaultdict(set)
        for h, r, t in total_triples:
            h_token_id = self.tokenizer.convert_tokens_to_ids(self.ent2text[h]['name'])
            t_token_id = self.tokenizer.convert_tokens_to_ids(self.ent2text[t]['name'])
            text_ent_filter[h, r].add(t_token_id)
            text_ent_filter[t, r].add(h_token_id)

        struc_ent_filter = defaultdict(set)
        for h, r, t in total_triples:
            struc_ent_filter[h, r].add(self.vocab[t])
            struc_ent_filter[t, r].add(self.vocab[h])
        return text_ent_filter, struc_ent_filter

    def _create_examples(self, triples, dataset):
        data = list()
        for h, r, t in triples:
            head_example, tail_example = self._create_one_example(h, r, t)
            head_example['dataset'] = dataset
            tail_example['dataset'] = dataset
            data.append(head_example)
            data.append(tail_example)
        return data

    def _create_one_example(self, h, r, t):
        assert h not in self.align
        assert t not in self.align

        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token

        head, rel, tail = self.ent2text[h], self.rel2text[r], self.ent2text[t]
        h_name, h_desc, h_raw_name = head['name'], head['desc'], head['raw_name']
        t_name, t_desc, t_raw_name = tail['name'], tail['desc'], tail['desc']
        r_name = rel['name']
        sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

        if self.encode_text:
            text_head_prompt = ' '.join([sep1, mask_token, sep2, r_name, sep3, t_name, sep4, t_desc])
            text_head_label = self.tokenizer.convert_tokens_to_ids(h_name)
            text_head_filters = list(self.text_ent_filter[t, r] - {self.tokenizer.convert_tokens_to_ids(h_name)})

            text_tail_prompt = ' '.join([sep1, h_name, sep2, r_name, sep3, mask_token, sep4, h_desc])
            text_tail_label = self.tokenizer.convert_tokens_to_ids(t_name)
            text_tail_filters = list(self.text_ent_filter[h, r] - {self.tokenizer.convert_tokens_to_ids(t_name)})
        else:
            text_head_prompt, text_tail_prompt = None, None
            text_head_label, text_tail_label = None, None
            text_head_filters, text_tail_filters = None, None
        if self.encode_struc:
            struc_head_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
            struc_head_label = self.vocab[h]
            struc_head_filters = list(self.struc_ent_filter[t, r] - {self.vocab[h]})

            struc_tail_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
            struc_tail_label = self.vocab[t]
            struc_tail_filters = list(self.struc_ent_filter[h, r] - {self.vocab[t]})
        else:
            struc_head_prompt, struc_tail_prompt = None, None
            struc_head_label, struc_tail_label = None, None
            struc_head_filters, struc_tail_filters = None, None

        head_example = {
            'triple': (t, r, h), 'text': (tail["raw_name"], r_name, head['raw_name'], 0),
            'text_prompt': text_head_prompt, 'text_label': text_head_label,
            'text_filters': text_head_filters,
            'struc_prompt': struc_head_prompt, 'struc_label': struc_head_label,
            'struc_filters': struc_head_filters,
        }
        tail_example = {
            'triple': (h, r, t), 'text': (head['raw_name'], r_name, tail['raw_name'], 1),
            'text_prompt': text_tail_prompt, 'text_label': text_tail_label,
            'text_filters': text_tail_filters,
            'struc_prompt': struc_tail_prompt, 'struc_label': struc_tail_label,
            'struc_filters': struc_tail_filters,
        }

        return head_example, tail_example

    def _create_pretrain_examples(self, ents, dataset=None):
        data = list()
        for ent in ents:
            name = self.ent2text[ent]['name']
            raw_name = self.ent2text[ent]['raw_name']
            desc = self.ent2text[ent]['desc']
            desc_tokens = desc.split()

            prompts = [f'The description of {self.tokenizer.mask_token} (also known as {raw_name}) is that {desc}']
            sample_num = 10
            for i in range(sample_num):
                begin = random.randint(0, len(desc_tokens))
                end = min(begin + self.max_seq_length, len(desc_tokens))
                new_desc = ' '.join(desc_tokens[begin: end])
                prompts.append(
                    f'The description of {self.tokenizer.mask_token}  (also known as {raw_name}) is that {new_desc}')
            for prompt in prompts:
                data.append({'prompt': prompt, 'label': self.tokenizer.convert_tokens_to_ids(name), 'dataset': dataset})
        return data

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))

        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'mask_pos': mask_pos}

    def struc_batch_encoding(self, inputs):
        input_ids = torch.tensor(inputs)
        return {'input_ids': input_ids}

    def collate_fn(self, batch_data):
        if self.is_pretrain:
            return self.collate_fn_for_pretrain(batch_data)

        datasets = [data_dit['dataset'] for data_dit in batch_data]
        assert len(set(datasets)) == 1
        dataset = datasets[0]

        data_triple = [data_dit['triple'] for data_dit in batch_data] 
        data_text = [data_dit['text'] for data_dit in batch_data] 

        if self.encode_text:
            text_prompts = [data_dit['text_prompt'] for data_dit in batch_data]
            text_data = self.text_batch_encoding(text_prompts)
            text_labels = torch.tensor([data_dit['text_label'] for data_dit in batch_data])
            text_filters = torch.tensor(
                [[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['text_filters']])
            text_ent_range = self.text_ent_range[dataset] if dataset is not None else None
        else:
            text_data, text_labels, text_filters, text_ent_range = None, None, None, None
        if self.encode_struc:
            struc_prompts = [copy.deepcopy(data_dit['struc_prompt']) for data_dit in batch_data] 
            struc_data = self.struc_batch_encoding(struc_prompts) if self.encode_struc else None
            struc_labels = torch.tensor([data_dit['struc_label'] for data_dit in batch_data])
            struc_filters = torch.tensor(
                [[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['struc_filters']])
            struc_ent_range = self.struc_ent_range[dataset] if dataset is not None else None
        else:
            struc_data, struc_labels, struc_filters, struc_ent_range = None, None, None, None

        return {
            'triples': data_triple, 'texts': data_text,
            'text_data': text_data, 'text_labels': text_labels,
            'text_filters': text_filters, 'text_ent_range': text_ent_range,
            'struc_data': struc_data, 'struc_labels': struc_labels,
            'struc_filters': struc_filters, 'struc_ent_range': struc_ent_range,
        }

    def collate_fn_for_pretrain(self, batch_data):
        assert self.is_pretrain
        datasets = [data_dit['dataset'] for data_dit in batch_data]
        assert len(set(datasets)) == 1
        dataset = datasets[0]

        lm_prompts = [data_dit['prompt'] for data_dit in batch_data] 
        lm_data = self.text_batch_encoding(lm_prompts)

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])

        return {
            'triples': None, 'texts': None,
            'text_data': lm_data, 'text_labels': labels, 'text_filters': None,
            'text_ent_range': self.text_ent_range[dataset] if dataset is not None else None,
        }

    def get_train_dataloader(self):
        data_loader = dict()
        for dataset in self.train_dataset:
            data_loader[dataset] = DataLoader(self.train_dataset[dataset], collate_fn=self.collate_fn,
                                              batch_size=self.batch_size, num_workers=self.num_workers,
                                              pin_memory=self.pin_memory, shuffle=True)
        return data_loader

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
