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


class BertDataModule:
    def __init__(self, args: dict, tokenizer: BertTokenizer):

        self.task = args['task']
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.max_seq_length = args['max_seq_length']

        self.datasets = ['DBpedia15K', 'Wikidata15K', 'Yago15K']

        self.entities, self.relations = self.read_support()

        self.tokenizer = tokenizer
        self.ent2text, self.rel2text, self.text_ent_range = self.resize_tokenizer()

        if self.task == 'pretrain':

            self.train_examples = dict()
            self.valid_examples = dict()
            self.test_examples = dict()
            for dataset in self.datasets:
                train_exams, valid_exams, test_exams = self._create_pretrain_examples(self.entities[dataset], dataset)
                self.train_examples[dataset] = train_exams
                self.valid_examples[dataset] = valid_exams
                self.test_examples[dataset] = test_exams

            train_examples = []
            for dataset in self.train_examples:
                train_examples += self.train_examples[dataset]
            self.train_dataset = KGCDataset(train_examples)

            self.valid_dataset = dict()
            for dataset in self.datasets:
                self.valid_dataset[dataset] = KGCDataset(self.valid_examples[dataset])

            self.test_dataset = dict()
            for dataset in self.datasets:
                self.test_dataset[dataset] = KGCDataset(self.test_examples[dataset])
        else:
            train_triples, valid_triples, test_triples = self.read_lines()
            self.text_ent_filter = self.get_entity_filter(train_triples, valid_triples, test_triples)

            train_examples = []
            for dataset in self.datasets:
                train_examples += self._create_examples(train_triples[dataset], None)
            self.train_dataset = KGCDataset(train_examples)

            self.valid_dataset = dict()
            for dataset in self.datasets:
                self.valid_dataset[dataset] = KGCDataset(self._create_examples(valid_triples[dataset], dataset))

            self.test_dataset = dict()
            for dataset in self.datasets:
                self.test_dataset[dataset] = KGCDataset(self._create_examples(test_triples[dataset], dataset))

    def read_support(self):
        entities, relations = dict(), dict()
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
                    'sep2': sep2,
                    'sep3': sep3,
                    'sep4': sep4,
                    'name': name,
                }

            entities[dataset] = ent_dit
            relations[dataset] = rel_dit

        return entities, relations

    def resize_tokenizer(self):

        ent2text = dict() 
        for dataset in self.entities:
            ents = self.entities[dataset] 
            for ent in ents:
                assert ent not in ent2text  
                ent2text[ent] = ents[ent]

        ent_names = sorted(set([ent2text[ent]['name'] for ent in ent2text]))


        rel2text = dict()
        rel_names = list()
        for dataset in self.relations:
            rels = self.relations[dataset]
            for r in rels:
                rel2text[r] = rels[r]
                rel_names += [rels[r]['sep1'], rels[r]['sep2'], rels[r]['sep3'], rels[r]['sep4']]


        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': ent_names + rel_names})
        assert len(ent_names) + len(rel_names) == num_added_tokens

        ent_range = dict()
        for dataset in self.entities:
            dataset_ents = self.entities[dataset]
            ent_tokens = [ent2text[ent]['name'] for ent in dataset_ents]
            ent_token_ids = sorted(self.tokenizer.convert_tokens_to_ids(ent_tokens))
            ent_range[dataset] = ent_token_ids

        return ent2text, rel2text, ent_range

    def read_lines(self):
        def _read_triples(data_path):
            triples = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    h, r, t = line.strip().split('\t')
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

        return text_ent_filter

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
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token

        head, rel, tail = self.ent2text[h], self.rel2text[r], self.ent2text[t]
        h_name, h_desc, h_raw_name = head['name'], head['desc'], head['raw_name']

        t_name, t_desc, t_raw_name = tail['name'], tail['desc'], tail['desc']

        r_name = rel['name']
        sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']


        text_head_prompt = ' '.join([sep1, mask_token, sep2, r_name, sep3, t_name, sep4, t_desc])
        text_head_label = self.tokenizer.convert_tokens_to_ids(h_name)
        text_head_filters = list(self.text_ent_filter[t, r] - {self.tokenizer.convert_tokens_to_ids(h_name)})

        text_tail_prompt = ' '.join([sep1, h_name, sep2, r_name, sep3, mask_token, sep4, h_desc])
        text_tail_label = self.tokenizer.convert_tokens_to_ids(t_name)
        text_tail_filters = list(self.text_ent_filter[h, r] - {self.tokenizer.convert_tokens_to_ids(t_name)})


        head_example = {
            'triple': (t, r, h), 'text': (tail["raw_name"], r_name, head['raw_name'], 0),
            'text_prompt': text_head_prompt, 'text_label': text_head_label, 'text_filters': text_head_filters,
        }
        tail_example = {
            'triple': (h, r, t), 'text': (head['raw_name'], r_name, tail['raw_name'], 1),
            'text_prompt': text_tail_prompt, 'text_label': text_tail_label, 'text_filters': text_tail_filters,
        }

        return head_example, tail_example

    def _create_pretrain_examples(self, ent2text: dict, dataset: str):
        mask_token = self.tokenizer.mask_token
        train_examples, valid_examples, test_examples = list(), list(), list()

        for ent in ent2text.keys():
            name = ent2text[ent]['name']
            raw_name = str(ent2text[ent]['raw_name'])
            desc = str(ent2text[ent]['desc'])
            desc_tokens = desc.split()

            prompts = [f'{mask_token}, also known as {raw_name}, {desc}']
            begins = random.sample(range(0, len(desc_tokens)), min(10, len(desc_tokens)))
            for begin in begins:
                end = min(begin + self.max_seq_length, len(desc_tokens))
                new_desc = ' '.join(desc_tokens[begin: end])
                prompts.append(f'{mask_token}, also known as {raw_name}, {new_desc}')

            for i in range(len(prompts)):
                example = {'prompt': prompts[i], 'label': self.tokenizer.convert_tokens_to_ids(name), 'dataset': dataset}
                if i == len(prompts) - 1:
                    test_examples.append(example)
                elif i == len(prompts) - 2:
                    valid_examples.append(example)
                else:
                    example['dataset'] = None
                    train_examples.append(example)
        return train_examples, valid_examples, test_examples

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))

        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'mask_pos': mask_pos}

    def collate_fn(self, batch_data):
        if self.task == 'pretrain':
            return self.collate_fn_for_pretrain(batch_data)

        datasets = [data_dit['dataset'] for data_dit in batch_data]
        assert len(set(datasets)) == 1
        dataset = datasets[0]


        data_triple = [data_dit['triple'] for data_dit in batch_data]  
        data_text = [data_dit['text'] for data_dit in batch_data]

        text_prompts = [data_dit['text_prompt'] for data_dit in batch_data]
        text_data = self.text_batch_encoding(text_prompts)
        text_labels = torch.tensor([data_dit['text_label'] for data_dit in batch_data])
        text_filters = torch.tensor(
            [[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['text_filters']])
        text_ent_range = self.text_ent_range[dataset] if dataset is not None else None

        return {
            'triples': data_triple, 'texts': data_text,
            'text_data': text_data, 'text_labels': text_labels,
            'text_filters': text_filters, 'text_ent_range': text_ent_range,
        }

    def collate_fn_for_pretrain(self, batch_data):
        assert self.task == 'pretrain'
        datasets = [data_dit['dataset'] for data_dit in batch_data]
        assert len(set(datasets)) == 1
        dataset = datasets[0]

        lm_prompts = [data_dit['prompt'] for data_dit in batch_data]
        lm_data = self.text_batch_encoding(lm_prompts)

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])

        return {
            'text_data': lm_data, 'text_labels': labels, 'text_filters': None,
            'text_ent_range': self.text_ent_range[dataset] if dataset is not None else None
        }

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_dev_dataloader(self):
        data_loader = dict()
        for dataset in self.valid_dataset:
            data_loader[dataset] = DataLoader(self.valid_dataset[dataset], collate_fn=self.collate_fn,
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



class KGCDataModule:
    def __init__(self, args: dict, tokenizer, encode_text=False, encode_struc=False):
        self.task = args['task']
        self.data_path = args['data_path']
        self.dataset = self.data_path.split('/')[-1]
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.max_seq_length = args['max_seq_length'] if encode_text else -1

        self.add_neighbors = args['add_neighbors']
        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.no_relation_token = args['no_relation_token']

        self.encode_text = encode_text
        self.encode_struc = encode_struc

        self.entities, self.relations = self.read_support()
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}')

        self.tokenizer = tokenizer
        text_offset = self.resize_tokenizer()
        self.text_ent_begin = text_offset['text_entity_begin_idx']
        self.text_ent_end = text_offset['text_entity_end_idx']
        self.vocab, struc_offset = self.get_vocab()
        args.update(text_offset)
        args.update(struc_offset)
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = struc_offset['struc_relation_end_idx'] - struc_offset['struc_relation_begin_idx']

        self.lines = self.read_lines()  
        self.neighbors = self.get_neighbors()  
        self.entity_filter = self.get_entity_filter()

        if self.task == 'pretrain':
            examples = self.create_pretrain_examples()
        else:
            examples = self.create_examples()

        self.train_ds = KGCDataset(examples['train'])
        self.dev_ds = KGCDataset(examples['dev'])
        self.test_ds = KGCDataset(examples['test'])

    def read_support(self):
        """
        read entities and relations from files
        :return: two Python Dict objects
        """
        entity_path = os.path.join(self.data_path, 'entity.json')
        entities = json.load(open(entity_path, 'r', encoding='utf-8'))
        for idx, e in enumerate(entities): 
            new_name = f'[{self.dataset}_E_{idx}]'
            raw_name = entities[e]['name']
            desc = entities[e]['desc']
            entities[e] = {
                'token_id': idx,  
                'name': new_name,
                'desc': desc,
                'raw_name': raw_name,
            }

        relation_path = os.path.join(self.data_path, 'relation.json')
        relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        for idx, r in enumerate(relations): 
            sep1, sep2, sep3, sep4 = f'[{self.dataset}_R_{idx}_SEP1]', f'[{self.dataset}_R_{idx}_SEP2]', f'[{self.dataset}_R_{idx}_SEP3]', f'[{self.dataset}_R_{idx}_SEP4]'
            name = relations[r]['name']
            relations[r] = {
                'sep1': sep1,
                'sep2': sep2,
                'sep3': sep3,
                'sep4': sep4,
                'name': name,
            }

        return entities, relations

    def resize_tokenizer(self):

        entity_begin_idx = len(self.tokenizer)
        entity_names = [self.entities[e]['name'] for e in self.entities]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_names})
        entity_end_idx = len(self.tokenizer)

        relation_begin_idx = len(self.tokenizer)
        relation_names = [self.relations[r]['sep1'] for r in self.relations]
        relation_names += [self.relations[r]['sep2'] for r in self.relations]
        relation_names += [self.relations[r]['sep3'] for r in self.relations]
        relation_names += [self.relations[r]['sep4'] for r in self.relations]
        if self.add_neighbors:
            relation_names += [self.neighbor_token, self.no_relation_token]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relation_names})
        relation_end_idx = relation_begin_idx + 4 * len(self.relations) + 2

        return {
            'text_entity_begin_idx': entity_begin_idx,
            'text_entity_end_idx': entity_end_idx,
            'text_relation_begin_idx': relation_begin_idx,
            'text_relation_end_idx': relation_end_idx,
        }

    def get_vocab(self):
        tokens = ['[PAD]', '[MASK]', '[SEP]', self.no_relation_token]
        entity_names = [e for e in self.entities]
        relation_names = []
        for r in self.relations:
            relation_names += [r, f'{r}_reverse']

        entity_begin_idx = len(tokens)
        entity_end_idx = len(tokens) + len(entity_names)
        relation_begin_idx = len(tokens) + len(entity_names)
        relation_end_idx = len(tokens) + len(entity_names) + len(relation_names)

        tokens = tokens + entity_names + relation_names
        vocab = dict()
        for idx, token in enumerate(tokens):
            vocab[token] = idx

        return vocab, {
            'struc_entity_begin_idx': entity_begin_idx,
            'struc_entity_end_idx': entity_end_idx,
            'struc_relation_begin_idx': relation_begin_idx,
            'struc_relation_end_idx': relation_end_idx,
        }

    def read_lines(self):

        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'valid.txt'),
            'test': os.path.join(self.data_path, 'test.txt')
        }

        lines = dict()
        for mode in data_paths:
            data_path = data_paths[mode]
            raw_data = list()

            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = str(line).strip().split('\t')
                    raw_data.append((h, r, t))

            data = list()
            for h, r, t in raw_data:
                if (h in self.entities) and (t in self.entities) and (r in self.relations):
                    data.append((h, r, t))
            if len(raw_data) > len(data):
                raise ValueError('There are some triplets missing textual information')

            lines[mode] = data

        return lines

    def get_neighbors(self):
        """
        construct neighbor prompts from training dataset
        :return: {entity_id: {text_prompt: [], struc_prompt: []}, ...}
        """
        sep_token = self.tokenizer.sep_token
        mask_token = self.tokenizer.mask_token

        lines = self.lines['train']
        data = {e: {'text_prompt': [], 'struc_prompt': []} for e in self.entities}
        for h, r, t in lines:
            head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
            h_name, r_name, t_name = head['name'], rel['name'], tail['name']
            sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

            head_text_prompt = f'{sep1} {mask_token} {sep2} {r_name} {sep3} {t_name} {sep4}'
            head_struc_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
            data[h]['text_prompt'].append(head_text_prompt)
            data[h]['struc_prompt'].append(head_struc_prompt)
            tail_text_prompt = f'{sep1} {h_name} {sep2} {r_name} {sep3} {mask_token} {sep4}'
            tail_struc_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
            data[t]['text_prompt'].append(tail_text_prompt)
            data[t]['struc_prompt'].append(tail_struc_prompt)

        for ent in data:
            if len(data[ent]['text_prompt']) == 0:
                h_name = self.entities[ent]['name']
                text_prompt = ' '.join([h_name, sep_token, self.no_relation_token, sep_token, mask_token])
                struc_prompt = [self.vocab[ent], self.vocab[self.no_relation_token], self.vocab[mask_token]]
                data[ent]['text_prompt'].append(text_prompt)
                data[ent]['struc_prompt'].append(struc_prompt)

        return data

    def get_entity_filter(self):

        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)
        for h, r, t in lines:
            entity_filter[h, r].add(self.entities[t]['token_id'])
            entity_filter[t, r].add(self.entities[h]['token_id'])
        return entity_filter

    def create_examples(self):

        examples = dict()
        for mode in self.lines:
            data = list()
            lines = self.lines[mode]
            for h, r, t in tqdm(lines):
                head_example, tail_example = self.create_one_example(h, r, t)
                data.append(head_example)
                data.append(tail_example)
            examples[mode] = data
        return examples

    def create_one_example(self, h, r, t):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token
        neighbor_token = self.neighbor_token

        head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
        h_name, h_desc, h_raw_name = head['name'], head['desc'], head['raw_name']
        r_name = rel['name']
        t_name, t_desc, t_raw_name = tail['name'], tail['desc'], tail['desc']
        sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

        if self.encode_text:
            if self.add_neighbors:
                text_head_prompt = ' '.join(
                    [sep1, mask_token, sep2, r_name, sep3, t_name, neighbor_token, sep4, t_desc])
                text_tail_prompt = ' '.join(
                    [sep1, h_name, neighbor_token, sep2, r_name, sep3, mask_token, sep4, h_desc])
            else:
                text_head_prompt = ' '.join([sep1, mask_token, sep2, r_name, sep3, t_name, t_raw_name, sep4, t_desc])
                text_tail_prompt = ' '.join([sep1, h_name, h_raw_name, sep2, r_name, sep3, mask_token, sep4, h_desc])
        else:
            text_head_prompt, text_tail_prompt = None, None
        if self.encode_struc:
            struc_head_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
            struc_tail_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
        else:
            struc_head_prompt, struc_tail_prompt = None, None
        head_filters = list(self.entity_filter[t, r] - {head['token_id']})
        tail_filters = list(self.entity_filter[h, r] - {tail['token_id']})
        head_example = {
            'data_triple': (t, r, h),
            'data_text': (tail["raw_name"], r_name, head['raw_name']),
            'text_prompt': text_head_prompt,
            'struc_prompt': struc_head_prompt,
            'neighbors_label': tail['token_id'],
            'label': head["token_id"],
            'filters': head_filters,
        }
        tail_example = {
            'data_triple': (h, r, t),
            'data_text': (head['raw_name'], r_name, tail['raw_name']),
            'text_prompt': text_tail_prompt,
            'struc_prompt': struc_tail_prompt,
            'neighbors_label': head['token_id'],
            'label': tail["token_id"],
            'filters': tail_filters,
        }

        return head_example, tail_example

    def create_pretrain_examples(self):
        examples = dict()
        for mode in ['train', 'dev', 'test']:
            data = list()
            for h in self.entities.keys():
                name = str(self.entities[h]['name'])
                desc = str(self.entities[h]['desc'])
                desc_tokens = desc.split()

                prompts = [f'The description of {self.tokenizer.mask_token} is that {desc}']
                for i in range(10):
                    begin = random.randint(0, len(desc_tokens))
                    end = min(begin + self.max_seq_length, len(desc_tokens))
                    new_desc = ' '.join(desc_tokens[begin: end])
                    prompts.append(f'The description of {self.tokenizer.mask_token} is that {new_desc}')
                for prompt in prompts:
                    data.append({'prompt': prompt, 'label': self.entities[h]['token_id']})
            examples[mode] = data
        return examples

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
        if self.task == 'pretrain':
            return self.collate_fn_for_pretrain(batch_data)

        data_triple = [data_dit['data_triple'] for data_dit in batch_data]
        data_text = [data_dit['data_text'] for data_dit in batch_data]

        text_prompts = [data_dit['text_prompt'] for data_dit in batch_data]
        text_data = self.text_batch_encoding(text_prompts) if self.encode_text else None
        struc_prompts = [copy.deepcopy(data_dit['struc_prompt']) for data_dit in batch_data]
        struc_data = self.struc_batch_encoding(struc_prompts) if self.encode_struc else None

        if self.add_neighbors:
            batch_text_neighbors = [[] for _ in range(self.neighbor_num)]
            batch_struc_neighbors = [[] for _ in range(self.neighbor_num)]
            for ent, _, _ in data_triple:
                text_neighbors, struc_neighbors = self.neighbors[ent]['text_prompt'], self.neighbors[ent]['struc_prompt']
                idxs = list(range(len(text_neighbors)))
                if len(idxs) >= self.neighbor_num:
                    idxs = random.sample(idxs, self.neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.neighbor_num
                for i, idx in enumerate(idxs):
                    batch_text_neighbors[i].append(text_neighbors[idx])
                    batch_struc_neighbors[i].append(struc_neighbors[idx])
            text_neighbors = [self.text_batch_encoding(batch_text_neighbors[i]) for i in range(self.neighbor_num)] \
                if self.encode_text else None
            struc_neighbors = [self.struc_batch_encoding(batch_struc_neighbors[i]) for i in range(self.neighbor_num)] \
                if self.encode_struc else None
        else:
            text_neighbors, struc_neighbors = None, None

        neighbors_labels = torch.tensor([data_dit['neighbors_label']for data_dit in batch_data]) \
            if self.add_neighbors else None
        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'data': data_triple, 'data_text': data_text,
            'text_data': text_data, 'text_neighbors': text_neighbors,
            'struc_data': struc_data, 'struc_neighbors': struc_neighbors,
            'labels': labels, 'filters': filters, 'neighbors_labels': neighbors_labels,
            'text_ent_range': [self.text_ent_begin, self.text_ent_end]
        }

    def collate_fn_for_pretrain(self, batch_data):
        assert self.task == 'pretrain'

        lm_prompts = [data_dit['prompt'] for data_dit in batch_data]  # [string, ...]
        lm_data = self.text_batch_encoding(lm_prompts)

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])

        return {'text_data': lm_data, 'labels': labels, 'filters': None}

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

    def get_tokenizer(self):
        return self.tokenizer


if __name__ == '__main__':
    pass
