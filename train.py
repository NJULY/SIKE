import os
import json
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time, strftime, localtime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import BertConfig, BertTokenizer, BertForMaskedLM

from data_process import DataModule
from model import SIKE
from utils import get_scores, score2str, read_all_align_dicts


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
DATASETS = ['YAGO3', 'FB15K237', 'WN18RR']
DATASET2ID = {dataset: idx for idx, dataset in enumerate(DATASETS)}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', help='train | validate')
    parser.add_argument('--data_path', type=str, default='dataset/DWY15K')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--kg1', type=str, default='DBpedia15K')
    parser.add_argument('--kg0', type=str, default='none')
    parser.add_argument('--embedding_lr', type=float, default=2e-3)
    parser.add_argument('--adapter_lr', type=float, default=5e-4, )
    parser.add_argument('--alpha', type=float, default=1e4, help='forward transfer weight')
    parser.add_argument('--beta', type=float, default=-1., help='backward transfer weight')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_norm', type=float, default=-1.0)
    parser.add_argument('--adapter_hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--label_smoothing', type=float, default=0.8)
    parser.add_argument('--bert_path', type=str, default='checkpoints/bert-base-cased')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--pin_memory', type=bool, default=True)
    args = parser.parse_args()
    args = vars(args)

    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join('output', args['kg1'], timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['output_path'] = output_dir

    # save hyper params
    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)

    # set random seed
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    return args


class ContinualTrainer:
    def __init__(self, config: dict):
        self.task = config['task']
        self.output_path = config['output_path']
        self.epoch = config['epoch']
        self.device = config['device']
        self.embedding_lr = config['embedding_lr']
        self.adapter_lr = config['adapter_lr']
        self.forward_weight = config['alpha']
        self.backward_weight = config['beta']
        self.max_norm = config['max_norm']

        self.kg0 = config['kg0'] if config['kg0'] != 'none' else None
        self.kg1 = config['kg1']

        self.data_loaders = dict()
        for dataset in DATASET2ID:
            data_path = os.path.join(config['data_path'], dataset)
            print(f'Load dataset {dataset} from path {data_path}')
            data_module = DataModule(
                kg_name=dataset, kg_path=data_path, tokenizer_path=config['bert_path'], num_soft_prompt=4,
                max_seq_length=config['max_seq_length'], batch_size=config['batch_size'],
                num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            )
            self.data_loaders[dataset] = {
                'train': data_module.train_dl,
                'valid': data_module.valid_dl,
                'test': data_module.test_dl,
                'ents': data_module.entities,
                'rels': data_module.relations,
                'vocab_size': len(data_module.vocab)
            }

        model_path = config['model_path']
        if model_path == 'none':

            vocab_sizes = [self.data_loaders[dataset]['vocab_size'] for dataset in self.data_loaders]
            self.model = SIKE(vocab_sizes, config['bert_path'], config['adapter_hidden_size'],
                              config['label_smoothing']).to(self.device)
        else:

            vocab_sizes = [self.data_loaders[dataset]['vocab_size'] for dataset in self.data_loaders]
            self.model = SIKE.from_pretrained(model_path, vocab_sizes).to(self.device)

        kg2entities = {kg: self.data_loaders[kg]['ents'] for kg in DATASET2ID}
        entity_alignment = read_all_align_dicts(config['data_path'])
        self.model.load_variables(DATASET2ID, kg2entities, entity_alignment, config['alpha'], config['beta'])

    def _train_one_epoch(self, model, data_loader, optimizer, scheduler, epoch_idx):
        model.train()
        outputs = list()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch_loss = model.training_step(batch, batch_idx, self.kg1, self.kg0)
            outputs.append(batch_loss)
            optimizer.zero_grad()
            batch_loss.backward()
            if self.max_norm > 0:
                clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        loss = model.training_epoch_end(outputs)
        return loss

    def _validate_one_epoch(self, model, data_loader, kg_name, output_path=None):
        self.model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_output = model.validation_step(batch, batch_idx, kg_name, output_path)
                outputs.append(batch_output)

        score = model.validation_epoch_end(outputs)
        return score

    def train(self):
        for dataset in self.data_loaders:
            valid_dataloader = self.data_loaders[dataset]['valid']
            test_dataloader = self.data_loaders[dataset]['test']

            valid_score = self._validate_one_epoch(self.model, valid_dataloader, dataset)
            print(f'[{dataset}-valid] {score2str(valid_score)}')
            test_score = self._validate_one_epoch(self.model, test_dataloader, dataset)
            print(f'[{dataset}-test]  {score2str(test_score)}')

        train_dataloader = self.data_loaders[self.kg1]['train']
        valid_dataloader = self.data_loaders[self.kg1]['valid']
        test_dataloader = self.data_loaders[self.kg1]['test']

        optimizer, scheduler = self.model.get_optimizer(
            total_steps=len(train_dataloader)*self.epoch, kg1=self.kg1, kg0=self.kg0,
            adapter_lr=self.adapter_lr, embedding_lr=self.embedding_lr
        )

        # 创建日志
        log_path = os.path.join(self.output_path, f'log.txt')
        with open(log_path, 'w') as f:
            pass

        best_score = None
        best_mrr = 0.
        for i in range(1, self.epoch + 1):
            begin_time = time()
            train_loss = self._train_one_epoch(self.model, train_dataloader, optimizer, scheduler, epoch_idx=i)
            if self.kg0 is not None and self.backward_weight > 0.:
                valid_score = self._validate_one_epoch(self.model, self.data_loaders[self.kg0]['valid'], self.kg0)
                test_score = self._validate_one_epoch(self.model, self.data_loaders[self.kg0]['test'], self.kg0)
            else:
                valid_score = self._validate_one_epoch(self.model, valid_dataloader, self.kg1)
                test_score = self._validate_one_epoch(self.model, test_dataloader, self.kg1)
            # update the best score and save the best model
            if best_mrr < test_score['MRR']:
                best_mrr = test_score['MRR']
                best_score = test_score
                best_score['epoch'] = i
                self.model.save_pretrained(os.path.join(self.output_path, f'model'))

            # save log of this epoch
            log = f'[train] epoch: {i}, loss: {train_loss}' + '\n'
            log += f'[valid] epoch: {i}, ' + score2str(valid_score) + '\n'
            log += f'[test]  epoch: {i}, ' + score2str(test_score) + '\n'
            log += '=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30 + '\n'
            print(log)
            with open(log_path, 'a') as f:
                f.write(log + '\n')
        # save the log of best epoch
        log = f'[best]  epoch: {best_score["epoch"]}, ' + score2str(best_score)
        print(log)
        with open(log_path, 'a') as f:
            f.write(log + '\n')

    def validate(self):
        shutil.rmtree(self.output_path)

        valid_dataloader = self.data_loaders[self.kg1]['valid']
        test_dataloader = self.data_loaders[self.kg1]['test']

        valid_score = self._validate_one_epoch(self.model, valid_dataloader, self.kg1)
        print(f'[{self.kg1}-valid-text] {score2str(valid_score)}')
        test_score = self._validate_one_epoch(self.model, test_dataloader, self.kg1)
        print(f'[{self.kg1}-test-text]  {score2str(test_score)}')

    def main(self):
        if self.task == 'train':
            self.train()
        elif self.task == 'validate':
            self.validate()
        else:
            print(f'Unknown Task: {self.task}')


if __name__ == '__main__':
    config = get_args()
    trainer = ContinualTrainer(config)
    trainer.main()
