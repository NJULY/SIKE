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

from data_process import ContinualDataModule as DataModule
from models import ContinualBert, ContinualFormer
from utils import get_scores, score2str, read_all_align_dicts


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
DATASETS = ['DBpedia15K', 'Wikidata15K', 'Yago15K', 'FB15K237']
DATASET2ID = {dataset: idx for idx, dataset in enumerate(DATASETS)}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='none', help='folder path for saved models')
    parser.add_argument('--kg1', type=str)
    parser.add_argument('--kg0', type=str)
    parser.add_argument('--embedding_lr', type=float)
    parser.add_argument('--adapter_lr', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--device', type=str)

    parser.add_argument('--task', type=str, default='train', help='train | validate')
    parser.add_argument('--data_path', type=str, default='dataset/DWY15K')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--label_smoothing', type=float, default=0.8)
    parser.add_argument('--bert_path', type=str, help='input your folder path for BERT')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--pin_memory', type=bool, default=True)

    args = parser.parse_args()
    args = vars(args)
    assert args['kg1'] in DATASETS

    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join('output', args['kg1'], timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['output_path'] = output_dir

    # save hyper params
    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)

    return args


class ContinualTrainer:
    def __init__(self, config: dict):
        self.task = config['task']
        self.output_path = config['output_path']
        self.epoch = config['epoch']
        self.device = config['device']
        self.embedding_lr = config['embedding_lr']
        self.adapter_lr = config['adapter_lr']

        self.kg0 = config['kg0'] if config['kg0'] != 'none' else None
        self.kg1 = config['kg1']
        self.do_backward_transfer = True if self.kg0 is not None and config['beta'] > 0. else False

        tokenizer_path = config['bert_path']
        print(f'Loading Tokenizer from {tokenizer_path}')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)

        self.data_loaders = dict()
        for dataset in DATASET2ID:
            data_path = os.path.join(config['data_path'], dataset)
            print(f'Load dataset {dataset} from path {data_path}')
            data_module = DataModule(
                data_path=data_path, tokenizer=self.tokenizer, is_pretrain=False, encode_text=True, encode_struc=True,
                max_seq_length=config['max_seq_length'], batch_size=config['batch_size'],
                num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            )
            self.data_loaders[dataset] = {
                'train': data_module.get_train_dataloader(),
                'valid': data_module.get_valid_dataloader(),
                'test': data_module.get_test_dataloader(),
                'ents': data_module.ents,
                'rels': data_module.rels,
                'vocab_size': data_module.text_vocab_size,
            }

        model_path = config['model_path']
        if model_path == 'none':
            bert_path = config['bert_path']
            vocab_sizes = [self.data_loaders[dataset]['vocab_size'] for dataset in self.data_loaders]
            adapter_config = {'num_adapters': len(vocab_sizes), 'adapter_hidden_size': 128}
            self.model = ContinualBert(bert_path, vocab_sizes, adapter_config, config['label_smoothing']).to(self.device)
        else:
            self.model = ContinualBert.from_pretrained(config['model_path']).to(self.device)
        kg2ents = {kg: self.data_loaders[kg]['ents'] for kg in DATASET2ID}
        entity_alignment = read_all_align_dicts(config['data_path'])
        self.model.load_variables(DATASET2ID, kg2ents, entity_alignment, config['alpha'], config['beta'])

    def _train_one_epoch(self, model, data_loader, optimizer, scheduler, epoch_idx):
        model.train()
        outputs = list()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch_loss = model.training_step(batch, batch_idx, self.kg1, self.kg0)
            outputs.append(batch_loss)
            optimizer.zero_grad()
            batch_loss.backward()
            if batch_idx == 0:
                print(f'梯度范数: {model.grad_norm()}')
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        loss = model.training_epoch_end(outputs)
        return loss

    def _validate_one_epoch(self, model, data_loader, kg_name):
        model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_output = model.validation_step(batch, batch_idx, kg_name)
                outputs.append(batch_output)

        scores = model.validation_epoch_end(outputs)
        return scores

    def train(self):
        # create log
        log_path = os.path.join(self.output_path, f'log.txt')
        with open(log_path, 'w') as f:
            pass

        # train valid test datasets
        train_dataloader = self.data_loaders[self.kg1]['train']
        valid_dataloader = self.data_loaders[self.kg1]['valid']
        test_dataloader = self.data_loaders[self.kg1]['test']
        print(f'Create optimizer for {self.kg1 if not self.do_backward_transfer else self.kg0}')
        optimizer, scheduler = self.model.get_optimizer(total_steps=len(train_dataloader)*self.epoch,
                                                        kg0=self.kg0, kg1=self.kg1,
                                                        adapter_lr=self.adapter_lr, embedding_lr=self.embedding_lr)
        best_score = None
        best_mrr = 0.
        for i in range(1, self.epoch + 1):
            begin_time = time()
            train_loss = self._train_one_epoch(model=self.model, data_loader=train_dataloader, optimizer=optimizer,
                                               scheduler=scheduler, epoch_idx=i)
            if self.do_backward_transfer:
                valid_score = self._validate_one_epoch(self.model, self.data_loaders[self.kg0]['valid'], self.kg0)
                test_score = self._validate_one_epoch(self.model, self.data_loaders[self.kg0]['test'], self.kg0)
            else:
                valid_score = self._validate_one_epoch(self.model, valid_dataloader, self.kg1)
                test_score = self._validate_one_epoch(self.model, test_dataloader, self.kg1)

            # update the best scores and save the best model
            if best_mrr < valid_score['MRR']:
                best_mrr = valid_score['MRR']
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

        dataset_name = self.kg1
        module_id = DATASET2ID[dataset_name]
        valid_dataloader = self.data_loaders[dataset_name]['valid']
        test_dataloader = self.data_loaders[dataset_name]['test']

        dev_scores = self._validate_one_epoch(self.model, module_id, valid_dataloader)
        print(f'{dataset_name} [valid] {score2str(dev_scores)}')
        test_scores = self._validate_one_epoch(self.model, module_id, test_dataloader)
        print(f'{dataset_name} [test]  {score2str(test_scores)}')

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