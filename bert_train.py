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

DATASETS = ['DBpedia15K', 'Wikidata15K', 'Yago15K']
DATASET2ID = {dataset: idx for idx, dataset in enumerate(DATASETS)}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', help='train | validate')
    parser.add_argument('--data_path', type=str, default='dataset/DWY15K', help='在本项目中不需要改动')

    parser.add_argument('--model_path', type=str, default='output/Yago15K/20221010_064028/model')
    parser.add_argument('--struct_model_path', type=str, default='none')

    parser.add_argument('--dataset', type=str, default='Yago15K', help='DBpedia15K Wikidata15K Yago15K')
    parser.add_argument('--support_dataset', type=str, default='Wikidata15K', help='DBpedia15K Wikidata15K Yago15K none')

    parser.add_argument('--embedding_lr', type=float, default=2e-3)
    parser.add_argument('--adapter_lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=10000.0, help='MSE蒸馏损失的权重, 设置为负数表示不使用此类蒸馏')
    parser.add_argument('--beta', type=float, default=1.0, help='结构向文本蒸馏的权重, 设置为负数表示不使用此类蒸馏')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--max_seq_length', type=int, default=64, help='BERT输入的最大长度, 理论上越大效果越好, 训练时间也越长')
    parser.add_argument('--label_smoothing', type=float, default=0.8, help='label smoothing for language model')
    parser.add_argument('--bert_path', type=str, default='checkpoints/bert-base-cased', help='用于加载分词器和原始BERT')
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    args = parser.parse_args()
    args = vars(args)
    assert args['dataset'] in DATASETS

    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join('output', args['dataset'], timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['output_path'] = output_dir

    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)

    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    return args


class ContinualTrainer:
    def __init__(self, config: dict):
        self.task = config['task']
        self.output_path = config['output_path']
        self.epoch = config['epoch']
        self.device = config['device']
        self.embedding_lr = config['embedding_lr']
        self.adapter_lr = config['adapter_lr']
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.dataset = config['dataset']
        self.module_id = DATASET2ID[self.dataset]

        self.support_dataset = config['support_dataset']
        self.support_module_id = DATASET2ID[self.support_dataset] if self.support_dataset != 'none' else -1

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

        align_dicts = read_all_align_dicts(config['data_path'])
        self.align_info = {
            'align_dict': align_dicts[self.dataset][self.support_dataset],
            'current_ents': self.data_loaders[self.dataset]['ents'],
            'support_ents': self.data_loaders[self.support_dataset]['ents'],
        } if self.support_dataset != 'none' else None

        model_path = config['model_path']
        if model_path == 'none':
            print('从零开始构建模型')
            bert_path = config['bert_path']
            vocab_sizes = [self.data_loaders[dataset]['vocab_size'] for dataset in self.data_loaders]
            adapter_config = {'num_adapters': len(vocab_sizes), 'adapter_hidden_size': 128}
            self.model = ContinualBert(bert_path, vocab_sizes, adapter_config, config['label_smoothing']).to(self.device)
        else:
            print(f'加载已有模型: {model_path}')
            self.model = ContinualBert.from_pretrained(config['model_path']).to(self.device)
        struct_model_path = config['struct_model_path']
        if struct_model_path == 'none':
            print('没有结构模型')
            self.struct_model = None
        else:
            print(f'加载结构模型: {struct_model_path}')
            self.struct_model = ContinualFormer.from_pretrained(struct_model_path).to(self.device)
            for p in self.struct_model.parameters():
                p.requires_grad = False

    def _train_one_epoch(self, model, data_loader, optimizer, scheduler, epoch_idx):
        model.train()
        outputs = list()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch['module_id'] = self.module_id
            batch['alpha'] = self.alpha
            batch['support_module_id'] = self.support_module_id
            batch['align_info'] = self.align_info
            batch['beta'] = self.beta
            batch['struct_model'] = self.struct_model
            batch['device'] = self.device

            batch_loss = model.training_step(batch, batch_idx)
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

    def _validate_one_epoch(self, model, module_id, data_loader):
        self.model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch['module_id'] = module_id
                batch['device'] = self.device
                batch['align_info'] = self.align_info

                batch_output = model.validation_step(batch, batch_idx)
                outputs.append(batch_output)

        scores = model.validation_epoch_end(outputs)
        return scores

    def train(self):
        print('在所有数据集上面测试模型')
        for module_id, dataset in enumerate(self.data_loaders):
            valid_dataloader = self.data_loaders[dataset]['valid']
            test_dataloader = self.data_loaders[dataset]['test']

            dev_scores = self._validate_one_epoch(self.model, module_id, valid_dataloader)
            test_scores = self._validate_one_epoch(self.model, module_id, test_dataloader)
            print(f'{dataset} [valid] {score2str(dev_scores)}')
            print(f'{dataset} [test]  {score2str(test_scores)}')

        train_dataloader = self.data_loaders[self.dataset]['train']
        valid_dataloader = self.data_loaders[self.dataset]['valid']
        test_dataloader = self.data_loaders[self.dataset]['test']

        print(f'创建{self.dataset}的优化器与调度器')
        optimizer, scheduler = self.model.get_optimizer(
            total_steps=len(train_dataloader)*self.epoch, module_id=self.module_id,
            adapter_lr=self.adapter_lr, embedding_lr=self.embedding_lr,
        )

        log_path = os.path.join(self.output_path, f'log.txt')
        with open(log_path, 'w') as f:
            pass

        best_score = None
        best_mrr = 0.
        for i in range(1, self.epoch + 1):
            begin_time = time()
            train_loss = self._train_one_epoch(
                model=self.model, data_loader=train_dataloader, optimizer=optimizer, scheduler=scheduler, epoch_idx=i,
            )
            dev_score = self._validate_one_epoch(self.model, self.module_id, valid_dataloader)
            test_score = self._validate_one_epoch(self.model, self.module_id, test_dataloader)

            if best_mrr < test_score['MRR']:
                best_mrr = test_score['MRR']
                best_score = test_score
                best_score['epoch'] = i
                self.model.save_pretrained(os.path.join(self.output_path, f'model'))

            log = f'[train] epoch: {i}, loss: {train_loss}' + '\n'
            log += f'[valid] epoch: {i}, ' + score2str(dev_score) + '\n'
            log += f'[test]  epoch: {i}, ' + score2str(test_score) + '\n'
            log += '=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30 + '\n'
            print(log)
            with open(log_path, 'a') as f:
                f.write(log + '\n')
        log = f'[best]  epoch: {best_score["epoch"]}, ' + score2str(best_score)
        print(log)
        with open(log_path, 'a') as f:
            f.write(log + '\n')

    def validate(self):
        shutil.rmtree(self.output_path)

        dataset_name = self.dataset
        module_id = DATASET2ID[dataset_name]
        valid_dataloader = self.data_loaders[dataset_name]['valid']
        test_dataloader = self.data_loaders[dataset_name]['test']

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
