import os
import json
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, AutoTokenizer

from data import SIKEDataModule as DataModule
from models import SIKE
from utils import read_all_align_dicts
from utils import mk_dirs_with_timestamp, get_logger


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
DATASETS = ['DBpedia15K', 'Wikidata15K', 'Yago15K']
DATASET2ID = {dataset: idx for idx, dataset in enumerate(DATASETS)}
# CUDA_VISIBLE_DEVICES=1

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--kg1', type=str, default=None, help='DBpedia15K Wikidata15K Yago15K')
    parser.add_argument('--kg0', type=str, default=None, help='DBpedia15K Wikidata15K Yago15K')
    parser.add_argument('--embedding_lr', type=float, default=2e-3)
    parser.add_argument('--adapter_lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=-1.0, help='forward transfer weight')
    parser.add_argument('--back_transfer', action='store_true')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--adapter_hidden_size', type=int, default=128)
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--ea_t', type=float, default=0.999, help='threshold for EA')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--is_test', action='store_true')
    parser.add_argument('--data_path', type=str, default='dataset/DWY15K')
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--label_smoothing', type=float, default=0.8, help='label smoothing for language model')
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--pin_memory', type=bool, default=True)
    
    args = parser.parse_args()
    args = vars(args)
    assert args['kg1'] in DATASETS

    args['output_path'] = mk_dirs_with_timestamp(prefix=os.path.join('output', args['kg1']))

    # set random seed
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    return args


class SIKETrainer:
    def __init__(self, args: dict):
        self.args = args

        self.output_path = args['output_path']
        self.epoch = args['epoch']
        self.embedding_lr = args['embedding_lr']
        self.adapter_lr = args['adapter_lr']

        self.kg0 = args['kg0']
        self.kg1 = args['kg1']
        self.do_backward_transfer = args['back_transfer']

        self.logger = get_logger(args['output_path'])
        self.logger.info(json.dumps(args, indent=4, ensure_ascii=False))
        self.scaler = torch.cuda.amp.GradScaler() if args['use_amp'] else None

        self.data_modules = dict()
        for dataset in DATASET2ID:
            data_module = DataModule(args, dataset)
            self.data_modules[dataset] = {
                'train': data_module.train_loader,
                'valid': data_module.valid_loader,
                'test': data_module.test_loader,
                'ents': data_module.ent2text,
                'rels': data_module.rel2text,
                'num_added_tokens': data_module.num_added_tokens,
            }
        args['datasets'] = DATASETS
        args['num_added_tokens'] = {dataset: self.data_modules[dataset]['num_added_tokens'] for dataset in self.data_modules}

        self.model = SIKE(args)
        model_path = args['model_path']
        if model_path is not None:
            print(f'Load model from {model_path}')
            self.model.load_model(args['model_path'])
        self.model.cuda()

        kg2ents = {kg: self.data_modules[kg]['ents'] for kg in DATASET2ID}
        entity_alignment = read_all_align_dicts(os.path.join(args['data_path'], 'links'), kgs=DATASETS, threshold=args['ea_t'])
        self.model.load_variables(kg2ents, entity_alignment)

    def _train_one_epoch(self, model, data_loader, optimizer, scheduler, epoch_idx):
        model.train()
        losses = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    batch_loss = model.training_step(batch, batch_idx, self.kg1, self.kg0)
            else:
                batch_loss = model.training_step(batch, batch_idx, self.kg1, self.kg0)
            losses.append(batch_loss.item())

            # backward
            optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(batch_loss).backward()
                self.scaler.unscale_(optimizer)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()
            
            if batch_idx == 0:
                print(f'grad norms: {model.grad_norm()}')
            if scheduler is not None:
                scheduler.step()
            
        return np.round(np.mean(losses), 3)

    def _validate_one_epoch(self, model, data_loader, kg_name, save_result=False):
        model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_output = model.validation_step(batch, batch_idx, kg_name, save_result)
                outputs.append(batch_output)

        score = model.validation_epoch_end(outputs)
        return score

    def train(self):
        print('Validate on all KGs')
        for kg_name in DATASETS:
            test_dataloader = self.data_modules[kg_name]['test']
            test_score = self._validate_one_epoch(self.model, test_dataloader, kg_name)
            self.logger.info(f'{kg_name} test: {json.dumps(test_score)}')

        # train valid test datasets
        train_dataloader = self.data_modules[self.kg1]['train']
        print(f'Create optimizer for {self.kg1 if not self.do_backward_transfer else self.kg0}')
        optimizer, scheduler = self.model.get_optimizer(total_steps=len(train_dataloader)*self.epoch,
                                                        kg0=self.kg0, kg1=self.kg1,
                                                        adapter_lr=self.adapter_lr, embedding_lr=self.embedding_lr)
        best_mrr = 0.
        best_score = None
        bad_count = 0
        for i in range(1, self.epoch + 1):
            begin_time = time()
            train_loss = self._train_one_epoch(model=self.model, data_loader=train_dataloader, optimizer=optimizer,
                                               scheduler=scheduler, epoch_idx=i)
            if self.do_backward_transfer:
                valid_score = self._validate_one_epoch(self.model, self.data_modules[self.kg0]['valid'], self.kg0)
                test_score = self._validate_one_epoch(self.model, self.data_modules[self.kg0]['test'], self.kg0)
            else:
                valid_score = self._validate_one_epoch(self.model, self.data_modules[self.kg1]['valid'], self.kg1)
                test_score = self._validate_one_epoch(self.model, self.data_modules[self.kg1]['test'], self.kg1)
            epoch_time = round(time() - begin_time)
            self.logger.info(f'Epoch {i}, train loss: {train_loss}, time: {epoch_time}s\nvalid score: {json.dumps(valid_score)}\ntest score: {json.dumps(test_score)}, ')

            # update the best score and save the best model
            if best_mrr < valid_score['mrr']:
                best_mrr = valid_score['mrr']
                best_score = test_score
                best_score['epoch'] = i
                self.model.save_model(os.path.join(self.output_path, 'model.bin'))
                bad_count = 0
            else:
                bad_count += 1
            
            if bad_count >= self.args['early_stop']:
                break

        # save the log of best epoch
        self.logger.info(f'[best] epoch: {best_score["epoch"]}, score: {json.dumps(best_score)}')
    
    def validate(self):
        dataset_name = self.kg1
        test_dataloader = self.data_modules[dataset_name]['test']

        test_score = self._validate_one_epoch(self.model, test_dataloader, self.kg1, save_result=True)
        print(f'{dataset_name} [test]: {json.dumps(test_score)}')
        shutil.rmtree(self.output_path)

    def main(self):
        if self.args['is_test']:
            self.validate()
        else:
            self.train()


if __name__ == '__main__':
    args = get_args()
    trainer = SIKETrainer(args)
    trainer.main()