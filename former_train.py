import os
import json
import torch
import shutil
import random
import argparse
from tqdm import tqdm
import numpy as np
from time import time, strftime, localtime
from transformers import BertTokenizer

from data_process import ContinualDataModule as DataModule
from models import ContinualFormer
from utils import score2str, read_all_align_dicts


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
DATASETS = ['DBpedia15K', 'Wikidata15K', 'Yago15K']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', help='train | validate')
    parser.add_argument('--model_path', type=str, default='none')
    parser.add_argument('--tokenizer_path', type=str, default='checkpoints/bert-base-cased')
    parser.add_argument('--data_path', type=str, default='dataset/DWY15K')

    parser.add_argument('--dataset', type=str, default='Yago15K', help='DBpedia15K Wikidata15K Yago15K')
    parser.add_argument('--support_dataset', type=str, default='Wikidata15K', help='DBpedia15K Wikidata15K Yago15K')
    parser.add_argument('--support_model_path', type=str, default='output/Transformer/20221011_201746/model')
    parser.add_argument('--mse_weight', type=float, default=10000.0, help='MSE蒸馏损失的权重, 设置为负数表示不使用此类蒸馏')
    parser.add_argument('--device', type=str, default='cuda:3', help='select a gpu like cuda:0')

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epoch', type=int, default=300, help='epoch')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--label_smoothing', type=float, default=0.8)

    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--input_dropout_prob', type=float, default=0.5)
    parser.add_argument('--context_dropout_prob', type=float, default=0.1)
    parser.add_argument('--attention_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--residual_dropout_prob', type=float, default=0.)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    args = parser.parse_args()
    args = vars(args)

    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join('output', 'Transformer', timestamp)
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
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return args


def get_model_config(config, vocab_size, num_rels):
    model_config = dict()
    model_config["hidden_size"] = config['hidden_size']
    model_config["num_hidden_layers"] = config['num_hidden_layers']
    model_config["num_attention_heads"] = config['num_attention_heads']
    model_config["input_dropout_prob"] = config['input_dropout_prob']
    model_config["attention_dropout_prob"] = config['attention_dropout_prob']
    model_config["hidden_dropout_prob"] = config['hidden_dropout_prob']
    model_config["residual_dropout_prob"] = config['residual_dropout_prob']
    model_config["context_dropout_prob"] = config['context_dropout_prob']
    model_config["initializer_range"] = config['initializer_range']
    model_config["intermediate_size"] = config['intermediate_size']

    model_config["vocab_size"] = vocab_size
    model_config["num_relations"] = num_rels
    return model_config


class Trainer:
    def __init__(self, config: dict):
        self.task = config['task']
        self.output_path = config['output_path']
        self.epoch = config['epoch']
        self.device = config['device']
        self.lr = config['lr']
        self.mse_weight = config['mse_weight']
        self.dataset = config['dataset']
        self.support_dataset = config['support_dataset']

        tokenizer_path = config['tokenizer_path']
        print(f'Loading Tokenizer from {tokenizer_path}')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)

        self.data_loaders = dict()
        for dataset in DATASETS:
            data_path = os.path.join(config['data_path'], dataset)
            print(f'Load dataset {dataset} from path {data_path}')
            data_module = DataModule(
                data_path=data_path, tokenizer=self.tokenizer, is_pretrain=False, encode_text=False, encode_struc=True,
                max_seq_length=64, batch_size=config['batch_size'],
                num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            )
            self.data_loaders[dataset] = {
                'train': data_module.get_train_dataloader(),
                'valid': data_module.get_valid_dataloader(),
                'test': data_module.get_test_dataloader(),
                'ents': data_module.ents,
                'rels': data_module.rels,
                'vocab_size': data_module.struc_vocab_size,
                'num_rels': data_module.struc_rels_num
            }

        if config['support_model_path'] != 'none':
            support_model = ContinualFormer.from_pretrained(config['support_model_path']).to(self.device)
            for p in support_model.parameters():
                p.requires_grad = False
            support_embeds = support_model.encoder.ele_embedding

            align_dicts = read_all_align_dicts(config['data_path'])
            self.supports = {
                'align_dict': align_dicts[self.dataset][self.support_dataset],
                'current_ents': self.data_loaders[self.dataset]['ents'],
                'former_ents': self.data_loaders[self.support_dataset]['ents'],
                'former_embeds': support_embeds,
            }
        else:
            self.supports = None

        if config['model_path'] == 'none':
            print('重新构建模型')
            model_config = get_model_config(
                config, self.data_loaders[self.dataset]['vocab_size'],
                self.data_loaders[self.dataset]['num_rels']
            )
            self.model = ContinualFormer(encoder_config=model_config, label_smoothing=config['label_smoothing']).to(
                self.device)
        else:
            self.model = ContinualFormer.from_pretrained(config['model_path']).to(self.device)

    def _train_one_epoch(self, model, dataloader, optimizer, scheduler, epoch_idx):
        model.train()
        outputs = list()
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data['device'] = self.device
            batch_data['supports'] = self.supports
            batch_data['mse_weight'] = self.mse_weight

            batch_loss = model.training_step(batch_data, batch_idx)
            outputs.append(batch_loss)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss = model.training_epoch_end(outputs)
        return loss

    def _validate_one_epoch(self, model, data_loader):
        model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                batch_data['device'] = self.device
                output = model.validation_step(batch_data, batch_idx)
                outputs.append(output)

        return model.validation_epoch_end(outputs)

    def train(self):
        train_dataloader = self.data_loaders[self.dataset]['train']
        valid_dataloader = self.data_loaders[self.dataset]['valid']
        test_dataloader = self.data_loaders[self.dataset]['test']
        optimizer = self.model.get_optimizer(lr=self.lr)
        log_path = os.path.join(self.output_path, 'log.txt')
        with open(log_path, 'w') as f:
            pass

        best_score = None
        best_mrr = 0
        for i in range(1, self.epoch + 1):
            begin_time = time()
            train_loss = self._train_one_epoch(
                model=self.model, dataloader=train_dataloader, optimizer=optimizer, scheduler=None, epoch_idx=i
            )
            log = f'[train] epoch: {i}, loss: {train_loss}' + '\n'

            dev_score = self._validate_one_epoch(model=self.model, data_loader=valid_dataloader)
            test_score = self._validate_one_epoch(model=self.model, data_loader=test_dataloader)
            log += f'[valid] epoch: {i}, ' + score2str(dev_score) + '\n'
            log += f'[test]  epoch: {i}, ' + score2str(test_score) + '\n'
            log += '=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30 + '\n'
            print(log)
            with open(log_path, 'a') as f:
                f.write(log + '\n')

            test_mrr = test_score['MRR']
            if best_mrr < test_mrr:
                best_mrr = test_mrr
                best_score = test_score
                best_score['epoch'] = i
                self.model.save_pretrained(os.path.join(self.output_path, 'model'))

        log = f'[best]  epoch: {best_score["epoch"]}, ' + score2str(best_score)
        print(log)
        with open(log_path, 'a') as f:
            f.write(log + '\n')

    def validate(self):
        shutil.rmtree(self.output_path)

        dev_scores = self._validate_one_epoch(self.model, self.valid_dataloader)
        test_scores = self._validate_one_epoch(self.model, self.test_dataloader)
        print(f'[valid] {score2str(dev_scores)}')
        print(f'[test]  {score2str(test_scores)}')

    def main(self):
        if self.task == 'train':
            self.train()
        elif self.task == 'validate':
            self.validate()
        else:
            print(f'Unknown Task: {self.task}')


if __name__ == '__main__':
    config = get_args()
    trainer = Trainer(config)
    trainer.main()

