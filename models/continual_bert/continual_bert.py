import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from transformers import get_cosine_schedule_with_warmup

from .my_transformers import MyBertModel, MyBertForMaskedLM
from ..utils import get_norms


class ContinualBert(nn.Module):
    def __init__(self, bert_path: str, vocab_sizes: list, adapter_config: dict, label_smoothing=0.8):
        super(ContinualBert, self).__init__()

        self.config = {
            'bert_path': bert_path,
            'vocab_sizes': vocab_sizes,
            'adapter_config': adapter_config,
        }

        self.bert_config = BertConfig.from_pretrained(bert_path)
        self.num_kgs = len(vocab_sizes)
        self.hidden_size = self.bert_config.hidden_size

        self.word_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_sizes[i], self.hidden_size) for i in range(self.num_kgs)])
        for i in range(self.num_kgs):
            nn.init.xavier_normal_(self.word_embeddings[i].weight, gain=nn.init.calculate_gain('relu'))

        self.bert_config.adapter_config = adapter_config
        self.bert = MyBertModel.from_pretrained(bert_path, config=self.bert_config)

        self.decoders = nn.ModuleList(
            [nn.Linear(self.hidden_size, vocab_sizes[i], bias=False) for i in range(self.num_kgs)])
        for i in range(self.num_kgs):
            self.decoders[i].weight = self.word_embeddings[i].weight
            self.decoders[i].bias = nn.Parameter(torch.zeros(vocab_sizes[i]))

        self.ce_loss_fc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss_fc = nn.KLDivLoss(reduction='none')
        self.mse_loss_fc = nn.MSELoss()

    def forward(self, batch, batch_idx):
        output = self.link_prediction(batch, batch_idx)
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch, batch_idx)
        return output['loss']

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 2)


    def validation_step(self, batch, batch_idx):
        output = self.link_prediction(batch, batch_idx)
        return output['loss'].item(), output['rank']

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = self._get_scores(rank, loss)
        return scores

    def link_prediction(self, batch: dict, batch_idx: int):
        module_id = batch.pop('module_id', -1)

        mse_weight = batch.pop('alpha', -1.0)
        align_info = batch.pop('align_info', None)
        support_module_id = batch.pop('support_module_id', -1)

        distill_weight = batch.pop('beta', -1.0)
        struct_model = batch.pop('struct_model', None)

        device = batch.pop('device', 'cpu')

        triples = batch['triples']
        texts = batch['texts']
        prompts = batch['prompts']
        input_ids = prompts['input_ids'].to(device)
        token_type_ids = prompts['token_type_ids'].to(device)
        attention_mask = prompts['attention_mask'].to(device)
        mask_pos = prompts['mask_pos'].to(device)  
        word_pos = prompts['word_pos'].to(device) 
        token_ids = prompts['token_ids'].to(device)  
        labels = batch['labels'].to(device)  
        filters = batch['filters'].to(device) if batch['filters'] is not None else None 
        ent_num = batch['ent_num']  

        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)  
        token_embeds = self.word_embeddings[module_id](token_ids)
        inputs_embeds[word_pos[:, 0], word_pos[:, 1]] = token_embeds

        output = self.bert(
            inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
            adapter_id=module_id,
        )
        outputs_embeds = output.last_hidden_state
        mask_embeds = outputs_embeds[mask_pos[:, 0], mask_pos[:, 1]]  
        logits = self.decoders[module_id](mask_embeds)[:, 0: ent_num] 

        rank, fake = self._get_ranks(F.softmax(logits, dim=-1), labels, filters)
        loss = self.ce_loss_fc(logits, labels)

        if mse_weight > 0:
            mse_loss = self.feature_distill(support_module_id, align_info, batch, token_embeds, device)
            if mse_loss is None:
                mse_loss = 0.
        else:
            mse_loss = 0.

        if struct_model is not None and distill_weight > 0.:
            batch['device'] = device
            struct_output = struct_model.link_prediction(batch, batch_idx)
            struct_logits = struct_output['logits']
            kl_loss = nn.KLDivLoss(reduction='batchmean')
            kd_loss = kl_loss(torch.log_softmax(logits, dim=-1), torch.softmax(struct_logits.detach(), dim=-1))
        else:
            kd_loss = 0.

        if batch_idx == 0 and (mse_weight > 0. or distill_weight > 0.):
            print(f'Loss = {loss} + {mse_weight} * {mse_loss} + {distill_weight} * {kd_loss}')
        loss = loss + mse_weight * mse_loss + distill_weight * kd_loss

        with open('res.txt', 'a', encoding='utf-8') as f:
            for i, r_ in enumerate(rank):
                false_ent = fake[i].item()
                ents = align_info['current_ents']
                for e in ents:
                    if ents[e]['token_id'] == false_ent:
                        false_ent = ents[e]['raw_name']
                        break
                h, r, t = triples[i]
                h_name, r_name, t_name, label = texts[i]
                if label == 0:
                    lst = [t, r, h, t_name, r_name, h_name, str(r_), false_ent, 'head']
                    f.write('\t'.join(lst) + '\n')
                elif label == 1:
                    lst = [h, r, t, h_name, r_name, t_name, str(r_), false_ent, 'tail']
                    f.write('\t'.join(lst) + '\n')
                else:
                    assert 0, 'no'

        return {'loss': loss, 'rank': rank, 'logits': logits}

    def feature_distill(self, support_module_id, align_info, batch_data, token_embeds, device):
        if align_info is None:
            return None
        align_dit = align_info['align_dict']
        current_ents = align_info['current_ents']
        former_ents = align_info['support_ents']

        token_ids = batch_data['prompts']['token_ids'].to(device)  
        head_pos = torch.nonzero(token_ids < len(current_ents)).squeeze(dim=-1)
        head_pos = head_pos.cpu().numpy().tolist()
        heads = [h for h, _, _ in batch_data['triples']]
        aligned_head_pos = [head_pos[i] for i, h in enumerate(heads) if h in align_dit]
        if len(aligned_head_pos) == 0:
            return None

        current_head_embeds = token_embeds[aligned_head_pos, :]
        former_head_token_ids = [former_ents[align_dit[h]]['token_id'] for i, h in enumerate(heads) if h in align_dit]
        former_head_embeds = self.word_embeddings[support_module_id](torch.tensor(former_head_token_ids).to(device))

        mse_loss = self.mse_loss_fc(current_head_embeds, former_head_embeds)
        return mse_loss

    def _get_ranks(self, probs, labels, filters=None):
        log_probs = torch.log(probs)
        if filters is not None:
            log_probs[filters[:, 0], filters[:, 1]] = -torch.inf

        sorted_idx = torch.argsort(log_probs, dim=-1, descending=True)
        labels = labels.unsqueeze(dim=1)
        rank = torch.nonzero(torch.eq(sorted_idx, labels))[:, 1] + 1
        return rank.cpu().numpy().tolist(), sorted_idx[:, 0]

    def _get_scores(self, rank: list, loss=None):
        rank = np.array(rank)
        hits1 = round(np.mean(rank <= 1) * 100, 2)
        hits3 = round(np.mean(rank <= 3) * 100, 2)
        hits10 = round(np.mean(rank <= 10) * 100, 2)
        mrr = round(float(np.mean(1. / rank)), 4)
        loss = round(loss, 2)
        return {'loss': loss, 'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10, 'MRR': mrr}

    def get_optimizer(self, total_steps: int, module_id: str, embedding_lr, adapter_lr):
        def get_adapter_params():
            params = []
            names = []
            for n, p in self.named_parameters():
                if f'adapters.{module_id}' in n:
                    names.append(n)
                    p.requires_grad = True
                    params.append(p)
            return [{'params': params, 'weight_decay': 0., 'lr': adapter_lr}], names

        def get_embedding_params():
            params = []
            names = []
            for n, p in self.named_parameters():
                if f'word_embeddings.{module_id}' in n or f'decoders.{module_id}' in n:
                    names.append(n)
                    p.requires_grad = True
                    params.append(p)
            return [{'params': params, 'weight_decay': 0., 'lr': embedding_lr}], names

        for n, p in self.named_parameters():
            p.requires_grad = False

        adapter_params, adapter_names = get_adapter_params()
        embedding_params, embedding_names = get_embedding_params()

        params = adapter_params + embedding_params
        names = adapter_names + embedding_names
        print(f'要优化的参数: {names}')

        optimizer = AdamW(params, eps=1e-6)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps,
        )
        return optimizer, scheduler

    def save_pretrained(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        json.dump(self.config, open(os.path.join(model_path, 'config.json'), 'w', encoding='utf-8'))
        torch.save(self.state_dict(), os.path.join(model_path, 'model.bin'))

    @classmethod
    def from_pretrained(cls, model_path):
        config = json.load(open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8'))
        model = cls(**config)
        model_dict = model.state_dict()

        state_dict = torch.load(os.path.join(model_path, 'model.bin'), map_location='cpu')
        useful_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        model_dict.update(useful_dict)
        model.load_state_dict(model_dict)
        return model

    def grad_norm(self):
        norms = get_norms(self.parameters()).item()
        return round(norms, 4)

    def link_prediction_with_bert(self, module_id, batch, batch_idx, device):
        prompts = batch['prompts']
        input_ids = prompts['input_ids'].to(device)
        token_type_ids = prompts['token_type_ids'].to(device)
        attention_mask = prompts['attention_mask'].to(device)
        mask_pos = prompts['mask_pos'].to(device)
        word_pos = prompts['word_pos'].to(device)
        token_ids = prompts['token_ids'].to(device)
        labels = batch['labels'].to(device)  
        filters = batch['filters'].to(device) if batch['filters'] is not None else None 
        ent_num = batch['ent_num']


        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        token_embeds = self.word_embeddings[module_id](token_ids)
        inputs_embeds[word_pos[:, 0], word_pos[:, 1]] = token_embeds

        output = self.bert(
            inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
            adapter_id=-1,
        )
        outputs_embeds = output.last_hidden_state
        mask_embeds = outputs_embeds[mask_pos[:, 0], mask_pos[:, 1]] 
        logits = self.decoders[module_id](mask_embeds)[:, 0: ent_num] 

        rank = self._get_ranks(F.softmax(logits, dim=-1), labels, filters)
        loss = self.ce_loss_fc(logits, labels)

        return {'loss': loss, 'rank': rank}

    def adapter_distill(
            self, teacher_id, student_id, inputs_embeds, attention_mask, token_type_ids, logits,
            mask_pos, labels, filters, ent_num
    ):
        teacher_embeds = self.bert(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids,
            adapter_id=teacher_id,
        ).last_hidden_state[mask_pos[:, 0], mask_pos[:, 1]]
        former_logits = self.decoders[student_id](teacher_embeds)[:, 0: ent_num]

        ranks = self._get_ranks(F.softmax(former_logits, dim=-1), labels, filters)
        idxs = [i for i, rank in enumerate(ranks) if rank <= 10]

        loss = self.kl_loss_fc(F.log_softmax(logits, dim=-1), F.softmax(former_logits, dim=-1))
        loss = torch.mean(torch.sum(loss, dim=-1))

        return loss