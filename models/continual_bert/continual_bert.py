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


class SIKE(nn.Module):
    def __init__(self, args: dict):
        super(SIKE, self).__init__()

        self.args = args
        self.bert_config = BertConfig.from_pretrained(args['bert_path'])
        self.hidden_size = self.bert_config.hidden_size
        self.kg2ents = None
        self.entity_alignment = None

        # PLM with KG-specific adapters
        self.bert_config.adapter_config = {'adapter_names': args['datasets'], 'adapter_hidden_size': args['adapter_hidden_size']}
        self.bert = MyBertModel.from_pretrained(args['bert_path'], config=self.bert_config)

        self.word_embeddings = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for kg in self.args['datasets']:
            num_add_tokens = args['num_added_tokens'][kg]

            self.word_embeddings[kg] = nn.Embedding(num_add_tokens, self.hidden_size)
            nn.init.xavier_normal_(self.word_embeddings[kg].weight, gain=nn.init.calculate_gain('relu'))

            self.decoders[kg] = nn.Linear(self.hidden_size, num_add_tokens, bias=False)
            self.decoders[kg].weight = self.word_embeddings[kg].weight
            self.decoders[kg].bias = nn.Parameter(torch.zeros(num_add_tokens))

        # loss functions
        self.ce_loss_fc = nn.CrossEntropyLoss(label_smoothing=args['label_smoothing'])
        self.kl_loss_fc = nn.KLDivLoss(reduction='none')
        self.mse_loss_fc = nn.MSELoss()

    def load_variables(self, kg2ents, entity_alignment):
        self.kg2ents = kg2ents
        self.entity_alignment = entity_alignment

    def training_step(self, batch, batch_idx, kg1, kg0):
        return self.forward(batch, batch_idx, kg1, kg0)

    def validation_step(self, batch, batch_idx, kg_name, save_result=False):
        labels = batch['labels'].cuda()  
        filters = batch['filters'].cuda() if batch['filters'] is not None else None

        logits = self.forward_text(batch['prompts'], kg_name)
        rank, fake = self._get_ranks(F.softmax(logits, dim=-1), labels, filters)
        if save_result:
            with open('result/output.txt', 'a', encoding='utf-8') as f:
                for i, (h, r, t) in enumerate(batch['triples']):
                    f.write(f'{h}\t{r}\t{t}\t{rank[i]}\n')
        loss = self.ce_loss_fc(logits, labels)
        return loss.item(), rank

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank

        rank = np.array(rank)
        score = {
            'loss': np.mean(loss),
            'hits@1': np.mean(rank <= 1),
            'hits@3': np.mean(rank <= 3),
            'hits@10': np.mean(rank <= 10),
            'mrr': np.mean(1. / rank),
        }
        return {k: np.round(v, 3) for k, v in score.items()}

    def forward(self, batch: dict, batch_idx: int, kg1, kg0):
        triples = batch['triples']
        texts = batch['texts']
        labels = batch['labels'].cuda() 
        filters = batch['filters'].cuda() if batch['filters'] is not None else None

        if kg0 is not None and self.args['back_transfer']:
            backward_loss = self.backward_transfer(kg0, kg1, batch['prompts'], [h for h, _, _ in triples], labels)
            return backward_loss

        logits = self.forward_text(batch['prompts'], kg1)
        loss = self.ce_loss_fc(logits, labels)

        if self.args['alpha'] < 0. or kg0 is None:
            return loss
        
        alpha = self.args['alpha']
        forward_loss = 0.

        prompts = batch['prompts']
        new_ids = prompts['new_ids'].cuda()  
        token_embeds = self.word_embeddings[kg1](new_ids)
        forward_loss = self.feature_distill(kg1, kg0, batch, token_embeds)

        loss = loss + alpha * forward_loss
        return loss

    def forward_text(self, batch_text, kg_name):
        ent_num = len(self.kg2ents[kg_name])

        input_ids = batch_text['input_ids'].cuda()
        token_type_ids = batch_text['token_type_ids'].cuda()
        attention_mask = batch_text['attention_mask'].cuda()
        mask_pos = batch_text['mask_pos'].cuda() 

        new_pos = batch_text['new_pos'].cuda()
        new_ids = batch_text['new_ids'].cuda()

        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        token_embeds = self.word_embeddings[kg_name](new_ids)
        inputs_embeds[new_pos[:, 0], new_pos[:, 1]] = token_embeds

        output = self.bert(
            inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
            adapter_id=kg_name,
        )
        outputs_embeds = output.last_hidden_state
        mask_embeds = outputs_embeds[mask_pos[:, 0], mask_pos[:, 1]]
        logits = self.decoders[kg_name](mask_embeds)[:, 0: ent_num]

    def feature_distill(self, kg1, kg0, batch_data, token_embeds):
        kg1_to_kg0 = self.entity_alignment[kg1][kg0]
        kg1_ents, kg0_ents = self.kg2ents[kg1], self.kg2ents[kg0]

        new_ids = batch_data['prompts']['new_ids'].cuda()
        head_pos = torch.nonzero(new_ids < len(kg1_ents)).squeeze(dim=-1)
        head_pos = head_pos.cpu().numpy().tolist()
        
        heads = [h for h, _, _ in batch_data['triples']]
        aligned_head_pos = [head_pos[i] for i, h in enumerate(heads) if h in kg1_to_kg0]
        if len(aligned_head_pos) == 0:
            return 0.

        current_head_embeds = token_embeds[aligned_head_pos, :]
        former_head_new_ids = [kg0_ents[kg1_to_kg0[h]['ent']]['idx'] for i, h in enumerate(heads) if h in kg1_to_kg0]
        former_head_embeds = self.word_embeddings[kg0](torch.tensor(former_head_new_ids).cuda())

        weights = torch.tensor([kg1_to_kg0[h]['cos']for h in heads if h in kg1_to_kg0]).cuda()
        mse_loss_fc = nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fc(current_head_embeds, former_head_embeds)
        weighted_mse_loss = torch.mean(mse_loss * weights.unsqueeze(dim=1))
        return weighted_mse_loss

    def backward_transfer(self, kg0, kg1, batch_text, head_entities, labels):
        kg0_ents, kg1_ents = self.kg2ents[kg0], self.kg2ents[kg1]
        ent_num = len(self.kg2ents[kg1])
        kg1_to_kg0 = self.entity_alignment[kg1][kg0]

        input_ids = batch_text['input_ids'].cuda()
        token_type_ids = batch_text['token_type_ids'].cuda()
        attention_mask = batch_text['attention_mask'].cuda()
        mask_pos = batch_text['mask_pos'].cuda()
        ent_pos = batch_text['ent_pos'].cuda()
        ent_ids = batch_text['ent_ids'].cuda()

        ents1, ents0, batch_idxs = [], [], []
        for i, h in enumerate(head_entities):
            if h in kg1_to_kg0:
                ents1.append(h)
                ents0.append(kg1_to_kg0[h]['ent'])
                batch_idxs.append(i)
        if len(ents0) == 0:
            return 0.
        ids0 = torch.tensor([kg0_ents[h]['idx'] for h in ents0]).cuda()
        embeds0 = self.word_embeddings[kg0](ids0)
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        inputs_embeds[ent_pos[batch_idxs, 0], ent_pos[batch_idxs, 1]] = embeds0

        output = self.bert(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
                           adapter_id=kg0)
        output_embeds = output.last_hidden_state
        mask_embeds = output_embeds[mask_pos[:, 0], mask_pos[:, 1]]
        logits = self.decoders[kg1](mask_embeds)[:, 0: ent_num]
        logits = logits[batch_idxs]
        ce_loss = self.ce_loss_fc(logits, labels[batch_idxs])
        return ce_loss

    def _get_ranks(self, probs, labels, filters=None):
        log_probs = torch.log(probs)
        if filters is not None:
            log_probs[filters[:, 0], filters[:, 1]] = -torch.inf

        sorted_idx = torch.argsort(log_probs, dim=-1, descending=True)
        labels = labels.unsqueeze(dim=1)
        rank = torch.nonzero(torch.eq(sorted_idx, labels))[:, 1] + 1
        return rank.cpu().numpy().tolist(), sorted_idx[:, 0]

    def get_optimizer(self, total_steps: int, kg0, kg1, embedding_lr, adapter_lr):
        def get_adapter_params(kg_name, lr):
            params, names = [], []
            for n, p in self.named_parameters():
                if f'adapters.{kg_name}' in n:
                    names.append(n)
                    p.requires_grad = True
                    params.append(p)
            return [{'params': params, 'weight_decay': 0., 'lr': lr}], names

        def get_embedding_params(kg_name, lr):
            params, names = [], []
            for n, p in self.named_parameters():
                if f'word_embeddings.{kg_name}' in n or f'decoders.{kg_name}' in n:
                    names.append(n)
                    p.requires_grad = True
                    params.append(p)
            return [{'params': params, 'weight_decay': 0., 'lr': lr}], names

        for n, p in self.named_parameters():
            p.requires_grad = False

        if kg0 is not None and self.args['back_transfer']:
            adapter_params, adapter_names = get_adapter_params(kg0, adapter_lr)
            embedding_params, embedding_names = get_embedding_params(kg0, embedding_lr)
        else:
            adapter_params, adapter_names = get_adapter_params(kg1, adapter_lr)
            embedding_params, embedding_names = get_embedding_params(kg1, embedding_lr)

        params = adapter_params + embedding_params
        names = adapter_names + embedding_names
        print(f'Parameters to be optimized: {names[0]}, ...')

        optimizer = AdamW(params, eps=1e-6)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps,
        )
        return optimizer, scheduler

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        model_dict = self.state_dict()

        state_dict = torch.load(model_path, map_location='cpu')
        useful_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        model_dict.update(useful_dict)
        self.load_state_dict(model_dict)

    def grad_norm(self):
        norms = get_norms(self.parameters()).item()
        return round(norms, 4)
