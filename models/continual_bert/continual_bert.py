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

        # config need to be saved
        self.config = {
            'bert_path': bert_path,
            'vocab_sizes': vocab_sizes,
            'adapter_config': adapter_config,
        }

        self.bert_config = BertConfig.from_pretrained(bert_path)
        self.num_kgs = len(vocab_sizes)
        self.hidden_size = self.bert_config.hidden_size
        self.kg2id = None
        self.kg2entities = None
        self.entity_alignment = None
        self.forward_weight = None
        self.backward_weight = None

        # word embeddings
        self.word_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_sizes[i], self.hidden_size) for i in range(self.num_kgs)])
        for i in range(self.num_kgs):
            nn.init.xavier_normal_(self.word_embeddings[i].weight, gain=nn.init.calculate_gain('relu'))

        # BERT
        self.bert_config.adapter_config = adapter_config
        self.bert = MyBertModel.from_pretrained(bert_path, config=self.bert_config)

        # decoders
        self.decoders = nn.ModuleList(
            [nn.Linear(self.hidden_size, vocab_sizes[i], bias=False) for i in range(self.num_kgs)])
        for i in range(self.num_kgs):
            self.decoders[i].weight = self.word_embeddings[i].weight
            self.decoders[i].bias = nn.Parameter(torch.zeros(vocab_sizes[i]))

        # loss func
        self.ce_loss_fc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss_fc = nn.KLDivLoss(reduction='none')
        self.mse_loss_fc = nn.MSELoss()

    def load_variables(self, kg2id, kg2entities, entity_alignment, forward_weight, backward_weight=0.):
        self.kg2id = kg2id
        self.kg2entities = kg2entities
        self.entity_alignment = entity_alignment
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight

    def training_step(self, batch, batch_idx, kg1, kg0):
        return self.forward(batch, batch_idx, kg1, kg0)

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 2)

    # validate by BERT with KG-specific adapters
    def validation_step(self, batch, batch_idx, kg_name):
        device = self.bert.device
        labels = batch['labels'].to(device)
        filters = batch['filters'].to(device) if batch['filters'] is not None else None

        logits = self.forward_text(batch['prompts'], kg_name)
        rank, fake = self._get_ranks(F.softmax(logits, dim=-1), labels, filters)
        loss = self.ce_loss_fc(logits, labels)
        return loss.item(), rank

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = self._get_scores(rank, loss)
        return scores

    def forward(self, batch: dict, batch_idx: int, kg1, kg0):
        kg_id = self.kg2id[kg1]
        device = self.bert.device

        triples = batch['triples']
        texts = batch['texts']
        labels = batch['labels'].to(device)
        filters = batch['filters'].to(device) if batch['filters'] is not None else None

        if kg0 is not None and self.backward_weight > 0:
            backward_loss = self.backward_transfer(kg0, kg1, batch['prompts'], [h for h, _, _ in triples], labels)
            return backward_loss

        logits = self.forward_text(batch['prompts'], kg1)
        loss = self.ce_loss_fc(logits, labels)

        # forward knowledge transfer
        forward_loss = 0.
        if kg0 is not None and self.forward_weight > 0:
            prompts = batch['prompts']
            new_ids = prompts['new_ids'].to(device)
            token_embeds = self.word_embeddings[kg_id](new_ids)
            forward_loss = self.feature_distill(kg1, kg0, batch, token_embeds)

        if batch_idx == 0 and self.forward_weight > 0.:
            print(f'Loss = {loss} + {self.forward_weight} * {forward_loss}')
        loss = loss + self.forward_weight * forward_loss

        return loss

    # link prediction with BERT and KG-specific adapters
    def forward_text(self, batch_text, kg_name):
        device = self.bert.device
        kg_id = self.kg2id[kg_name]
        ent_num = len(self.kg2entities[kg_name])

        input_ids = batch_text['input_ids'].to(device)
        token_type_ids = batch_text['token_type_ids'].to(device)
        attention_mask = batch_text['attention_mask'].to(device)
        mask_pos = batch_text['mask_pos'].to(device)
        new_pos = batch_text['new_pos'].to(device)
        new_ids = batch_text['new_ids'].to(device)

        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)  # (batch_size, seq_len, hidden_size)
        token_embeds = self.word_embeddings[kg_id](new_ids)  # ( 5 * batch_size, hidden_size)
        inputs_embeds[new_pos[:, 0], new_pos[:, 1]] = token_embeds

        output = self.bert(
            inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
            adapter_id=kg_id,
        )
        outputs_embeds = output.last_hidden_state
        mask_embeds = outputs_embeds[mask_pos[:, 0], mask_pos[:, 1]]  # (batch_size, hidden_size)
        logits = self.decoders[kg_id](mask_embeds)[:, 0: ent_num]  # (batch_size, ent_num)
        return logits

    def feature_distill(self, kg1, kg0, batch_data, token_embeds):
        kg1_id, kg0_id = self.kg2id[kg1], self.kg2id[kg0]
        kg1_to_kg0 = self.entity_alignment[kg1][kg0]
        kg1_ents, kg0_ents = self.kg2entities[kg1], self.kg2entities[kg0]
        device = self.bert.device

        # token ids for entities and soft prompts
        new_ids = batch_data['prompts']['new_ids'].to(device)  
        head_pos = torch.nonzero(new_ids < len(kg1_ents)).squeeze(dim=-1) 
        head_pos = head_pos.cpu().numpy().tolist()

        heads = [h for h, _, _ in batch_data['triples']]
        aligned_head_pos = [head_pos[i] for i, h in enumerate(heads) if h in kg1_to_kg0]
        if len(aligned_head_pos) == 0: 
            return 0.

        current_head_embeds = token_embeds[aligned_head_pos, :]
        former_head_new_ids = [kg0_ents[kg1_to_kg0[h]['ent']]['token_id'] for i, h in enumerate(heads) if h in kg1_to_kg0]
        former_head_embeds = self.word_embeddings[kg0_id](torch.tensor(former_head_new_ids).to(device))

        weights = torch.tensor([kg1_to_kg0[h]['cos']for h in heads if h in kg1_to_kg0]).to(device)
        mse_loss_fc = nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fc(current_head_embeds, former_head_embeds)
        weighted_mse_loss = torch.mean(mse_loss * weights.unsqueeze(dim=1))
        return weighted_mse_loss

    def backward_transfer(self, kg0, kg1, batch_text, head_entities, labels):
        device = self.bert.device
        kg0_id, kg1_id = self.kg2id[kg0], self.kg2id[kg1]
        kg0_entities, kg1_entities = self.kg2entities[kg0], self.kg2entities[kg1]
        ent_num = len(self.kg2entities[kg1])
        kg1_to_kg0 = self.entity_alignment[kg1][kg0]

        input_ids = batch_text['input_ids'].to(device)
        token_type_ids = batch_text['token_type_ids'].to(device)
        attention_mask = batch_text['attention_mask'].to(device)
        mask_pos = batch_text['mask_pos'].to(device)  # (batch_size, 2), positions for [MASK]
        ent_pos = batch_text['ent_pos'].to(device)  # positions for entities
        ent_ids = batch_text['ent_ids'].to(device)

        # 1. replace word embeddings of entities, and treat the prompt sentences as training for kg0
        # 1.1 find aligned entities and their batch idxs
        ents1, ents0, batch_idxs = [], [], []
        for i, h in enumerate(head_entities):
            if h in kg1_to_kg0:
                ents1.append(h)
                ents0.append(kg1_to_kg0[h]['ent'])
                batch_idxs.append(i)
        if len(ents0) == 0:
            return 0.
        ids0 = torch.tensor([kg0_entities[h]['token_id'] for h in ents0]).to(device)
        embeds0 = self.word_embeddings[kg0_id](ids0)
        # 1.2 replace word embeddings
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)  # (batch_size, seq_len, hidden_size)
        inputs_embeds[ent_pos[batch_idxs, 0], ent_pos[batch_idxs, 1]] = embeds0

        # 2. encode as normal, but use adapters for kg0
        output = self.bert(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,
                           adapter_id=kg0_id)
        output_embeds = output.last_hidden_state
        mask_embeds = output_embeds[mask_pos[:, 0], mask_pos[:, 1]]  # (batch_size, hidden_size)
        logits = self.decoders[kg1_id](mask_embeds)[:, 0: ent_num]  # (batch_size, entity_num)
        logits = logits[batch_idxs]  # (aligned_batch_size, entity_num)
        ce_loss = self.ce_loss_fc(logits, labels[batch_idxs])

        # 3. encode as normal, use adapters for kg1
        teacher_logits = self.forward_text(batch_text, kg1)
        teacher_logits = teacher_logits[batch_idxs, :]
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        kd_loss = kl_loss(torch.log_softmax(logits, dim=-1), torch.softmax(teacher_logits.detach(), dim=-1))
        return ce_loss + self.backward_weight * kd_loss

    def _get_ranks(self, probs, labels, filters=None):
        # probs is output by softmax
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

    def get_optimizer(self, total_steps: int, kg0, kg1, embedding_lr, adapter_lr):
        def get_adapter_params(kg_name, lr):
            kg_id = self.kg2id[kg_name]
            params, names = [], []
            for n, p in self.named_parameters():
                if f'adapters.{kg_id}' in n:
                    names.append(n)
                    p.requires_grad = True
                    params.append(p)
            return [{'params': params, 'weight_decay': 0., 'lr': lr}], names

        def get_embedding_params(kg_name, lr):
            kg_id = self.kg2id[kg_name]
            params, names = [], []
            for n, p in self.named_parameters():
                if f'word_embeddings.{kg_id}' in n or f'decoders.{kg_id}' in n:
                    names.append(n)
                    p.requires_grad = True
                    params.append(p)
            return [{'params': params, 'weight_decay': 0., 'lr': lr}], names

        for n, p in self.named_parameters():
            p.requires_grad = False

        if kg0 is not None and self.backward_weight > 0:
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

        # load saved models, keep useful parameters
        state_dict = torch.load(os.path.join(model_path, 'model.bin'), map_location='cpu')
        useful_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        # update the state_dict for the new model
        model_dict.update(useful_dict)
        model.load_state_dict(model_dict)
        return model

    def grad_norm(self):
        norms = get_norms(self.parameters()).item()
        return round(norms, 4)

    # deprecated
    def adapter_distill(
            self, teacher_id, student_id, inputs_embeds, attention_mask, token_type_ids, logits,
            mask_pos, labels, filters, ent_num
    ):
        teacher_embeds = self.bert(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids,
            adapter_id=teacher_id,
        ).last_hidden_state[mask_pos[:, 0], mask_pos[:, 1]]
        former_logits = self.decoders[student_id](teacher_embeds)[:, 0: ent_num]  # (batch_size, ent_num)

        ranks = self._get_ranks(F.softmax(former_logits, dim=-1), labels, filters)
        idxs = [i for i, rank in enumerate(ranks) if rank <= 10]

        loss = self.kl_loss_fc(F.log_softmax(logits, dim=-1), F.softmax(former_logits, dim=-1))
        loss = torch.mean(torch.sum(loss, dim=-1))

        return loss