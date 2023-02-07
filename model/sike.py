import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import get_cosine_schedule_with_warmup

from .adapter_bert.my_transformers import MyBertModel
from .utils import get_norms


class SIKE(nn.Module):
    def __init__(self, vocab_sizes: list, bert_path: str, adapter_hidden_size: int, label_smoothing=0.8):
        super(SIKE, self).__init__()

        self.config = {
            'vocab_sizes': vocab_sizes,
            'bert_path': bert_path,
            'adapter_hidden_size': adapter_hidden_size,
            'label_smoothing': label_smoothing,
        }

        bert_config = BertConfig.from_pretrained(bert_path)
        bert_config.adapter_config = {'num_adapters': len(vocab_sizes), 'adapter_hidden_size': adapter_hidden_size}

        self.num_kgs = len(vocab_sizes)
        self.hidden_size = bert_config.hidden_size
        self.kg2id = None
        self.kg2entities = None
        self.entity_alignment = None
        self.forward_weight = None
        self.backward_weight = None

        # 1. embeddings
        self.word_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_sizes[i], self.hidden_size) for i in range(self.num_kgs)])
        for i in range(self.num_kgs):
            nn.init.xavier_normal_(self.word_embeddings[i].weight)

        # 2. text encoder
        self.text_encoder = MyBertModel.from_pretrained(bert_path, config=bert_config)

        # 4. decoder
        self.decoders = nn.ModuleList(
            [nn.Linear(self.hidden_size, vocab_sizes[i], bias=False) for i in range(self.num_kgs)])
        for i in range(self.num_kgs):
            self.decoders[i].weight = self.word_embeddings[i].weight
            self.decoders[i].bias = nn.Parameter(torch.zeros(vocab_sizes[i]))

        # 5. loss function
        self.ce_loss_fc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def load_variables(self, kg2id, kg2entities, entity_alignment, forward_weight, backward_weight=0.):
        self.kg2id = kg2id
        self.kg2entities = kg2entities
        self.entity_alignment = entity_alignment
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight

    def training_step(self, batch, batch_idx, kg1, kg0):
        output = self.link_prediction(batch, batch_idx, kg1, kg0)
        return output['loss']

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 2)

    def validation_step(self, batch, batch_idx, kg, output_path=None):
        device = self.text_encoder.device
        labels = batch['label'].to(device)
        filters = batch['filters'].to(device) if batch['filters'] is not None else None
        # 1. encode text
        text_logits = self.forward_text(batch['text'], kg)  # (batch_size, ent_num)
        text_rank, wrong_ans = self._get_ranks(F.softmax(text_logits, dim=-1), labels, filters)
        text_loss = self.ce_loss_fc(text_logits, labels)

        if output_path is not None:
            triples = batch['triple']
            texts = batch['raw_text']
            with open(output_path, 'a', encoding='utf-8') as f:
                for i in range(len(triples)):
                    h, r, t = triples[i]
                    _, _, _, direction = texts[i]
                    f.write(f'{h}\t{r}\t{t}\t{wrong_ans[i]}\t{direction}\n')

        return text_loss.item(), text_rank

    def validation_epoch_end(self, outputs):
        text_loss, text_rank = list(), list()
        for batch_text_loss, batch_text_rank in outputs:
            text_loss.append(batch_text_loss)
            text_rank += batch_text_rank
        text_scores = self._get_scores(text_rank, np.mean(text_loss))
        return text_scores

    def link_prediction(self, batch: dict, batch_idx: int, kg1, kg0):
        device = self.text_encoder.device
        labels = batch['label'].to(device)
        filters = batch['filters'].to(device) if batch['filters'] is not None else None

        backward_loss = 0.
        if kg0 is not None and self.backward_weight > 0.:
            backward_loss = self.backward_transfer(batch['text'], kg0, kg1, batch['triple'], labels)
            return {'loss': self.backward_weight * backward_loss}
        # 1. encode text
        text_logits = self.forward_text(batch['text'], kg1)  # (batch_size, ent_num)
        # text_rank = self._get_ranks(F.softmax(text_logits, dim=-1), labels, filters)
        text_loss = self.ce_loss_fc(text_logits, labels)

        forward_loss = 0.
        if kg0 is not None and self.forward_weight > 0.:
            forward_loss = self.forward_transfer(kg0, kg1, triples=batch['triple'])

        if batch_idx == 0:
            loss1 = forward_loss.item() if type(forward_loss) is not float else forward_loss
            print(f'Loss={round(text_loss.item(), 2)} + {self.forward_weight} * {loss1}')
        loss = text_loss + self.forward_weight * forward_loss + self.backward_weight * backward_loss
        return {'loss': loss}

    def forward_text(self, batch_text, kg_name):
        kg1_id = self.kg2id[kg_name]
        device = self.text_encoder.device
        ent_num = len(self.kg2entities[kg_name])

        input_ids = batch_text['input_ids'].to(device)
        token_type_ids = batch_text['token_type_ids'].to(device)
        attention_mask = batch_text['attention_mask'].to(device)
        mask_pos = batch_text['mask_pos'].to(device)
        new_pos = batch_text['new_pos'].to(device)
        new_ids = batch_text['new_ids'].to(device)

        inputs_embeds = self.text_encoder.embeddings.word_embeddings(input_ids)
        token_embeds = self.word_embeddings[kg1_id](new_ids)
        inputs_embeds[new_pos[:, 0], new_pos[:, 1]] = token_embeds
        output = self.text_encoder(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask, adapter_id=kg1_id)
        output_embeds = output.last_hidden_state
        mask_embeds = output_embeds[mask_pos[:, 0], mask_pos[:, 1]]
        logits = self.decoders[kg1_id](mask_embeds)[:, 0: ent_num]
        return logits

    def forward_transfer(self, kg0, kg1, triples):
        device = self.text_encoder.device
        kg0_id, kg1_id = self.kg2id[kg0], self.kg2id[kg1]
        kg0_entities, kg1_entities = self.kg2entities[kg0], self.kg2entities[kg1]
        kg1_to_kg0 = self.entity_alignment[kg1][kg0]

        heads1 = [h for h, _, _ in triples if h in kg1_to_kg0]
        if len(heads1) == 0:
            return 0.
        ids1 = torch.tensor([kg1_entities[h]['token_id'] for h in heads1]).to(device)
        embeds1 = self.word_embeddings[kg1_id](ids1)
        heads0 = [kg1_to_kg0[h]['ent'] for h in heads1]
        ids0 = torch.tensor([kg0_entities[h]['token_id'] for h in heads0]).to(device)
        embeds0 = self.word_embeddings[kg0_id](ids0)
        weights = torch.tensor([kg1_to_kg0[h]['cos'] for h in heads1]).to(device)
        mse_loss_fc = nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fc(embeds1, embeds0)
        weighted_mse_loss = torch.mean(mse_loss * weights.unsqueeze(dim=1))
        return weighted_mse_loss

    def backward_transfer(self, batch_text, kg0, kg1, triples, labels):
        device = self.text_encoder.device
        kg0_id, kg1_id = self.kg2id[kg0], self.kg2id[kg1]
        kg0_entities, kg1_entities = self.kg2entities[kg0], self.kg2entities[kg1]
        ent_num = len(self.kg2entities[kg1])
        kg1_to_kg0 = self.entity_alignment[kg1][kg0]

        input_ids = batch_text['input_ids'].to(device)
        token_type_ids = batch_text['token_type_ids'].to(device)
        attention_mask = batch_text['attention_mask'].to(device)
        mask_pos = batch_text['mask_pos'].to(device)
        ent_pos = batch_text['ent_pos'].to(device)
        ent_ids = batch_text['ent_ids'].to(device)

        heads1 = [h for h, _, _ in triples if h in kg1_to_kg0]
        batch_idxs = [i for i in range(len(triples)) if triples[i][0] in kg1_to_kg0]
        if len(heads1) == 0:
            return 0.
        heads0 = [kg1_to_kg0[h]['ent'] for h in heads1]
        ids0 = torch.tensor([kg0_entities[h]['token_id'] for h in heads0]).to(device)
        embeds0 = self.word_embeddings[kg0_id](ids0)
        inputs_embeds = self.text_encoder.embeddings.word_embeddings(input_ids)
        idxs1, idxs2 = [2*i for i in batch_idxs], [2*i+1 for i in batch_idxs]
        ent_pos1, ent_pos2 = ent_pos[idxs1], ent_pos[idxs2]
        inputs_embeds[ent_pos1[:, 0], ent_pos1[:, 1]] = embeds0
        inputs_embeds[ent_pos2[:, 0], ent_pos2[:, 1]] = embeds0
        output = self.text_encoder(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask, adapter_id=kg0_id)
        output_embeds = output.last_hidden_state
        mask_embeds = output_embeds[mask_pos[:, 0], mask_pos[:, 1]]
        logits = self.decoders[kg1_id](mask_embeds)[:, 0: ent_num]

        logits = logits[batch_idxs]
        ce_loss = self.ce_loss_fc(logits, labels[batch_idxs])
        t_logits = self.forward_text(batch_text, kg1)
        t_logits = t_logits[batch_idxs]
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        kd_loss = kl_loss(torch.log_softmax(logits, dim=-1), torch.softmax(t_logits.detach(), dim=-1))
        return ce_loss + 1.0 * kd_loss

    def _get_ranks(self, probs, labels, filters=None):
        log_probs = torch.log(probs)
        if filters is not None:
            log_probs[filters[:, 0], filters[:, 1]] = -torch.inf

        sorted_idx = torch.argsort(log_probs, dim=-1, descending=True)
        labels = labels.unsqueeze(dim=1)
        rank = torch.nonzero(torch.eq(sorted_idx, labels))[:, 1] + 1
        return rank.cpu().numpy().tolist(), sorted_idx[:, 0].cpu().numpy().tolist()

    def _get_scores(self, rank: list, loss=None):
        rank = np.array(rank)
        hits1 = round(np.mean(rank <= 1) * 100, 2)
        hits3 = round(np.mean(rank <= 3) * 100, 2)
        hits10 = round(np.mean(rank <= 10) * 100, 2)
        mrr = round(float(np.mean(1. / rank)), 4)
        loss = round(loss, 2)
        return {'loss': loss, 'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10, 'MRR': mrr}

    def get_optimizer(self, total_steps: int, kg1: str, kg0: str, embedding_lr, adapter_lr):
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

        if kg0 is None or self.backward_weight <= 0.:
            adapter_params, adapter_names = get_adapter_params(kg1, adapter_lr)
            embedding_params, embedding_names = get_embedding_params(kg1, embedding_lr)
            params = adapter_params + embedding_params
        else:
            kg0_adapter_params, kg0_adapter_names = get_adapter_params(kg0, adapter_lr)
            kg0_embedding_params, kg0_embedding_names = get_embedding_params(kg0, embedding_lr)
            params = kg0_adapter_params + kg0_embedding_params

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
    def from_pretrained(cls, model_path, vocab_sizes=None):
        config = json.load(open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8'))
        if vocab_sizes is not None:
            config['vocab_sizes'] = vocab_sizes

        model = cls(**config)
        model_dict = model.state_dict()
        state_dict = torch.load(os.path.join(model_path, 'model.bin'), map_location='cpu')
        # useful_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # model_dict.update(useful_dict)
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        return model

    def grad_norm(self):
        norms = get_norms(self.parameters()).item()
        return round(norms, 4)

