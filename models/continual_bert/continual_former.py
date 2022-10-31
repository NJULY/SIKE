import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from ..structure_based.knowformer import Knowformer
from ..utils import get_ranks, get_norms, get_scores


class ContinualFormer(nn.Module):
    def __init__(self, encoder_config: dict, label_smoothing=0.8):
        super(ContinualFormer, self).__init__()

        self.config = {
            'encoder_config': encoder_config
        }

        self.encoder = Knowformer(encoder_config)

        self.ce_loss_fc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse_loss_fc = nn.MSELoss()

    def forward(self, batch_data, batch_idx):
        output = self.link_prediction(batch_data, batch_idx)
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch, batch_idx)
        return output['loss']

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 4)

    def validation_step(self, batch, batch_idx):
        output = self.link_prediction(batch, batch_idx)

        return output['loss'].item(), output['rank']

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = get_scores(rank, loss)
        return scores

    def link_prediction(self, batch: dict, batch_idx: int):
        device = batch.pop('device', 'cpu')
        supports = batch.pop('supports', None)
        mse_weight = batch.pop('mse_weight', -1.0)

        input_ids = batch['struc_inputs'].to(device)
        labels = batch['labels'].to(device)
        filters = batch['filters'].to(device) if batch['filters'] is not None else None
        ent_num = batch['ent_num']

        output = self.encoder(input_ids)

        logits = output['without_neighbors'][:, 0: ent_num]

        loss = self.ce_loss_fc(logits, labels)
        if mse_weight > 0.0:
            mse_loss = self.feature_distill(batch, supports, device)
            if mse_loss is not None:
                if batch_idx == 0:
                    print(f'分类损失{loss}; MSE权重: {mse_weight}; MSE损失{mse_loss}')
                loss = loss + mse_weight * mse_loss

        rank = self._get_ranks(F.softmax(logits, dim=-1), labels, filters) if filters is not None else None

        return {'loss': loss, 'rank': rank, 'logits': logits}

    def feature_distill(self, batch_data, supports, device):
        if supports is None:
            return None

        align_dit = supports['align_dict']
        current_ents = supports['current_ents']
        former_ents = supports['former_ents']
        former_embeds = supports['former_embeds']

        heads = [h for h, _, _ in batch_data['triples'] if h in align_dit]
        if len(heads) == 0:
            return None

        current_token_ids = torch.tensor([current_ents[h]['token_id'] for h in heads]).to(device)
        current_head_embeds = self.encoder.ele_embedding(current_token_ids)

        former_token_ids = torch.tensor([former_ents[align_dit[h]]['token_id'] for h in heads]).to(device)
        former_head_embeds = former_embeds(former_token_ids)

        mse_loss = self.mse_loss_fc(current_head_embeds, former_head_embeds)
        return mse_loss

    def _get_ranks(self, probs, labels, filters=None):
        log_probs = torch.log(probs)
        if filters is not None:
            log_probs[filters[:, 0], filters[:, 1]] = -torch.inf

        sorted_idx = torch.argsort(log_probs, dim=-1, descending=True)
        labels = labels.unsqueeze(dim=1)
        rank = torch.nonzero(torch.eq(sorted_idx, labels))[:, 1] + 1 
        return rank.cpu().numpy().tolist()

    def get_optimizer(self, lr):
        opt = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        return opt

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
