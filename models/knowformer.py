import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .knowformer_encoder import Embeddings, Encoder, truncated_normal_init, norm_layer_init
import time


class Knowformer(nn.Module):
    def __init__(self, config):
        super(Knowformer, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._input_dropout_prob = config['input_dropout_prob']
        self._attention_dropout_prob = config['attention_dropout_prob']
        self._hidden_dropout_prob = config['hidden_dropout_prob']
        self._residual_dropout_prob = config['residual_dropout_prob']
        self._context_dropout_prob = config['context_dropout_prob']
        self._initializer_range = config['initializer_range']
        self._intermediate_size = config['intermediate_size']

        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']

        self.ele_embedding = Embeddings(self._emb_size, self._voc_size, self._initializer_range)

        self.triple_encoder = Encoder(config)

        # 对输入的准备
        self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
        self.context_dropout_layer = nn.Dropout(p=self._context_dropout_prob)

    def __forward_triples(self, triple_ids, context_emb=None):
        # convert token id to embedding
        emb_out = self.ele_embedding(triple_ids)  # (batch_size, 3, embed_size)

        # merge context_emb into emb_out
        if context_emb is not None:
            context_emb = self.context_dropout_layer(context_emb)
            emb_out[:, 0, :] = (emb_out[:, 0, :] + context_emb) / 2

        emb_out = self.input_dropout_layer(emb_out)
        encoder = self.triple_encoder
        emb_out = encoder(emb_out, mask=None)  # (batch_size, 3, embed_size)
        return emb_out

    def __process_mask_feat(self, mask_feat):
        return torch.matmul(mask_feat, self.ele_embedding.lut.weight.transpose(0, 1))

    def forward(self, src_ids, window_ids=None):
        # src_ids: (batch_size, seq_size, 1)
        # window_ids: (batch_size, seq_size) * neighbor_num

        # 1. do not use embeddings from neighbors
        seq_emb_out = self.__forward_triples(src_ids, context_emb=None)
        mask_emb = seq_emb_out[:, 2, :]  # (batch_size, embed_size)
        logits_from_triplets = self.__process_mask_feat(mask_emb)  # (batch_size, vocab_size)

        if window_ids is None:
            return {'without_neighbors': logits_from_triplets, 'with_neighbors': None, 'neighbors': None}

        # 2. encode neighboring triplets
        logits_from_neighbors = []
        embeds_from_neighbors = []
        for i in range(len(window_ids)):
            seq_emb_out = self.__forward_triples(window_ids[i], context_emb=None)
            mask_emb = seq_emb_out[:, 2, :]
            logits = self.__process_mask_feat(mask_emb)

            embeds_from_neighbors.append(mask_emb)
            logits_from_neighbors.append(logits)
        # get embeddings from neighboring triplets by averaging
        context_embeds = torch.stack(embeds_from_neighbors, dim=0)  # (neighbor_num, batch_size, 768)
        context_embeds = torch.mean(context_embeds, dim=0)

        # 3. leverage both the triplet and neighboring triplets
        seq_emb_out = self.__forward_triples(src_ids, context_emb=context_embeds)
        mask_embed = seq_emb_out[:, 2, :]
        logits_from_both = self.__process_mask_feat(mask_embed)

        return {
            'without_neighbors': logits_from_triplets,
            'with_neighbors': logits_from_both,
            'neighbors': logits_from_neighbors,
        }
