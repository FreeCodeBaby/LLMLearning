# -*- coding: utf-8 -*-
"""
@Author  : Haixin Wu
@Time    : 2025/2/16 17:18
@Email   : whx3412@163.com
@Function: 
"""
import torch.nn as nn
from DecoderLayer import DecoderLayer
import torch
import numpy as np


class Decoder(nn.Module):
    def __init__(self, dropout_emb, dropout_posffn, dropout_attn, num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size):
        super(Decoder, self).__init__()

        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in
                range(num_layers)
            ]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask, cache=None):
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)

        for layer in self.layers:
            dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out

def pos_sinusoid_embedding(seq_len, d_model):
    embeddings = torch.zeros(seq_len, d_model)
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()












