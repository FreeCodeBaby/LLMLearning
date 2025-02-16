# -*- coding: utf-8 -*-
"""
@Author  : Haixin Wu
@Time    : 2025/2/16 16:29
@Email   : whx3412@163.com
@Function: 
"""
from torch import nn
import torch
from EncoderLayer import EncoderLayer
import numpy as np


class Encoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, enc_dim, num_heads, dff, tgt_len
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the maximum length of sequences
        """
        super(Encoder, self).__init__()
        self.tgt_len = tgt_len
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, enc_dim), freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )

    def forward(self, X, X_lens, mask=None):
        batch_size, seq_len, d_model = X.shape
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device))
        out = self.emb_dropout(out)
        for layer in self.layers:
            out = layer(out, mask)
        return out

def pos_sinusoid_embedding(seq_len, d_model):
    embeddings = torch.zeros(seq_len, d_model)
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()


















