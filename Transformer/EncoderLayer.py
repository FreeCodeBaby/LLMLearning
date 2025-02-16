# -*- coding: utf-8 -*-
"""
@Author  : Haixin Wu
@Time    : 2025/2/16 16:13
@Email   : whx3412@163.com
@Function: 
"""
from torch import nn
from MultiHeadAttention import MultiHeadAttention
from PoswiseFFN import PoswiseFFN


class EncoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        assert dim % n == 0
        hdim = dim // n
        super(EncoderLayer, self).__init__()

        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # MultiHeadAttention
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)

    def forward(self, enc_in, attn_mask):
        residual = enc_in
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        out = self.norm1(residual + context)
        residual = out
        out = self.poswise_ffn(out)
        out = self.norm2(residual + out)

        return out
