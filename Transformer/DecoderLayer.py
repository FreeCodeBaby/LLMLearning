# -*- coding: utf-8 -*-
"""
@Author  : Haixin Wu
@Time    : 2025/2/16 16:41
@Email   : whx3412@163.com
@Function: 
"""
from torch import nn

from MultiHeadAttention import MultiHeadAttention
from PoswiseFFN import PoswiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        super(DecoderLayer, self).__init__()
        assert dim % n == 0
        hdim = dim // n
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)
        self.dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)

    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask, cache=None, freqs_cis=None):
        residual = dec_in
        context = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        dec_out = self.norm1(residual + context)
        # encoder-decoder cross attention
        residual = dec_out
        context = self.enc_dec_attn(dec_out, enc_out, enc_out, dec_enc_mask)
        dec_out = self.norm2(residual + context)

        residual = dec_out
        out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(residual + out)
        return dec_out






















