# -*- coding: utf-8 -*-
"""
@Author  : Haixin Wu
@Time    : 2025/2/16 18:13
@Email   : whx3412@163.com
@Function: 
"""
import torch
import torch.nn as nn

from Decoder import Decoder
from Encoder import Encoder
from Transformer import Transformer

if __name__ == '__main__':
    # constants
    batch_size = 16
    max_feat_len = 100
    fbank_dim = 80
    hidden_dim = 512
    vocab_size = 26
    max_label_len = 100

    # dummy data
    fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim)
    feat_lens = torch.randint(1, max_feat_len, (batch_size,))
    labels = torch.randint(0, 26, (batch_size, max_label_len))
    label_lens = torch.randint(1, 10, (batch_size,))

    # model
    feature_extractor = nn.Linear(fbank_dim, hidden_dim)
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0, num_layers=6,
        enc_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0, num_layers=6,
        dec_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048, tgt_vocab_size=vocab_size
    )
    transformer = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size)

    # forward check
    logits = transformer(fbank_feature, feat_lens, labels)
    print(logits.shape)
