# -*- coding: utf-8 -*-
"""
@Author  : Haixin Wu
@Time    : 2025/2/16 17:36
@Email   : whx3412@163.com
@Function: 
"""
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, frontend: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 dec_out_dim: int, vocab: int
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)

    def forward(self, X: torch.Tensor, X_lens: torch.Tensor, labels:torch.Tensor):
        X_lens, labels = X_lens.long(), labels.long()
        b = X.size(0)
        device = X.device
        # frontend
        out = self.frontend(X)
        max_feat_len = out.size(1)
        max_label_len = labels.size(1)
        # encoder
        enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        enc_out = self.encoder(out, X_lens, enc_mask)
        # decoder
        dec_mask = get_subsequent_mask(b, max_feat_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        logits = self.linear(dec_out)

        return logits



def get_len_mask(b: int, max_len: int, feat_lens:torch.Tensor, device: torch.Tensor) -> torch.Tensor:
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)

def get_subsequent_mask(b: int, max_len: int, device: torch.Tensor) -> torch.Tensor:
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(
        torch.bool
    )

def get_enc_dec_mask(b: int, max_feat_len:int, feat_lens: torch.Tensor, max_label_len: int, device: torch.Tensor) -> torch.Tensor:
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)
