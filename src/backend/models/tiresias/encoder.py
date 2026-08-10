#!/usr/bin/env python3
"""
encoder.py

Deep encoder components
    build deep encoder stack for processing molecules w/ relative pos embeddings

Largely based on LTG-BERT (see ltgoslo/ltg-bert) w/
    DeBERTa's disentangled self-attn (see microsoft/DeBERTa)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .disentangled_attn import DisentangledSelfAttention


class GeGLU(nn.Module):
    """
    Gated GELU from model.GeGLU from ltgoslo/ltg-bert repo
    """
    def forward(self, x) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class DeepFeedForward(nn.Module):
    """
    FF module for deep encoder

    Based on model.FeedForward from ltgoslo/ltg-bert repo
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float=0.1, layer_norm_eps: float=1e-5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=layer_norm_eps, elementwise_affine=False),
            nn.Linear(hidden_size, 2 * intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(intermediate_size, eps=layer_norm_eps, elementwise_affine=False),
            nn.Linear(intermediate_size, hidden_size, bias=False),
            nn.Dropout(dropout),
        )
        self._initialize(hidden_size)

    def _initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)





class DeepEncoderLayer(nn.Module):
    """
    Single encoder layer w/ disentangled self-attn

    Based on model.EncoderLayer from ltgoslo/ltg-bert repo
    """

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 position_buckets, max_relative_positions,
                 dropout=0.1, layer_norm_eps=1e-5, share_att_key=False):
        super().__init__()
        self.attention = DisentangledSelfAttention(
            hidden_size, num_heads, position_buckets, max_relative_positions,
            dropout, layer_norm_eps, share_att_key,
        )
        self.ffn = DeepFeedForward(hidden_size, intermediate_size, dropout, layer_norm_eps)

    def forward(self, x, rel_embeddings, key_padding_mask=None) -> torch.Tensor:
        x = x + self.attention(x, rel_embeddings, key_padding_mask=key_padding_mask)
        x = x + self.ffn(x)
        return x


class DeepEncoder(nn.Module):
    """
    Deep encoder stack for molecules

    Largely BERT-like w/ key changes:
        1) disentangled attn w/ relative molecule pos embeddings (DeBERTa)
        2) pre-layer LayerNorm of embeddings (LTG-BERT)
        3) layer-wise feed-forward scaling (LTG-BERT)
    """

    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size,
                 position_buckets=256, max_relative_positions=2048,
                 dropout=0.1, layer_norm_eps=1e-5, share_att_key=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.gradient_checkpointing = False

        self.rel_embeddings = nn.Embedding(2 * position_buckets, hidden_size)
        self.rel_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.layers = nn.ModuleList([
            DeepEncoderLayer(
                hidden_size, num_heads, intermediate_size,
                position_buckets, max_relative_positions,
                dropout, layer_norm_eps, share_att_key,
            )
            for _ in range(num_layers)
        ])
        self._apply_ltg_init()

    def _apply_ltg_init(self):
        """
        Re/apply deep encoder init because HF PreTrainedModel uses uniform nn.Linear/nn.LayerNorm weights:
          1) Truncate init per relative pos embeds
          2) Truncate init per self-attn + FFN projs
          3) Layer-wise FF scaling
        """
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.rel_embeddings.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

        for layer in self.layers:
            layer.attention._initialize()
            layer.ffn._initialize(self.hidden_size)

        for i, layer in enumerate(self.layers):
            scale = math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.ffn.mlp[1].weight.data *= scale
            layer.ffn.mlp[-2].weight.data *= scale

    def get_rel_embeddings(self) -> torch.Tensor:
        return self.rel_layer_norm(self.rel_embeddings.weight)

    def forward(self, x, key_padding_mask=None) -> torch.Tensor:
        rel_embeddings = self.get_rel_embeddings()

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    layer.__call__, x, rel_embeddings, key_padding_mask
                )
            else:
                x = layer(x, rel_embeddings, key_padding_mask=key_padding_mask)
        return x