#!/usr/bin/env python3

"""
Disentangled Self-Attn utils

Based on microsoft/DeBERTa repo
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_log_bucket_position(relative_pos, bucket_size, max_position) -> torch.Tensor:
    """
    Map relative pos offsets -> log-spaced bucket idxs

    Based on da_utils.make_log_bucket_dict() from microsoft/DeBERTa repo
    """
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1, device=relative_pos.device, dtype=relative_pos.dtype),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos.float() / mid)
            / math.log((max_position - 1) / mid)
            * (mid - 1)
        ).to(relative_pos.dtype)
        + mid
    )
    bucket_pos = torch.where(
        abs_pos <= mid, relative_pos, (log_pos * sign).to(relative_pos)
    ).long()
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1,
                            max_position=-1, device=None) -> torch.Tensor:
    """
    Build relative pos idx matrix

    Based on da_utils.build_relative_position() from microsoft/DeBERTa repo
    """
    q_ids = torch.arange(0, query_size, device=device)
    k_ids = torch.arange(0, key_size, device=device)
    rel_pos_ids = q_ids.unsqueeze(1) - k_ids.unsqueeze(0)
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    return rel_pos_ids.unsqueeze(0)


class DisentangledSelfAttention(nn.Module):
    """
    Multi-head self-attn w/:
        - disentangled attention (DeBERTa)
        - NormFormer Pre-/Post-LayerNorm (LTG-BERT)

    Based on modeling_bert.BertSelfAttention from huggingface/transformers
        + disentangled_attention from microsoft/DeBERTa
        + model.Attention from ltgoslo/ltg-bert     
    """

    def __init__(self, hidden_size: int, num_heads: int, position_buckets: int, max_relative_positions: int,
                 dropout: float=0.1, layer_norm_eps: float=1e-5, share_att_key=False):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} not a multiple of number of attention heads {num_heads}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.share_att_key = share_att_key

        # scaling factor for 3 attn terms: c2c, c2p, p2c
        self.scale = 1.0 / math.sqrt(3 * self.head_size)

        # NormFormer pre-/post-layer
        self.pre_layer_norm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps, elementwise_affine=False
        )
        self.post_layer_norm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps, elementwise_affine=True
        )

        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        if not share_att_key:
            self.pos_key_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.pos_query_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(dropout)

        self._initialize()

    def _initialize(self):
        """
        Init projection weights from truncated normal
        """
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        for proj in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            proj.bias.data.zero_()
        if not self.share_att_key:
            for proj in [self.pos_key_proj, self.pos_query_proj]:
                nn.init.trunc_normal_(proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
                proj.bias.data.zero_()

    def _transpose_for_scores(self, x):
        """
        Need to reshape
        (B, T, D) -> (B, num_heads, T, head_size)
        for multi-head attn
        """
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, rel_embeddings, key_padding_mask=None) -> torch.Tensor:

        B, T, _ = x.shape

        # NormFormer pre-LayerNorm
        h = self.pre_layer_norm(x)

        query_states = self._transpose_for_scores(self.query_proj(h))
        key_states = self._transpose_for_scores(self.key_proj(h))
        value_states = self._transpose_for_scores(self.value_proj(h))

        c2c = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale 

        # calculate log-bucketed rel distance for each key + query
        relative_pos = build_relative_position(
            T, T,
            bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
            device=x.device,
        ) 

        att_span = self.position_buckets
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1).long()
        c2p_pos = c2p_pos.expand(B, T, T).unsqueeze(1).expand(B, self.num_heads, T, T)

        rel_emb = self.pos_dropout(rel_embeddings)
        rel_emb = rel_emb.unsqueeze(0)

        if self.share_att_key:
            # reuse content proj to reduce parameters by tieing content + pos
            pos_key = self._transpose_for_scores(self.key_proj(rel_emb))
            pos_query = self._transpose_for_scores(self.query_proj(rel_emb))
        else:
            pos_key = self._transpose_for_scores(self.pos_key_proj(rel_emb))
            pos_query = self._transpose_for_scores(self.pos_query_proj(rel_emb))

        # expand pos projections to batch size
        pos_key = pos_key.expand(B, -1, -1, -1)
        pos_query = pos_query.expand(B, -1, -1, -1)

        c2p_raw = torch.matmul(query_states, pos_key.transpose(-2, -1)) * self.scale
        c2p = torch.gather(c2p_raw, dim=-1, index=c2p_pos)

        p2c_raw = torch.matmul(pos_query, key_states.transpose(-2, -1)) * self.scale
        p2c = torch.gather(p2c_raw, dim=-2, index=c2p_pos)

        # combine attn terms
        scores = c2c + c2p + p2c

        if key_padding_mask is not None:
            # expand (B, T) -> (B, 1, 1, T) so we can send across heads + queries
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, value_states)
        context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        output = self.out_proj(context)
        # NormFormer post-LayerNorm
        output = self.post_layer_norm(output)
        output = self.out_dropout(output)

        return output