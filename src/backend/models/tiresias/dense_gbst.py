#!/usr/bin/env python3
"""
dense_gbst.py

Based on lucidrains/charformer-pytorch
Dense variant based on BORT (Cao 2023)
"""
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def masked_mean(tensor, mask, dim = -1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    mean = tensor.sum(dim = dim) / total_el.clamp(min = 1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean

def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple



class DenseGBST(nn.Module):
    """
    Dense variant of GBST per Cao 2023 w/ modifications:
        1) replace non-overlapping block mean pooling w/ dense sliding-window average pooling
        2) replace orig block sizes w/ filter widths (1, 2, 3, 4, 5)
        3) replace linear scoring w/ 2-layer MLP w/ GeLU
        4) rmv initial pos convolution
    """

    def __init__(self, *, dim, filter_widths=(1, 2, 3, 4, 5),
                 downsample_factor=4, score_consensus_attn=True,
                 score_hidden_dim=128):
        super().__init__()
        self.filter_widths = filter_widths
        self.num_widths = len(filter_widths)
        self.downsample_factor = downsample_factor
        self.score_consensus_attn = score_consensus_attn
        self.dim = dim

        self.score_fn = nn.Sequential(
            nn.Linear(dim, score_hidden_dim),
            nn.GELU(),
            nn.Linear(score_hidden_dim, 1),
            Rearrange('... () -> ...')
        )

    def _dense_masked_pool(self, x_t, mask, w, n):
        """
        Sliding-window masked mean pooling w/ window size w + stride 1
        """
        x_padded = F.pad(x_t, (0, w - 1), value=0.0)

        feat_sum = F.avg_pool1d(x_padded, kernel_size=w, stride=1) * w
        feat_sum = feat_sum[:, :, :n]

        mask_t = mask.float().unsqueeze(1)
        mask_padded = F.pad(mask_t, (0, w - 1), value=0.0)
        valid_count = F.avg_pool1d(mask_padded, kernel_size=w, stride=1) * w
        valid_count = valid_count[:, :, :n]

        block_repr = (feat_sum / valid_count.clamp(min=1.0)).transpose(1, 2)

        block_mask = (valid_count.squeeze(1) > 0.5)

        return block_repr, block_mask

    def _dense_pool(self, x_t, w, n):
        """
        Sliding-window mean pooling w/out masking
        """
        x_padded = F.pad(x_t, (0, w - 1), value=0.0)
        pooled = F.avg_pool1d(x_padded, kernel_size=w, stride=1)
        return pooled[:, :, :n].transpose(1, 2)

    def forward(self, embeddings, mask=None):
        """
        Apply dense GBST downsampling to char embds

        Pipeline:
            1. Pad sequence to nearest multiple of downsample_factor
            2. Zero out padding positions in the input
            3. For each filter width, compute stride-1 sliding-window
               average pooling (with masked mean at boundaries)
            4. Score each width at each position with the 2-layer MLP
            5. Softmax across widths, optionally refine with consensus attn
            6. Weighted combination across widths
            7. Mean-pool downsample to molecule length

        Parameters:
            embeddings: Char embeddings w/ shape (batch, seq_len, dim)
            mask: Boolean mask w/ shape (batch, seq_len)
        """
        b, n, d = embeddings.shape
        ds_factor = self.downsample_factor

        # param dtype for mixed precision
        target_dtype = next(self.parameters()).dtype
        embeddings = embeddings.to(target_dtype)

        m = next_divisible_length(n, ds_factor)
        if n < m:
            embeddings = F.pad(embeddings, (0, 0, 0, m - n), value=0.0)
            if exists(mask):
                mask = F.pad(mask, (0, m - n), value=False)
            n = m

         # prevent char features mixing w/ padding at sample boundaries
        if exists(mask):
            embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)

        # transpose for F.avg_pool1d
        x_t = embeddings.transpose(1, 2)

        block_reprs = []
        block_masks = []

        for w in self.filter_widths:
            if w == 1:
                block_reprs.append(embeddings)
                if exists(mask):
                    block_masks.append(mask.clone())
            else:
                if exists(mask):
                    block_repr, block_mask = self._dense_masked_pool(
                        x_t, mask, w, n
                    )
                    block_reprs.append(block_repr)
                    block_masks.append(block_mask)
                else:
                    block_reprs.append(self._dense_pool(x_t, w, n))

        block_reprs = torch.stack(block_reprs, dim=2)

        scores = self.score_fn(block_reprs)

        if exists(mask):
            block_masks = torch.stack(block_masks, dim=2)
            max_neg_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~block_masks, max_neg_value)

        scores = scores.softmax(dim=2)

        if self.score_consensus_attn:
            score_sim = einsum('b i d, b j d -> b i j', scores, scores)

            if exists(mask):
                cross_mask = (
                    rearrange(mask, 'b i -> b i ()')
                    * rearrange(mask, 'b j -> b () j')
                )
                max_neg_value = -torch.finfo(score_sim.dtype).max
                score_sim = score_sim.masked_fill(~cross_mask, max_neg_value)

            score_attn = score_sim.softmax(dim=-1)
            scores = einsum('b i j, b j m -> b i m', score_attn, scores)

        scores = rearrange(scores, 'b n m -> b n m ()')
        x = (block_reprs * scores).sum(dim=2)

        x = rearrange(x, 'b (n m) d -> b n m d', m=ds_factor)

        if exists(mask):
            mask = rearrange(mask, 'b (n m) -> b n m', m=ds_factor)
            x = masked_mean(x, mask, dim=2)
            mask = torch.any(mask, dim=-1)
        else:
            x = x.mean(dim=2)

        return x, mask