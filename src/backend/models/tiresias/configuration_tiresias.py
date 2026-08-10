#!/usr/bin/env python3

from transformers import PretrainedConfig

class TiresiasConfig(PretrainedConfig):
    """
    Tiresias encoder config class

    Parameters:
        vocab_size: # of chars in vocab
        hidden_size: dim for encoder hidden states
        num_hidden_layers: # of deep encoder layers
        num_attention_heads: # of attn heads for deep + shallow transformers
        intermediate_size: FF intermediate size for shallow transformers
        max_position_embeddings: max seq length for absolute char pos embeddings (upsampling)
        pad_token_id: 
        mask_token_id: 
        gbst_downsample_factor: compression factor of chars -> molecules
        num_hash_functions: # hash functions for char hashing
        num_hash_buckets: has table size
        local_window_size: banded local attn window size (char transformer)
        num_local_layers: # layers in shallow char transformer (pre downsampling)
        position_buckets: # log-spaced buckets per rel pos encoding direction (deep encoder)
        max_relative_positions: max relative dist for log-bucket mapping (deep encoder)
        share_att_key: True = deep encoder disentangled attn reuses content Q/K projections for positional Q/K
        num_final_char_layers: # layers in final character encoder (post upsampling)
        char_position_buckets: # log-spaced buckets per rel pos encoding direction (final char encoder)
        char_max_relative_positions: max relative dist for log-bucket mapping (final char encoder)
        dropout: dropout probability
        layer_norm_eps: LayerNorm epsilon
    """

    model_type = "tiresias"

    def __init__(
        self,
        vocab_size=256,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=8192,
        pad_token_id=0,
        mask_token_id=1,
        gbst_downsample_factor=4,
        num_hash_functions=8,
        num_hash_buckets=16000,
        max_span_length=10,
        local_window_size=128,
        num_local_layers=1,
        position_buckets=256,
        max_relative_positions=2048,
        share_att_key=False,
        num_final_char_layers=3,
        char_position_buckets=128,
        char_max_relative_positions=512,
        dropout=0.1,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.mask_token_id = mask_token_id
        self.gbst_downsample_factor = gbst_downsample_factor
        self.num_hash_functions = num_hash_functions
        self.num_hash_buckets = num_hash_buckets
        self.max_span_length = max_span_length
        self.local_window_size = local_window_size
        self.num_local_layers = num_local_layers
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.share_att_key = share_att_key
        self.num_final_char_layers = num_final_char_layers
        self.char_position_buckets = char_position_buckets
        self.char_max_relative_positions = char_max_relative_positions
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps