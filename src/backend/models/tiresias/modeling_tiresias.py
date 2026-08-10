#!/usr/bin/env python3
"""
modeling_tiresias.py

Tiresias architecture

Character-level encoder-only LM

Integrates developments from:
    - Charformer-CANINE (BORT)
    - LTG-BERT
    - DeBERTa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.utils import logging

from .configuration_tiresias import TiresiasConfig
from .dense_gbst import DenseGBST
from .encoder import DeepEncoder

logger = logging.get_logger(__name__)



def create_local_attention_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """
    Banded local attn mask for windowed self-attn
    """
    positions = torch.arange(seq_len, device=device)
    dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    half_window = window_size // 2
    mask = torch.where(
        dist <= half_window,
        torch.tensor(0.0, device=device),
        torch.tensor(float('-inf'), device=device)
    )
    return mask


class CharEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom nn.TransformerEncoderLayer w/ broadcasted additive mask in self-attn
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal: bool = False) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal: bool = False) -> torch.Tensor:
        if key_padding_mask is not None:
            raise ValueError(
                "CharEncoderLayer takes no key_padding_mask"
            )
        if is_causal:
            raise ValueError("CharEncoderLayer doesn't support is_causal=True")
        mha = self.self_attn
        batch_size, seq_len, _ = x.shape
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        query, key, value = (
            t.view(batch_size, seq_len, mha.num_heads, mha.head_dim).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        )
        if attn_mask is not None and attn_mask.is_floating_point() \
                and attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(query.dtype)
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask,
            dropout_p=mha.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch_size, seq_len, mha.embed_dim)
        return self.dropout1(mha.out_proj(context))


class CharHashEmbedding(nn.Module):
    def __init__(self, num_hashes: int=8, embedding_dim: int=768, num_buckets: int=16000, vocab_size: int=256, dropout: float=0.1):
        super().__init__()
        self.num_hashes = num_hashes
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_buckets, embedding_dim // num_hashes)
            for _ in range(num_hashes)
        ])
        self.learnt_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    # multiple hashes per char to avoid collisions
    _HASH_COPRIMES = (2654435761, 2654435789, 2654435827, 2654435837,
                         2654435843, 2654435849, 2654435859, 2654435879)
    def _hash(self, ids, hash_id):
        coprimes = self._HASH_COPRIMES[hash_id % len(self._HASH_COPRIMES)]
        return ((ids.long() * coprimes) % self.num_buckets).long()

    def forward(self, input_ids) -> torch.Tensor:
        hash_embeds = []
        for hash_id, embedding_layer in enumerate(self.embeddings):
            hashed_ids = self._hash(input_ids, hash_id)
            hash_embeds.append(embedding_layer(hashed_ids))
        hash_embedding = torch.cat(hash_embeds, dim=-1)
        learnt_embedding = self.learnt_embeddings(input_ids)
        output = hash_embedding + learnt_embedding
        return self.dropout(output)
    

class TiresiasPreTrainedModel(PreTrainedModel):
    config_class = TiresiasConfig
    base_model_prefix = "tiresias"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """
        Duplicate CANINE _init_weights + null check prior to accessing bias,
            (will overwrite only w/r/t deep encoder via _apply_ltg_init())
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def _initialize_missing_keys(self, *args, **kwargs):
        """
        Preserve checkpoint weights from preivous checkpoint/pretrained model
        """
        for module in self.modules():
            params = list(module.parameters(recurse=False))
            buffers = [b for b in module.buffers(recurse=False) if b is not None]
            if not params and not buffers:
                continue
            if all(getattr(p, "_is_hf_initialized", False) for p in params) and all(
                getattr(b, "_is_hf_initialized", False) for b in buffers
            ):
                module._is_hf_initialized = True
        return super()._initialize_missing_keys(*args, **kwargs)


class TiresiasModel(TiresiasPreTrainedModel):
    """
    TiresiasModel architecture:
        1. Hash-based char embeddings
        2. Shallow BERT-like char transformer lyr
        3. Charformer downsampling (compress chars -> molecules)
        4. Deep encoder (num_hidden_layers=layers) w/:
            - GeGLU
            - NormFormer Pre-/Post-LayerNorm (LTG-BERT)
            - disentangled attn on molecule relative pos (DeBERTa)
        5. CANINE upsampling (expand molecules -> chars)
        6. Final char encoder (num_final_char_layers=layers) w/:
            - same architecture as deep encoder
            - char-level relative pos embeddings
            - layer-wise FFN scaling

    Return char hidden states to finetune w/ task heads
    """

    def __init__(self,
                 config: TiresiasConfig,
                 _skip_post_init: bool = False):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.mask_token_id = config.mask_token_id
        self.downsample_factor = config.gbst_downsample_factor
        self.local_window_size = config.local_window_size
        self.gradient_checkpointing = False

        # GeGLU intermediate = 2/3 of standard intermediate for param parity
        self.geglu_intermediate_size = int(config.intermediate_size * 2 / 3)

        self.char_hash_embeddings = CharHashEmbedding(
            num_hashes=config.num_hash_functions,
            embedding_dim=config.hidden_size,
            num_buckets=config.num_hash_buckets,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
        )

        self.local_transformer = nn.TransformerEncoder(
            CharEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.num_local_layers,
        )

        self.gbst = DenseGBST(
            dim=config.hidden_size,
            filter_widths=(1, 2, 3, 4, 5),
            downsample_factor=config.gbst_downsample_factor,
            score_consensus_attn=True,
            score_hidden_dim=128,
        )

        # keep char absolute pos embeddings for upsampling + prediction (per DeBERTa)
        self.char_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.encoder = DeepEncoder(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=self.geglu_intermediate_size,
            position_buckets=config.position_buckets,
            max_relative_positions=config.max_relative_positions,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            share_att_key=config.share_att_key,
        )

        self.upsample_conv = nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=config.gbst_downsample_factor,
            stride=1,
            padding=config.gbst_downsample_factor // 2,
        )

        self.final_char_encoder = DeepEncoder(
            num_layers=config.num_final_char_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=self.geglu_intermediate_size,
            position_buckets=config.char_position_buckets,
            max_relative_positions=config.char_max_relative_positions,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            share_att_key=config.share_att_key,
        )

        # avoid double init by skipping when TiresiasModel built as child of TiresiasForMaskedLM
        if not _skip_post_init:
            self.post_init()
            self.encoder._apply_ltg_init()
            self.final_char_encoder._apply_ltg_init()

    def _repeat_molecules(self, molecule_embeddings, char_seq_length):
        batch_size, num_molecules, hidden_dim = molecule_embeddings.shape
        repeated = molecule_embeddings.unsqueeze(2).repeat(1, 1, self.downsample_factor, 1)
        repeated = repeated.reshape(batch_size, num_molecules * self.downsample_factor, hidden_dim)
        if repeated.size(1) > char_seq_length:
            repeated = repeated[:, :char_seq_length, :]
        elif repeated.size(1) < char_seq_length:
            padding = torch.zeros(
                batch_size, char_seq_length - repeated.size(1), hidden_dim,
                device=repeated.device, dtype=repeated.dtype
            )
            repeated = torch.cat([repeated, padding], dim=1)
        return repeated

    def forward(self, input_ids, attention_mask=None, return_dict=None, **kwargs):
        batch_size, char_seq_len = input_ids.shape
        device = input_ids.device

        return_dict = True if return_dict is None else return_dict

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, char_seq_len), dtype=torch.long, device=device
            )

        # Char embeddings
        char_embeddings = self.char_hash_embeddings(input_ids)

        # Shallow char transformer
        local_attn_mask = create_local_attention_mask(
            seq_len=char_seq_len, window_size=self.local_window_size, device=device
        )
        char_padding_mask = ~attention_mask.bool()

        local_mask = local_attn_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        local_mask.masked_fill_(char_padding_mask.unsqueeze(1), float('-inf'))
        diag = torch.arange(char_seq_len, device=device)
        local_mask[:, diag, diag] = 0.0
        local_mask = local_mask.unsqueeze(1)

        if self.gradient_checkpointing and self.training:
            char_embeddings = self._gradient_checkpointing_func(
                self.local_transformer.__call__, char_embeddings, local_mask
            )
        else:
            char_embeddings = self.local_transformer(char_embeddings, mask=local_mask)

        # Charformer downsampling for molecules
        molecule_embeddings, molecule_mask = self.gbst(
            embeddings=char_embeddings, mask=attention_mask.bool()
        )

        # Deep LTG-BERT encoder w/ disentangled attn + absolute molecule pos embeddings
        molecule_contextualized = self.encoder(
            molecule_embeddings, key_padding_mask=~molecule_mask.bool(),
        )

        # Upsample back to chars
        repeated_molecules = self._repeat_molecules(molecule_contextualized, char_seq_len)
        combined = torch.cat([char_embeddings, repeated_molecules], dim=-1)
        conv_out = self.upsample_conv(combined.transpose(1, 2)).transpose(1, 2)
        conv_out = conv_out[:, :char_seq_len, :]

        # Add char absolute pos embeddings (per DeBERTa)
        char_positions = torch.arange(char_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        conv_out = conv_out + self.char_position_embeddings(char_positions)

        # Shallow encoder w/ relative char pos embeddings
        final_char_embeddings = self.final_char_encoder(
            conv_out, key_padding_mask=char_padding_mask,
        )

        if not return_dict:
            return (final_char_embeddings,)

        return BaseModelOutput(last_hidden_state=final_char_embeddings)



class TiresiasMLMHead(nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.initial_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.initial_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states) + self.bias
        return logits

class TiresiasForMaskedLM(TiresiasPreTrainedModel):
    _tied_weights_keys = {
        "mlm_head.decoder.weight": "tiresias.char_hash_embeddings.learnt_embeddings.weight"
    }

    def __init__(self,
                 config: TiresiasConfig):
        super().__init__(config)
        # ensure HF weight tying always on for safety
        self.config.tie_word_embeddings = True
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.mask_token_id = config.mask_token_id
        logger.info("Initializing TiresiasForMaskedLM...")

        # skip possible child re-init to keep init identical to orig (avoid double init)
        self.tiresias = TiresiasModel(config, _skip_post_init=True)

        self.mlm_head = TiresiasMLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size
        )

        # PreTrainedModel assumes uniform init
        # apply deep encoder init + weights after uniform init
        # then 1 tensor for char embeddings + pred
        self.post_init()
        self.tiresias.encoder._apply_ltg_init()
        self.tiresias.final_char_encoder._apply_ltg_init()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model has {total_params / 1e6:.1f}M parameters")

    def get_input_embeddings(self):
        return self.tiresias.char_hash_embeddings.learnt_embeddings

    def set_input_embeddings(self, value):
        self.tiresias.char_hash_embeddings.learnt_embeddings = value

    def get_output_embeddings(self):
        return self.mlm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_head.decoder = new_embeddings

    def tie_weights(self, *args, **kwargs):
        super().tie_weights(*args, **kwargs)
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()
        if output_embeddings is not None and input_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight


    def forward(self, input_ids, attention_mask=None, labels=None,
                return_dict=None, **kwargs) -> MaskedLMOutput:
        return_dict = True if return_dict is None else return_dict

        outputs = self.tiresias(input_ids, attention_mask, return_dict=return_dict)
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]

        logits = self.mlm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (logits, hidden_states)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=(hidden_states,))