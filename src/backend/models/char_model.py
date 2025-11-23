import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import CanineConfig
from transformers.modeling_outputs import MaskedLMOutput
from . import gbst_mod
from transformers.models.canine.modeling_canine import CaninePreTrainedModel





class CharacterHashEmbedding(nn.Module):

    def __init__(self,
                 num_hashes: int = 8,
                 embedding_dim: int = 768,
                 num_buckets: int = 16000,
                 dropout=0.1):
        super().__init__()
        self.num_hashes = num_hashes
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_buckets, embedding_dim // num_hashes)
            for _ in range(num_hashes)
        ])
        self.learnt_embeddings = nn.Embedding(65536, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _hash(self, ids: torch.Tensor, hash_id: int) -> torch.Tensor:
        primes = [2654435761, 2654435789, 2654435827, 2654435837,
                  2654435843, 2654435849, 2654435859, 2654435879]
        prime = primes[hash_id % len(primes)]
        return ((ids.long() * prime) % self.num_buckets).long()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        hash_embeds = []
        for hash_id, embedding_layer in enumerate(self.embeddings):
            hashed_ids = self._hash(input_ids, hash_id)
            hash_embeds.append(embedding_layer(hashed_ids))
        hash_embedding = torch.cat(hash_embeds, dim=-1)
        learnt_embedding = self.learnt_embeddings(torch.clamp(input_ids, max=65535))
        output = hash_embedding + learnt_embedding
        return self.dropout(output)

    
class GBSTWithExternalEmbeddings(nn.Module):
    
    def __init__(self, 
                 **kwargs):
        super().__init__()
        
        self.gbst_original = gbst_mod.GBST(**kwargs)
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None):

        return self.gbst_original.forward_from_embeddings(embeddings, mask)


        

class CharformerCanineForMaskedLM(CaninePreTrainedModel):

    def __init__(self,
                 config: CanineConfig,
                 vocab_size: int,
                 pad_token_id: int,
                 mask_token_id: int,
                 gbst_dim: int = 768,
                 gbst_max_block_size: int = 8,
                 gbst_downsample_factor: int = 4,
                 num_hash_functions: int = 8,
                 num_hash_buckets: int = 16000,
                 max_span_length: int = 10):
        super().__init__(config)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.downsample_factor = gbst_downsample_factor
        logging.info(f"Initializing CharformerCanineForMaskedLM...")


        self.char_hash_embeddings = CharacterHashEmbedding(
            num_hashes=num_hash_functions,
            embedding_dim=config.hidden_size,
            num_buckets=num_hash_buckets
        )


        self.gbst = GBSTWithExternalEmbeddings(
            num_tokens=vocab_size,
            dim=gbst_dim,
            max_block_size=gbst_max_block_size,
            downsample_factor=gbst_downsample_factor
        )
        self.gbst_proj = nn.Linear(gbst_dim, config.hidden_size) if gbst_dim != config.hidden_size else nn.Identity()

        max_molecule_positions = config.max_position_embeddings // gbst_downsample_factor
        self.molecule_position_embeddings = nn.Embedding(
            max_molecule_positions,
            config.hidden_size
        )
        self.char_position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )


        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=.1, #0.1,
                activation="gelu",
                batch_first=True
            ),
            num_layers=config.num_hidden_layers
        )


        self.upsample_conv = nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=4,
            stride=1,
            padding=2
        )
        self.final_char_encoder = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=.1,
            activation="gelu",
            batch_first=True
        )


        self.mlm_head = nn.Linear(config.hidden_size, vocab_size)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        self.init_weights()

    def _repeat_molecules(self, molecule_embeddings: torch.Tensor, char_seq_length: int) -> torch.Tensor:
        batch_size, num_molecules, hidden_dim = molecule_embeddings.shape
        repeated = molecule_embeddings.unsqueeze(2).repeat(1, 1, self.downsample_factor, 1)
        repeated = repeated.reshape(batch_size, num_molecules * self.downsample_factor, hidden_dim)
        if repeated.size(1) > char_seq_length:
            repeated = repeated[:, :char_seq_length, :]
        elif repeated.size(1) < char_seq_length:
            padding = torch.zeros(
                batch_size,
                char_seq_length - repeated.size(1),
                hidden_dim,
                device=repeated.device,
                dtype=repeated.dtype
            )
            repeated = torch.cat([repeated, padding], dim=1)
        return repeated

    def forward(self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor = None,
            span_boundaries: torch.LongTensor = None):

        batch_size, char_seq_len = input_ids.shape
        device = input_ids.device
        
        char_embeddings = self.char_hash_embeddings(input_ids)
        
        gbst_mask = attention_mask.bool()
        molecule_embeddings, molecule_mask = self.gbst(
            embeddings=char_embeddings,
            mask=gbst_mask
        )
        molecule_embeddings = self.gbst_proj(molecule_embeddings)
        
        molecule_seq_len = molecule_embeddings.size(1)
        molecule_positions = torch.arange(molecule_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        molecule_position_embeds = self.molecule_position_embeddings(molecule_positions)
        molecule_embeddings = molecule_embeddings + molecule_position_embeds

        encoder_attention_mask = molecule_mask.bool()
        molecule_contextualized = self.encoder(
            molecule_embeddings,
            src_key_padding_mask=~encoder_attention_mask
        )
        
        repeated_molecules = self._repeat_molecules(molecule_contextualized, char_seq_len)
        
        combined = torch.cat([char_embeddings, repeated_molecules], dim=-1)
        
        combined_transposed = combined.transpose(1, 2)  # (batch, 2*hidden, seq_len)
        conv_out_raw = self.upsample_conv(combined_transposed)  # (batch, hidden, seq_len+1)
        conv_out = conv_out_raw.transpose(1, 2)  # (batch, seq_len, hidden)
        conv_out = conv_out[:, :char_seq_len, :]

        char_positions = torch.arange(char_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        char_position_embeds = self.char_position_embeddings(char_positions)
        conv_out = conv_out + char_position_embeds
        
        char_attention_mask = attention_mask.bool()
        final_char_embeddings = self.final_char_encoder(
            conv_out,
            src_key_padding_mask=~char_attention_mask
        )
        
        logits = self.mlm_head(final_char_embeddings) + self.mlm_bias
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            
            
            loss = mlm_loss
        
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=final_char_embeddings
        )
