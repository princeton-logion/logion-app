from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    ElectraTokenizer,
    ElectraForMaskedLM,
)
import torch
import logging
import platform

import os
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import CanineConfig
from . import char_model
from huggingface_hub import hf_hub_download


def load_character_mlm(model_path: str, vocab_path: str = None, use_auth_token: str = None, cache_dir: str = None):
    """
    Load custom character model from HF or local

        Paramters:
            model_path (str) --
            vocab_path (str) --
            use_auth_token (str) --
            cache_dir (str) -- 
        
        Returns:
            model --
            char_stoi -- 
            char_itos --
            mask_id -- 
    """
    is_remote = "/" in model_path and not os.path.exists(model_path)
    
    if is_remote:
        print(f"Retrieving model from HF: {model_path}...")
        
        try:
            config_path = hf_hub_download(
                repo_id=model_path,
                filename="config.json",
                token=use_auth_token,
                cache_dir=cache_dir
            )
            
            model_file = None
            try:
                model_file = hf_hub_download(
                    repo_id=model_path,
                    filename="model.safetensors",
                    token=use_auth_token,
                    cache_dir=cache_dir
                )
                use_safetensors = True
            except:
                try:
                    model_file = hf_hub_download(
                        repo_id=model_path,
                        filename="pytorch_model.bin",
                        token=use_auth_token,
                        cache_dir=cache_dir
                    )
                    use_safetensors = False
                except:
                    raise FileNotFoundError(f"Cannot find model weights at {model_path}")
            
            if vocab_path is None:
                try:
                    vocab_path = hf_hub_download(
                        repo_id=model_path,
                        filename="vocab.json",
                        token=use_auth_token,
                        cache_dir=cache_dir
                    )
                except:
                    raise FileNotFoundError(f"Cannot find vocab at {model_path}")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
            
    else:
        config_path = os.path.join(model_path, "config.json")
        model_file_sf = os.path.join(model_path, "model.safetensors")
        model_file_bin = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(model_file_sf):
            model_file = model_file_sf
            use_safetensors = True
        elif os.path.exists(model_file_bin):
            model_file = model_file_bin
            use_safetensors = False
        else:
            raise FileNotFoundError(f"Cannot find model weights at {model_path}")
            
        if vocab_path is None:
            vocab_path = os.path.join(model_path, "vocab.json")
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Cannot find vocab at {model_path}")
    
    print(f"Retrieving model vocab...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_stoi = json.load(f)
    char_itos = {v: k for k, v in char_stoi.items()}
    pad_id = char_stoi["[PAD]"]
    mask_id = char_stoi["[MASK]"]
    vocab_size = len(char_stoi)
    
    print(f"Retrieving model config...")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = CanineConfig(**{k: v for k, v in config_dict.items() 
                            if k in CanineConfig().to_dict()})
    
    if all(k in config_dict for k in ['gbst_dim', 'mask_token_id']):
        training_config = {
            'gbst_dim': config_dict['gbst_dim'],
            'gbst_max_block_size': config_dict.get('gbst_max_block_size', 8),
            'gbst_downsample_factor': config_dict.get('gbst_downsample_factor', config.downsampling_rate),
            'num_hash_functions': config_dict.get('num_hash_functions', 8),
            'num_hash_buckets': 16000,
            'max_span_length': config_dict.get('max_span_length', 10)
        }
    else:
        training_config = {
            'gbst_dim': config.hidden_size,
            'gbst_max_block_size': 8,
            'gbst_downsample_factor': config.downsampling_rate,
            'num_hash_functions': 8,
            'num_hash_buckets': 16000,
            'max_span_length': 10
        }
    
    print(f"Initializing model architecture...")
    model = char_model.CharformerCanineForMaskedLM(
        config=config,
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
        gbst_dim=training_config['gbst_dim'],
        gbst_max_block_size=training_config['gbst_max_block_size'],
        gbst_downsample_factor=training_config['gbst_downsample_factor'],
        num_hash_functions=training_config['num_hash_functions'],
        num_hash_buckets=training_config['num_hash_buckets'],
        max_span_length=training_config['max_span_length']
    )
    
    print(f"Retrieving model weights...")
    if use_safetensors:
        state_dict = load_file(model_file)
    else:
        state_dict = torch.load(model_file, map_location='cpu')
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Model loaded successfully from {'HF' if is_remote else 'local path'}")
    
    return model, char_stoi, char_itos, mask_id

def load_encoder(model_path: str, model_type: str, tokenizer_path: str):
    """
    Load encoder model using HF transformers library

    Parameters:
        model_path (str) -- path to local model or model repo (from config)
        model_type (str) -- model achitecture (from conifg)
        tokenizer_path (str) -- path to local tokenizer or tokenizer repo (from config)

    Return:
        model (eval mode)
        tokenizer
    """
    try:
        logging.info(f"Loading model from {model_path}\nLoading tokenizer from {tokenizer_path}")
        if model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            model = BertForMaskedLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Invalid model/tokenizer selected.")
        return model.eval(), tokenizer
    except Exception as e:
        logging.info(f"Unable to load model/tokenizer: {e}")
        raise


def load_device(model: torch.nn.Module):
    """
    Load model to device

    Parameters:
        model (torch.nn.Module) -- model in eval mode

    Returns:
        device -- loaded device (cuda, mps or cpu)
        model -- model loaded to GPU/CPU
    """
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        device = torch.device("cuda")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        logging.info("MPS Metal available")
        device = torch.device("mps")
    # Intel XPU under construction
    #elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        #logging.info("Intel XPU available")
        #device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device {device}.")
    model.to(device)
    return device, model


def patch_char_model_for_mps(model: torch.nn.Module):
    """
    Disable nested tensor in TransformerEncoder layers per 
    'aten::_nested_tensor_from_mask_left_aligned' MPS error
    """
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoder):
            module.enable_nested_tensor = False
            module.use_nested_tensor = False

    return model
