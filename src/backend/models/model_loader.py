from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
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
from huggingface_hub import hf_hub_download
from . import tiresias


def load_tiresias(model_path: str,
                  vocab_path: str = None,
                  use_auth_token: str = None,
                  cache_dir: str = None):
    """
    Load Tiresias MLM from HF or local via AutoModel
    """
    if vocab_path is None:
        if os.path.isdir(model_path):
            vocab_path = os.path.join(model_path, "vocab.json")
        else:
            vocab_path = hf_hub_download(repo_id=model_path, filename="vocab.json",
                                         token=use_auth_token, cache_dir=cache_dir)

    with open(vocab_path, "r", encoding="utf-8") as f:
        char_stoi = json.load(f)

    missing = [t for t in ("[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]") if t not in char_stoi]
    if missing:
        raise ValueError(f"vocab.json at {vocab_path} missing special tokens: {missing}")

    char_itos = {v: k for k, v in char_stoi.items()}
    mask_id = char_stoi["[MASK]"]

    logging.info(f"Loading Tiresias model from {model_path}")
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, token=use_auth_token, cache_dir=cache_dir,
        dtype=torch.float32,
    )
    model.eval()

    # as precaution, check model formatted correct
    cfg = model.config
    if cfg.vocab_size != len(char_stoi):
        raise ValueError(f"config vocab_size {cfg.vocab_size} != vocab.json size {len(char_stoi)}")
    if cfg.mask_token_id != mask_id:
        raise ValueError(f"config mask_token_id {cfg.mask_token_id} != vocab.json [MASK] id {mask_id}")
    if cfg.pad_token_id != char_stoi["[PAD]"]:
        raise ValueError(f"config pad_token_id {cfg.pad_token_id} != vocab.json [PAD] id {char_stoi['[PAD]']}")

    return model, char_stoi, char_itos, mask_id

def load_encoder(model_path: str, 
                 model_type: str, 
                 tokenizer_path: str, 
                 model_lang: str, 
                 trust_remote_code: bool):
    """
    Load encoder model using HF transformers library

    Parameters:
        model_path (str) -- path to local model or model repo (from config)
        model_type (str) -- model achitecture (from conifg)
        tokenizer_path (str) -- path to local tokenizer or tokenizer repo (from config)
        subword_encoder_path (str) -- path to tensor2tensor vocab file

    Return:
        model (eval mode)
        tokenizer
    """
    try:
        logging.info(f"Loading model from {model_path}\nLoading tokenizer from {tokenizer_path}")
        if model_type == "bert":
            if model_lang == "la":
                if trust_remote_code:
                    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
                    model = AutoModelForMaskedLM.from_pretrained(tokenizer_path)
            else:
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


# def patch_char_model_for_mps(model: torch.nn.Module):
#     """
#     Disable nested tensor in TransformerEncoder layers per 
#     'aten::_nested_tensor_from_mask_left_aligned' MPS error
#     """
#     for module in model.modules():
#         if isinstance(module, nn.TransformerEncoder):
#             module.enable_nested_tensor = False
#             module.use_nested_tensor = False

#     return model
