from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    ElectraTokenizer,
    ElectraForMaskedLM,
)
import torch
import logging
import platform


def load_encoder(model_path: str, model_type: str, tokenizer_path: str):
    """
    Load encoder model using HF transformers library

    Parameters:
        model_path (str) -- path to local model or model repo (from config)
        model_type (str) -- model achitecture (from conifg)
        tokenizer_pathn(str) -- path to local tokenizer or tokenizer repo (from config)

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
