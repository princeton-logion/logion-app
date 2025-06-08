from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    ElectraTokenizer,
    ElectraForMaskedLM,
)
import torch
import logging
import platform


def load_encoder(model_name: str, model_type: str):
    """
    Load encoder model using HF transformers library

    Parameters:
        model_name (str) -- name of model

    Return:
        model (in eval mode) and tokenizer
    """
    try:
        tokenizer_name = "princeton-logion/LOGION-50k_wordpiece"
        if not model_name.startswith("princeton-logion"):
            raise ValueError(f"{model_name} not a valid Logion model.")
        elif model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
            model = BertForMaskedLM.from_pretrained(model_name)
        elif model_type == "electra":
            tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
            model = ElectraForMaskedLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Invalid model selected.")
        return model.eval(), tokenizer
    except Exception as e:
        logging.info(f"Unable to load model {model_name}: {e}")
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
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        logging.info("Intel XPU available")
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device {device}.")
    model.to(device)
    return device, model
