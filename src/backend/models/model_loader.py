from transformers import BertTokenizer, BertForMaskedLM, ElectraTokenizer, ElectraForMaskedLM
import torch
import logging
import platform


def load_encoder(model_name: str, model_type: str):
    """
    Load encoder-only model using HF transformers library
    
    Parameters:
        model_name (str) -- name of model (received from front-end list)
    
    Return:
        model (in eval mode) and tokenizer
    """
    try:
        tokenizer_name = "princeton-logion/LOGION-50k_wordpiece"
        if not model_name.startswith("princeton-logion"):
          raise ValueError(f"{model_name} not an available Logion model.")
        elif model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
            model = BertForMaskedLM.from_pretrained(model_name)
        elif model_type == "electra":
            tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
            model = ElectraForMaskedLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Invalid model type selected.")
        return model.eval(), tokenizer
    except Exception as e:
        logging.info(f"Error loading model {model_name}: {e}")
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
        device = torch.device("cuda:0")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
         logging.info("MPS (Metal) is available")
         device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device {device}.")
    model.to(device)
    return device, model