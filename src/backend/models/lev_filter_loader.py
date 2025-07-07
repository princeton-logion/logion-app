import torch
import logging
import tempfile
import requests
import numpy as np


# comprehensive seed to ensure reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def load_filter(lev_filter_path: str) -> torch.tensor:
    """
    Loads Levenshtein filter matrix .npy file from url (desktop):
        - Stream .npy matrix piecemeal to temp local file
        - Load temp file to numpy array
        - Convert array to tensor
    Avoids wholesale download of filter to avoid OOM kill.

    Fall back to local file if url fails (OnDemand).

    Parameters:
        url (str) -- url to HF-hosted .npy file

    Returns:
        torch.tensor -- filter matrix as torch tensor
    """
    try:
        logging.info(f"Retrieving Lev filter from {lev_filter_path}")
        with tempfile.NamedTemporaryFile(delete=True, suffix=".npy") as tmp_file:
            with requests.get(lev_filter_path, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file.flush()
            lev_matrix_numpy = np.load(tmp_file.name)
            return torch.from_numpy(lev_matrix_numpy)
    except requests.exceptions.RequestException as e_remote:
        logging.info(f"Unable to load remote Lev filter from {lev_filter_path}: {e_remote}")
    # if URL fails, load local .npy
    try:
        with open(lev_filter_path, "rb") as file:
            lev_matrix_numpy = np.load(file)
            return torch.from_numpy(lev_matrix_numpy)
    except Exception as e_local:
        logging.info(f"Unable to load local Lev filter from {lev_filter_path}: {e_local}")
        raise