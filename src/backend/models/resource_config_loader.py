import logging
import requests
from typing import Dict
import yaml



def load_resource_config(config_path: str) -> Dict:
    """
    Loads model config .yaml file file from url (desktop).

    Fall back to local file if url fails (OnDemand).

    Parameters:
        url (str) -- url to Git-hosted resources_config.yaml

    Returns:
        response.text -- url string from resources_config.yaml
    """
    try:
        response = requests.get(config_path)
        response.raise_for_status()
        return yaml.safe_load(response.text)
    except requests.exceptions.RequestException as e_remote:
        logging.info(f"Unable to load remote resources_config.yaml from {config_path}: {e_remote}")
    # if URL fails, load local .yaml
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e_local:
        logging.info(f"Unable to load local resources_config.yaml from {config_path}: {e_local}")
        raise