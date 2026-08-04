import logging
from typing import Any, Dict

from huggingface_hub import hf_hub_download


def resolve_concordance_path(config_entry: Dict[str, Any]) -> str:
    """
    Resolve config formular_concordance entry to local path via HF

    Parameters:
        config_entry ( Dict[str, Any] ):
            repo_id (str) -- HF repo
            filename (str) -- file in repo
            repo_type (str) -- repo type
            revision (str) -- commit branch
    """
    kwargs = dict(
        repo_id=config_entry["repo_id"],
        filename=config_entry["filename"],
        repo_type=config_entry.get("repo_type", "dataset"),
        revision=config_entry.get("revision"),
    )
    try:
        return hf_hub_download(**kwargs)
    except Exception as e_remote:
        logging.info(f"Unable to reach remote concordance {kwargs['repo_id']}/{kwargs['filename']}: {e_remote}")
        return hf_hub_download(**kwargs, local_files_only=True)
