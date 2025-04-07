import os
import sys
import logging
import requests
import io
import yaml
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import model_loader
from utils import prediction_schemas, detection_schemas
from prediction import predict
from detection import logion_class, detect
import random
import include
import uvicorn
import regex as re


# comprehensive seed to ensure reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# set up log
log_file_path = os.environ.get("LOGION_LOG_PATH", "logion-app.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
)
logging.info(f"Path to log: {log_file_path}")


app = FastAPI(title="Logion", port=8000)


# CORS middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow any origin requests - temp for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config_from_url(url):
    """
    Loads model config file file from url

    Parameters:
        url (str) -- url to Git-hosted resources_config.yaml file

    Returns:
        response.text -- url string from resources_config.yaml
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return yaml.safe_load(response.text)
    except requests.exceptions.RequestException as e:
        logging.error(f"Unable to load remote resources_config.yaml: {e}")
    # if URL fails, load local file
    try:
        if hasattr(sys, "_MEIPASS"):
            local_path = os.path.join(sys._MEIPASS, "resources_config.yaml")
        else:
            local_path = os.path.join(
                os.path.dirname(__file__), "resources_config.yaml"
            )
        with open(local_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        logging.error(f"Unable to load local resources_config.yaml: {e}")
        raise


# load resources_config.yaml
logging.info(f"CWD: {os.getcwd()}")
if hasattr(sys, "_MEIPASS"):
    config_path = os.path.join(sys._MEIPASS, "resources_config.yaml")
else:
    config_path = os.path.join(os.path.dirname(__file__), "resources_config.yaml")
logging.info(f"Config file path: {config_path}")
try:
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
        MODEL_CONFIG = config_data["models"]
        LEV_FILTER_URLS = config_data["lev_filter"]
except Exception as e:
    logging.error(f"Unable to load resources_config.yaml: {e}")
    raise SystemExit("Quitting application.") from e


def load_filter_from_url(url):
    """
    Loads Levenshtein filter matrix .npy file from url

    Parameters:
        url (str) -- url to HF-hosted .npy file

    Returns:
        torch.tensor -- filter matrix as torch tensor
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        lev_matrix_file = io.BytesIO(response.content)
        return torch.tensor(np.load(lev_matrix_file))
    except requests.exceptions.RequestException as e:
        logging.error(f"Unable to load remote Lev filter matrix: {e}")

    # if URL fails, load local file
    try:
        if hasattr(sys, "_MEIPASS"):
            local_path = os.path.join(sys._MEIPASS, "lev_filter.npy")
        else:
            local_path = os.path.join(os.path.dirname(__file__), "lev_filter.npy")
        with open(local_path, "rb") as file:
            return torch.tensor(np.load(file))
    except Exception as e:
        logging.error(f"Unable to load local Lev filter matrix: {e}")
        raise


"""
Server status endpoint
"""

@app.get("/health")
async def health_check():
    return {"status": "ok"}


"""
Model retrieval endpoint
"""

@app.get("/models", response_model=list[str])
def get_models():
    config_data = load_config_from_url("https://raw.githubusercontent.com/princeton-logion/logion-app/main/src/backend/resources_config.yaml")
    if "models" in config_data:
        model_names = [model["name"] for model in config_data["models"]]
        return model_names
    raise HTTPException(status_code=404, detail="Unable to access models.")


"""
Token prediction task
"""

@app.post("/prediction", response_model=prediction_schemas.PredictionResponse)
async def prediction_endpoint(request: prediction_schemas.PredictionRequest):
    try:
        # receive model name from front-end and retrieve via model_config file
        frontend_selection = request.model_name
        model_info = next(
            (m for m in MODEL_CONFIG if m["name"] == frontend_selection), None
        )
        model_name = model_info["repo"]
        model_type = model_info["type"]

        # load model from HF Hub and to device
        try:
            model, tokenizer = model_loader.load_encoder(model_name, model_type)
            device, model = model_loader.load_device(model)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Unable to load model.") from e

        # receive input text from front-end, pass to prediction function
        text = request.text
        text = re.sub(r"\?", "[MASK]", text)
        results = predict.prediction_function(
            text,
            model,
            tokenizer,
            device,
            window_size=512,
            overlap=128,
            num_predictions=5,
        )

        # format results for response class
        formatted_results = {}
        for masked_index, predictions in results.items():
            token_predictions = [
                prediction_schemas.TokenPrediction(token=pred[0], probability=pred[1])
                for pred in predictions
            ]
            formatted_results[masked_index] = prediction_schemas.MaskedIndexPredictions(
                predictions=token_predictions
            )
        logging.info(f"Formatted results: {formatted_results}")
        return prediction_schemas.PredictionResponse(predictions=formatted_results)

    except HTTPException as e:
        raise
    except (IndexError, ValueError) as e:
        logging.exception(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail="Invalid input.") from e
    except Exception as e:
        logging.exception(f"Prediction task error: {e}")
        raise HTTPException(status_code=500) from e


"""
Error detection task
"""

@app.post("/detection", response_model=detection_schemas.DetectionResponse)
async def detection_endpoint(request: detection_schemas.DetectionRequest):
    try:
        # receive model name from front-end, retrieve via model_config file
        frontend_selection = request.model_name
        model_info = next(
            (m for m in MODEL_CONFIG if m["name"] == frontend_selection), None
        )
        model_name = model_info["repo"]
        model_type = model_info["type"]
        lev_distance = request.lev_distance

        # load HF model to device
        try:
            model, tokenizer = model_loader.load_encoder(model_name, model_type)
            device, model = model_loader.load_device(model)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Unable to load model.") from e

        # load HF Lev filter per selected lev_distance
        lev_filter_url = LEV_FILTER_URLS.get(f"lev{lev_distance}")

        try:
            lev_filter = load_filter_from_url(lev_filter_url)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Unable to load Levenshtein matrix."
            )

        # create instance of Logion class from selected model and Lev filter
        model = logion_class.Logion(model, tokenizer, lev_filter, device)

        # receive input text from front-end, pass to detection function
        text = request.text
        results, ccr = detect.detection_function(text, model, tokenizer, device)

        # format results for response class
        formatted_results = []
        for (
            original_word,
            chance_score,
            global_word_index,
        ), suggestions in results.items():
            formatted_suggestions = [
                detection_schemas.MaskPrediction(token=sug, probability=prob)
                for sug, prob in suggestions
            ]
            formatted_results.append(
                detection_schemas.WordPrediction(
                    original_word=original_word,
                    chance_score=chance_score,
                    global_word_index=global_word_index,
                    suggestions=formatted_suggestions,
                )
            )
        # account for zero-value CCR scores
        ccr = np.nan_to_num(ccr, nan=100000, posinf=100000, neginf=100000)
        ccr_results = [
            detection_schemas.CCRResult(ccr_value=ccr_value) for ccr_value in ccr
        ]
        logging.info(f"Formatted Results: {formatted_results}")
        logging.info(f"CCR Results: {ccr_results}")
        return detection_schemas.DetectionResponse(
            predictions=formatted_results, ccr=ccr_results
        )

    except HTTPException as e:
        raise
    except (IndexError, ValueError) as e:
        logging.exception(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail="Invalid input.") from e
    except Exception as e:
        logging.exception(f"Detection task error: {e}")
        raise HTTPException(status_code=500) from e


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
