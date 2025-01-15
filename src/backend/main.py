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
import include # necessary to find root folder when running from a dist folder
import uvicorn



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
logging.basicConfig(filename=log_file_path,
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')
logging.info(f"API logging configured at: {log_file_path}")


app = FastAPI(title="Logion", port=8000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)




def load_urls_from_config(config_path):
    """
    Loads urls from config

    Parameters:
        config_path (str) -- path to local url_config.yaml file

    Returns:

    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"URL config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing URL config file: {e}")
        raise

# load the config file
logging.info(f"Current working dir: {os.getcwd()}")
if hasattr(sys, '_MEIPASS'):
    urls_config = os.path.join(sys._MEIPASS, "urls.yaml")
else:
    urls_config = os.path.join(os.path.dirname(__file__), "urls.yaml")
logging.info(f"URL config file path: {urls_config}")
try:
    with open(urls_config, "r") as f:
        url_data = yaml.safe_load(f)
        MODEL_CONFIG_URL = url_data["model_config"]
        LEV_FILTER_URL = url_data["lev_filter"]
        logging.info(f"CONFIG_URL: {MODEL_CONFIG_URL}")
        logging.info(f"LEV_FILTER_URL: {LEV_FILTER_URL}")
except Exception as e:
    logging.error(f"Failed to load URL configuration: {e}")
    raise SystemExit("URL configuration unavailable; exiting application.") from e



def load_config_from_url(url):
    """
    Loads model config file file from url

    Parameters:
        url (str) -- url to Git-hosted model_config.yaml file

    Returns:
        response.text -- text from model_config.yaml
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return yaml.safe_load(response.text)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve model config file via URL: {e}")
    
    # Attempt to load from local file if URL fails
    try:
        if hasattr(sys, '_MEIPASS'):
            local_path = os.path.join(sys._MEIPASS, 'model_config.yaml')
        else:
            local_path = os.path.join(os.path.dirname(__file__), 'model_config.yaml')
        logging.info(f"Attempting to load model_config.yaml from local file: {local_path}")
        with open(local_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Local model config file not found at {local_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing local model config file: {e}")
        raise
        

try:
    model_config = load_config_from_url(MODEL_CONFIG_URL)
except Exception as e:
    logging.error(f"Failed to load model configuration: {e}")
    raise SystemExit("Model configuration unavailable; exiting application.") from e



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
        file_like_object = io.BytesIO(response.content)
        return torch.tensor(np.load(file_like_object))
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve Lev filter file via URL: {e}")
    
    # Attempt to load from local file if URL fails
    try:
        if hasattr(sys, '_MEIPASS'):
            local_path = os.path.join(sys._MEIPASS, 'lev_filter.npy')
        else:
            local_path = os.path.join(os.path.dirname(__file__), 'lev_filter.npy')
        logging.info(f"Attempting to load lev_filter.npy from local file: {local_path}")
        with open(local_path, 'rb') as f:
            return torch.tensor(np.load(f))
    except FileNotFoundError:
        logging.error(f"Local lev filter file not found at {local_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading local lev filter: {e}")
        raise



"""
Server status endpoint
"""
@app.get("/health")
async def health_check():
    return {"status": "ok"}



"""
Token prediction task
"""
@app.post("/prediction", response_model=prediction_schemas.PredictionResponse)
async def prediction_endpoint(request: prediction_schemas.PredictionRequest):
    try:
        # receive model name from front-end and retrieve via model_config file
        frontend_option = request.model_name
        model_info = next((m for m in model_config["models"] if m["name"] == frontend_option), None)
        if not model_info:
            raise ValueError(f"Invalid model selected: '{frontend_option}'. Available models: {[m['name'] for m in model_config['models']]}")
        model_name = model_info["repo"]
        model_type = model_info["type"]

        # load model from HF Hub and to device
        try:
            model, tokenizer = model_loader.load_encoder(model_name, model_type)
            device, model = model_loader.load_device(model)
        except Exception as e:
            logging.exception(f"Error loading model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model {model_name}: {e}") from e
        
        # receive input text from front-end and pass to prediction function
        text = request.text
        results = predict.prediction_function(text, model, tokenizer, device, window_size=512, overlap=128, num_predictions=5)
        
        # format results for response class
        formatted_results = {}
        for masked_index, predictions in results.items():
            token_predictions = [prediction_schemas.TokenPrediction(token=pred[0], probability=pred[1]) for pred in predictions]
            formatted_results[masked_index] = prediction_schemas.MaskedIndexPredictions(predictions=token_predictions)
        logging.info(f"Formatted results: {formatted_results}")
        return prediction_schemas.PredictionResponse(predictions=formatted_results)
    
    except HTTPException as e:
        raise
    except (IndexError, ValueError) as e:
        logging.exception(f"Prediction task error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction error: {e}") from e
    except Exception as e:
        logging.exception(f"Error during prediction task: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction.") from e



"""
Error detection task
"""
@app.post("/detection", response_model=detection_schemas.DetectionResponse)
async def detection_endpoint(request: detection_schemas.DetectionRequest):
    try:
        # receive model name from front-end and retrieve via model_config file
        frontend_option = request.model_name
        model_info = next((m for m in model_config["models"] if m["name"] == frontend_option), None)
        if not model_info:
            raise ValueError(f"Invalid model selected: '{frontend_option}'. Available models: {[m['name'] for m in model_config['models']]}")
        model_name = model_info["repo"]
        model_type = model_info["type"]
        lev_distance = request.lev_distance

        # load model from HF Hub and to device
        try:
            model, tokenizer = model_loader.load_encoder(model_name, model_type)
            device, model = model_loader.load_device(model)
        except Exception as e:
            logging.exception(f"Error loading model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model {model_name}: {e}") from e
        
        # load Lev filter via HF URL per selected lev_distance
        if lev_distance == 1:
            lev_filter_url = LEV_FILTER_URL["lev1"]
        elif lev_distance == 2:
            lev_filter_url = LEV_FILTER_URL["lev2"]
        elif lev_distance == 3:
            lev_filter_url = LEV_FILTER_URL["lev3"]
        else:
            raise ValueError(f"Invalid lev_distance selected: '{lev_distance}'. Available values: [1, 2, 3]")
        try:
            lev_filter = load_filter_from_url(lev_filter_url)
        except Exception as e:
            logging.error(f"Failed to load lev_filter: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load lev_filter: {e}")
        
        # create instance of Logion class from selected model and Lev filter
        model = logion_class.Logion(model, tokenizer, lev_filter, device)

        # receive input text from front-end and pass to detection function
        text = request.text
        results, ccr = detect.detection_function(text, model, tokenizer, device)

        # format results for response class
        formatted_results = []
        for (original_word, chance_score, global_word_index), suggestions in results.items():
            formatted_suggestions = [detection_schemas.MaskPrediction(token=sug, probability=prob) for sug, prob in suggestions]
            formatted_results.append(detection_schemas.WordPrediction(original_word=original_word, chance_score=chance_score, global_word_index=global_word_index, suggestions=formatted_suggestions))
        logging.info(f"{detection_schemas.MaskPrediction}")
        logging.info(f"{detection_schemas.WordPrediction}")
        # account for zero-value CCR scores
        ccr = np.nan_to_num(ccr, nan=100000, posinf=100000, neginf=100000)
        ccr_results = [detection_schemas.CCRResult(ccr_value=ccr_value) for ccr_value in ccr]
        logging.info(f"Formatted Results: {formatted_results}")
        logging.info(f"CCR Results: {ccr_results}")
        logging.info(f"DetectionResponse: {type(detection_schemas.DetectionResponse(predictions=formatted_results, ccr=ccr_results))} {detection_schemas.DetectionResponse(predictions=formatted_results, ccr=ccr_results)}")
        return detection_schemas.DetectionResponse(predictions=formatted_results, ccr=ccr_results)

    except HTTPException as e:
        raise
    except (IndexError, ValueError) as e:
        logging.exception(f"Detection task error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction error: {e}") from e
    except Exception as e:
        logging.exception(f"Error during detection task: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction.") from e

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)