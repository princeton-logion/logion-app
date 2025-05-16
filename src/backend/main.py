import os
import sys
import logging
import requests
import io
import yaml
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from models import model_loader
from utils import prediction_schemas, detection_schemas, ws_schemas
from features import predict, detect, logion_class, cancel
import random
import include
import uvicorn
import regex as re
import asyncio
from typing import Dict, Any, Callable, Coroutine
import pydantic


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
    level="INFO",
    format="%(asctime)s - %(filename)s - %(lineno)d - %(message)s",
)

app = FastAPI(title="Logion", port=8000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# store active WebSocket connections: {user_id: WebSocket}
active_connections: Dict[str, WebSocket] = {}
# store current tasks: {task_id: (asyncio.Task, asyncio.Event)}
active_tasks: Dict[str, tuple[asyncio.Task, asyncio.Event]] = {}


"""
Remote resource handling
"""
RESOURCES_URL = "https://raw.githubusercontent.com/princeton-logion/logion-app/main/src/backend/resources_config.yaml"
MODEL_CONFIG = []
LEV_FILTER_URLS = {}

def resource_loader(url: str) -> Dict:
    """
    Loads model config .yaml file file from url

    Parameters:
        url (str) -- url to Git-hosted resources_config.yaml

    Returns:
        response.text -- url string from resources_config.yaml
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return yaml.safe_load(response.text)
    except requests.exceptions.RequestException as e:
        logging.info(f"Unable to load remote resources_config.yaml: {e}. Trying local.")
    # if URL fails, load local .yaml
    try:
        local_path = os.path.join(
            getattr(sys, "_MEIPASS", os.path.dirname(__file__)),
            "resources_config.yaml"
        )
        with open(local_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e_local:
        logging.info(f"Unable to load local resources_config.yaml: {e_local}")
        raise


@app.on_event("startup")
async def load_app_config():
    """
    Load resources_config.yaml on app startup.
    If .yaml is ill-formed or inaccessible, exit app.
    """
    global MODEL_CONFIG, LEV_FILTER_URLS
    logging.info("Retrieving resources")
    try:
        config_data = resource_loader(RESOURCES_URL)

        if not config_data:
            logging.info("Unable to access resources_config.yaml")
            raise SystemExit("No resources available. Exiting app.")

        if "models" and "lev_filter" not in config_data:
            logging.info("Ill-formed resources_config.yaml")
            raise SystemExit("No resources . Exiting app.")

        MODEL_CONFIG = config_data["models"]
        LEV_FILTER_URLS = config_data["lev_filter"]
        logging.info("Successfully loaded resources via config")

    except Exception as e:
        logging.info(f"Unable to load app resources at startup: {e}")
        raise SystemExit(f"No resources available. Exiting app.")


def filter_loader(url: str) -> torch.tensor:
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
        logging.info(f"Loaded Lev filter from URL: {url}")
        return torch.tensor(np.load(lev_matrix_file))
    except requests.exceptions.RequestException as e:
        logging.info(f"Unable to load remote Lev filter from {url}: {e}")
    # if URL fails, load local file
    try:
        if hasattr(sys, "_MEIPASS"):
            local_path = os.path.join(sys._MEIPASS, "lev_filter.npy")
        else:
            local_path = os.path.join(os.path.dirname(__file__), "lev_filter.npy")
        with open(local_path, "rb") as file:
            return torch.tensor(np.load(file))
    except Exception as e:
        logging.info(f"Unable to load local Lev filter: {e}")
        raise


async def send_message(websocket: WebSocket, message: Dict[str, Any]) -> None:
    """
    Send JSON message between server-user via WebSocket connection

    Parameters:
        websocket (WebSocket) -- WebSocket connection for msg
        message (Dict[str, Any]) -- msg to sent
    """
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logging.info(f"Server disconnected before Websocket message sent: {message.get('task_id', 'N/A')}")
    except Exception as e:
        logging.info(f"Unable to send WebSocket message: {e} for task {message.get('task_id', 'N/A')}")


ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]


async def run_prediction_task(
    request_data: prediction_schemas.PredictionRequest,
    task_id: str,
    progress_callback: ProgressCallback,
    cancellation_event: asyncio.Event
) -> Dict[str, Any]:
    """
    Runs word prediction task asynchronously. Steps include:
        - Loads model/tokenizer
        - Passese items from PredictionRequest to prediction_function()
        - Format prediction_function() results for front-end
    
    Parameters:
        request_data (PredictionRequest) -- data from front-end (contains text with '?' and model_name)
        task_id (str) -- task UID
        progress_callback (ProgressCallback) -- callback function to report progress
        cancellation_event (asyncio.Event) -- event object signaling task cancellation
    
    Returns:
        Dict[str, Any] -- dictionary containing formatted word prediction results
    """
    try:
        # receive model name from front-end and retrieve via model_config file
        frontend_selection = request_data.model_name
        model_info = next(
            (m for m in MODEL_CONFIG if m["name"] == frontend_selection), None
        )
        if not model_info:
             raise ValueError(f"{frontend_selection} model unavailable.")
        
        # load model from HF Hub to device via config
        await progress_callback(5.0, f"Loading {frontend_selection} model")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        model_name = model_info["repo"]
        model_type = model_info["type"]

        try:
            model, tokenizer = model_loader.load_encoder(model_name, model_type)
            device, model = model_loader.load_device(model)
        except Exception as e:
            logging.info(f"Task {task_id}: Unable to load model: {e}")
            raise HTTPException(status_code=500, detail="Unable to load model.") from e

        text = request_data.text
        text = re.sub(r"\?", "[MASK]", text)

        await progress_callback(10.0, "Initiating word prediction")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        results = await predict.prediction_function(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            window_size=512,
            overlap=128,
            num_predictions=5,
            task_id=task_id,
            progress_callback=progress_callback,
            cancellation_event=cancellation_event
        )
        if results is None:
             logging.info(f"Task {task_id}: Cannot process text prediction.")
             return None

        await progress_callback(98.0, "Formatting results")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

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

        final_response = prediction_schemas.PredictionResponse(predictions=formatted_results)
        await progress_callback(100.0, "Word prediction complete.")

        return final_response.dict()

    except asyncio.CancelledError:
        logging.info(f"Task {task_id}: User cancelled word prediction task.")
        return None
    except Exception as e:
        logging.info(f"Task {task_id}: Error during word prediction task: {e}")
        raise


async def run_detection_task(
    request_data: detection_schemas.DetectionRequest,
    task_id: str,
    progress_callback: ProgressCallback,
    cancellation_event: asyncio.Event
) -> Dict[str, Any]:
    """
    Runs error detection task asynchronously. Steps include:
        - Loads model/tokenizer
        - Loads Lev filter matrix
        - Creates Logion class from resources
        - Passese items from DetectionRequest to detection_function()
        - Format detection_function() results for front-end
        - Account for zero-value CCR scores
    
    Parameters:
        request_data (DetectionRequest) -- data from front-end (contains text, model name, and Lev dist)
        task_id (str) -- task UID
        progress_callback (ProgressCallback) -- callback function to report progress
        cancellation_event (asyncio.Event) -- event object signaling task cancellation
    
    Returns:
        Dict[str, Any] -- dict containing formatted error detection results with chance-confidence ratios
    """
    try:
        # receive model name from front-end and retrieve via model_config file
        frontend_selection = request_data.model_name
        model_info = next(
            (m for m in MODEL_CONFIG if m["name"] == frontend_selection), None
        )
        if not model_info:
            raise ValueError(f"{frontend_selection} model not available.")
        
        # load model from HF Hub to device via conifg
        await progress_callback(5.0, f"Loading {frontend_selection} model")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        model_name = model_info["repo"]
        model_type = model_info["type"]
        lev_distance = request_data.lev_distance

        try:
            model, tokenizer = model_loader.load_encoder(model_name, model_type)
            device, model = model_loader.load_device(model)
        except Exception as e:
            logging.info(f"Task {task_id}: Unable to load model: {e}")
            raise HTTPException(status_code=500, detail="Unable to load model.") from e
        
        # load .npy matrix from HF via config
        await progress_callback(10.0, f"Loading Levenshtein filter {lev_distance}")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        lev_filter_url = LEV_FILTER_URLS.get(f"lev{lev_distance}")
        if not lev_filter_url:
            raise ValueError(f"Unable to find URL for Lev filter {lev_distance}.")

        try:
            lev_filter = filter_loader(lev_filter_url)
        except Exception as e:
            logging.info(f"Task {task_id}: Unable to load Lev filter: {e}")
            raise HTTPException(status_code=500, detail="Unable to load Levenshtein filter.") from e

        # create instance of Logion class
        logion_model = logion_class.Logion(model, tokenizer, lev_filter, device)

        text = request_data.text

        await progress_callback(10.0, "Initiating error detection")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        results, ccr = await detect.detection_function(
            text=text,
            model=logion_model,
            tokenizer=tokenizer,
            device=device,
            chunk_size=500,
            lev=1,
            no_beam=False,
            task_id=task_id,
            progress_callback=progress_callback,
            cancellation_event=cancellation_event
        )
        if results is None:
             logging.info(f"Task {task_id}: Cannot process error detection.")
             return None

        await progress_callback(98.0, "Formatting results")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

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

        final_response = detection_schemas.DetectionResponse(
            predictions=formatted_results, ccr=ccr_results
        )
        await progress_callback(100.0, "Error detection complete")

        return final_response.dict()

    except asyncio.CancelledError:
        logging.info(f"Task {task_id}: User cancelled error detection task")
        return None
    except Exception as e:
        logging.info(f"Task {task_id}: Error during error detection: {e}")
        raise


"""
WebSocket endpoint
"""
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    active_connections[user_id] = websocket
    logging.info(f"User {user_id}: Connected to WebSocket")

    try:
        while True:
            try:
                data = await websocket.receive_json()
                logging.info(f"User {user_id}: Received message: {data}")

                message_type = data.get("type")
                task_id = data.get("task_id")


                if message_type == "start_prediction":
                    """
                    Execute Word Prediction task
                    """
                    # check active tasks first
                    if task_id in active_tasks:
                         await send_message(websocket, {"type": "error", "task_id": task_id, "detail": "Task ID already exists"})
                         continue

                    request_data = data.get("request_data", {})
                    try:
                        # validate against pydantic schema
                        request = prediction_schemas.PredictionRequest(**request_data)
                    except pydantic.ValidationError as e:
                         logging.info(f"Task {task_id}: Invalid word prediction request data: {e}")
                         await send_message(websocket, {"type": "error", "task_id": task_id, "detail": f"Invalid request_data: {e}"})
                         continue

                    await send_message(websocket, ws_schemas.ServerAckMsg(type="ack", task_id=task_id, message="Word prediction task received").dict())

                    cancellation_event = asyncio.Event()

                    async def progress_callback(percentage: float, message: str):
                        await send_message(websocket, ws_schemas.ServerProgressMsg(type="progress", task_id=task_id, percentage=percentage, message=message).dict())

                    task = asyncio.create_task(
                        run_prediction_task(request, task_id, progress_callback, cancellation_event)
                    )
                    active_tasks[task_id] = (task, cancellation_event)
                    task.add_done_callback(
                        lambda t: handle_task_completion(t, task_id, websocket)
                    )


                elif message_type == "start_detection":
                    """
                    Execute Error Detection task
                    """
                    # check active tasks first
                    if task_id in active_tasks:
                         await send_message(websocket, {"type": "error", "task_id": task_id, "detail": "Task ID already exists"})
                         continue

                    request_data = data.get("request_data", {})
                    try:
                        # validate against pydantic schema
                        request = detection_schemas.DetectionRequest(**request_data)
                    except pydantic.ValidationError as e:
                         logging.info(f"Task {task_id}: Invalid error detection request data: {e}")
                         await send_message(websocket, {"type": "error", "task_id": task_id, "detail": f"Invalid request_data: {e}"})
                         continue

                    await send_message(websocket, ws_schemas.ServerAckMsg(type="ack", task_id=task_id, message="Error detection task received").dict())

                    cancellation_event = asyncio.Event()

                    async def progress_callback(percentage: float, message: str):
                         await send_message(websocket, ws_schemas.ServerProgressMsg(type="progress", task_id=task_id, percentage=percentage, message=message).dict())

                    task = asyncio.create_task(
                        run_detection_task(request, task_id, progress_callback, cancellation_event)
                    )
                    active_tasks[task_id] = (task, cancellation_event)
                    task.add_done_callback(
                        lambda t: handle_task_completion(t, task_id, websocket)
                    )


                elif message_type == "cancel_task":
                    """
                    Cancel tasks
                    """
                    if task_id in active_tasks:
                        task, cancellation_event = active_tasks[task_id]
                        if not task.done():
                            logging.info(f"User {user_id}: Request to cancel task {task_id}")
                            cancellation_event.set()
                        else:
                            logging.info(f"User {user_id}: Unable to cancel already completed task {task_id}")
                    else:
                        logging.info(f"User {user_id}: Unable to cancel unknown task {task_id}")
                        await send_message(websocket, {"type": "error", "task_id": task_id, "detail": "Task ID unknown or already complete."})
                

                # UNK msgs handled by pydantic, but in case
                else:
                    logging.info(f"User {user_id}: Unknown message type: {message_type}")
                    await send_message(websocket, {"type": "error", "task_id": task_id, "detail": f"Unknown message type: {message_type}"})


            except WebSocketDisconnect:
                logging.info(f"User {user_id}: WebSocket disconnected")
                # cancel all User tasks
                tasks_to_cancel = [task_id for task_id, (task, _) in active_tasks.items() if active_connections.get(user_id) == websocket]
                for task_id in tasks_to_cancel:
                     if task_id in active_tasks:
                         task, cancellation_event = active_tasks[task_id]
                         if not task.done():
                             logging.info(f"User {user_id} disconnected, cancelling task {task_id}")
                             cancellation_event.set()
                break 
            except Exception as e:
                logging.info(f"User {user_id}: Error processing WebSocket message: {e}")
                await send_message(websocket, {"type": "error", "task_id": "N/A", "detail": f"Internal server error: {e}"})
    finally:
        # clean connection
        if user_id in active_connections:
            del active_connections[user_id]
        logging.info(f"User {user_id}: Cleaned connection")


def handle_task_completion(task: asyncio.Task, task_id: str, websocket: WebSocket):
    """
    
    """
     # clean even though WebSocket disconnects
    if task_id in active_tasks:
        del active_tasks[task_id]
        logging.info(f"Task {task_id}: Removed from active tasks")

    try:
        if task.cancelled():
            logging.info(f"Task {task_id}: Cancelled")
            asyncio.create_task(send_message(websocket, ws_schemas.ServerCancelMsg(type="cancelled", task_id=task_id).dict()))

        elif task.exception():
            exc = task.exception()
            logging.info(f"Task {task_id}: Error: {exc}")
            error_detail = f"Task failed: {exc}"
            if isinstance(exc, HTTPException):
                 error_detail = exc.detail
            asyncio.create_task(send_message(websocket, ws_schemas.ServerErrorMsg(type="error", task_id=task_id, detail=error_detail).dict()))

        else:
            result = task.result()
            # None = Cancelled
            if result is None:
                 logging.info(f"Task {task_id}: Cancelled")
                 asyncio.create_task(send_message(websocket, ws_schemas.ServerCancelMsg(type="cancelled", task_id=task_id).dict()))
            else:
                 logging.info(f"Task {task_id}: Completed")
                 asyncio.create_task(send_message(websocket, ws_schemas.ServerResultMsg(type="result", task_id=task_id, data=result).dict()))

    except Exception as e:
        logging.info(f"Task {task_id}: Error in task handler: {e}")
        asyncio.create_task(send_message(websocket, ws_schemas.ServerErrorMsg(type="error", task_id=task_id, detail=f"Error handling task").dict()))



"""
Server status endpoint
"""
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_connections": len(active_connections), "active_tasks": len(active_tasks)}


"""
Model retrieval endpoint
"""
@app.get("/models", response_model=list[str])
async def get_models():
    """
    Returns list of available model names loaded at startup.
    To load list from updated resources_config.yaml, restart app.
    """
    if not MODEL_CONFIG:
        logging.info("Unable to access MODEL_CONFIG in /models endpoint.")
        raise HTTPException(
            status_code=503,
            detail="Model configurations not available."
        )
    try:
        model_names = [model["name"] for model in MODEL_CONFIG]
        return model_names
    except Exception as e:
        logging.info(f"Unexpected error in /models endpoint: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during model list retrieval.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)