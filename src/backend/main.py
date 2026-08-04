import os
import sys
import logging
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models import model_loader, resource_config_loader, concordance_loader #, lev_filter_loader
from utils import prediction_schemas, detection_schemas, ws_schemas
from features import predict, predict_utils, predict_char_auto, detect, logion_class, cancel, formular_concordance, clean_txt
import random
import include
import uvicorn
import regex as re
import asyncio
from typing import Dict, Any, Callable, Coroutine
import pydantic
import socket
import threading
import time
from contextlib import asynccontextmanager


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
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    stream=sys.stderr,
    force=True,
)
# ensure dependencyies out of log
for _noisy in ("urllib3", "asyncio", "filelock", "huggingface_hub", "transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_app_config()
    yield


app = FastAPI(title="Logion", lifespan=lifespan)

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
RESOURCES_CONFIG = os.environ.get("LOGION_RESOURCES_CONFIG")
if not RESOURCES_CONFIG:
    logging.info("LOGION_RESOURCES_CONFIG env variable not set.")
    sys.exit(1)
else:
    logging.info(f"LOGION_RESOURCES_CONFIG env variable: {RESOURCES_CONFIG}")

MODELS_CONFIG = []
LEV_CONFIG_ENTRY = {}
CONCORDANCE_CONFIG_ENTRY = {}
FORMULAR_CONCORDANCE = None
_CONCORDANCE_RESOLVED = False
_CONCORDANCE_LOCK = asyncio.Lock()

async def load_app_config():
    """
    Load resources_config.yaml on app startup.
    If .yaml is ill-formed or inaccessible, exit app.
    """
    global MODELS_CONFIG, LEV_CONFIG_ENTRY, CONCORDANCE_CONFIG_ENTRY
    logging.info("Retrieving resources")
    try:
        config_data = resource_config_loader.load_resource_config(RESOURCES_CONFIG)

        if not config_data:
            logging.info("Unable to access resources_config.yaml")
            raise SystemExit("No resources available. Exiting app.")

        if "models" not in config_data:
            logging.info("Ill-formed resources_config.yaml")
            raise SystemExit("No resources available. Exiting app.")

        MODELS_CONFIG = config_data["models"]
        LEV_CONFIG_ENTRY = config_data.get("lev_filter") or {}
        CONCORDANCE_CONFIG_ENTRY = config_data.get("formular_concordance") or {}
        logging.info(f"Successfully loaded resources from {RESOURCES_CONFIG}")

    except Exception as e:
        logging.info(f"Unable to load external resources at startup: {e}")
        raise SystemExit(f"No resources available. Exiting app.")


async def get_formular_concordance(progress_callback=None):
    """
    Lazily resolve concordance at first hexameter prediction task
    """
    global FORMULAR_CONCORDANCE, _CONCORDANCE_RESOLVED
    if _CONCORDANCE_RESOLVED:
        return FORMULAR_CONCORDANCE
    async with _CONCORDANCE_LOCK:
        if _CONCORDANCE_RESOLVED:
            return FORMULAR_CONCORDANCE
        if not CONCORDANCE_CONFIG_ENTRY:
            _CONCORDANCE_RESOLVED = True
            return None
        loop = asyncio.get_running_loop()
        try:
            if progress_callback:
                await progress_callback(
                    5.0, "Retrieving formular concordance..."
                )
            concordance_path = await loop.run_in_executor(
                None,
                concordance_loader.resolve_concordance_path,
                CONCORDANCE_CONFIG_ENTRY,
            )
            FORMULAR_CONCORDANCE = await loop.run_in_executor(
                None, formular_concordance.FormularConcordance.load, concordance_path
            )
            logging.info(f"Successfully loaded concordance from {concordance_path} (strata: {', '.join(FORMULAR_CONCORDANCE.strata.keys())})")
        except Exception as e:
            FORMULAR_CONCORDANCE = None
            logging.warning(f"Unable to load concordance: {e}.")
        _CONCORDANCE_RESOLVED = True
        return FORMULAR_CONCORDANCE





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

def clean_prediction_token(token: str, model_lang: str) -> str:
    """
    Clean model-specific tokenizer artifacts before sending predictions to frontend.
    LatinBERT uses underscore markers in its tokenizer vocabulary.
    """
    if model_lang == "la":
        return token.rstrip("_")
    return token

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
        request_data (PredictionRequest) -- data from front-end (contains text with '-', model_name, and text_type)
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
            (m for m in MODELS_CONFIG if m["name"] == frontend_selection), None
        )
        if not model_info:
            logging.warning(f"Task {task_id}: Unable to load {frontend_selection}")
            raise HTTPException(status_code=422, detail=f"{frontend_selection} not available.")
        
        # load model from HF Hub to device via config
        await progress_callback(5.0, f"Loading {frontend_selection} model")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        model_name = model_info["model_path"]
        model_type = model_info["type"]
        tokenizer_path = model_info["tokenizer_path"]
        model_lang = model_info["lang"]
        trust_remote_code = model_info.get("trust_remote_code", False)

        text = request_data.text
        text_type = request_data.text_type  

        if model_lang == "la" and text_type == "hexameter":
            raise HTTPException(
                status_code=422,
                detail = "Latin BERT cannot be used with hexameter text type"
            )

        try:
            model, tokenizer = model_loader.load_encoder(
                model_name, 
                model_type, 
                tokenizer_path, 
                model_lang=model_lang,
                trust_remote_code=trust_remote_code)
            device, model = model_loader.load_device(model)
        except Exception as e:
            logging.info(f"Task {task_id}: Unable to load model: {e}")
            raise HTTPException(status_code=500, detail="Unable to load model.") from e

        text = clean_txt.normalize_grc_input(text)
        text_w_mask = re.sub(r"\-", "[MASK]", text)

        await progress_callback(10.0, "Initiating word prediction")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        # lazily resolve concordance
        concordance = None
        if text_type == "hexameter":
            concordance = await get_formular_concordance(progress_callback)

        # formular concordance stratum weights
        formula_weights = None
        target_stratum = getattr(request_data, "formula_target_stratum", None)
        if concordance is not None and target_stratum:
            if target_stratum in concordance.strata:
                all_strata = list(concordance.strata.keys())
                others = [t for t in all_strata if t != target_stratum]
                formula_weights = formular_concordance.stratum_weights(
                    [target_stratum], others, all_strata,
                )
            else:
                logging.warning(
                    f"Task {task_id}: unknown formula stratum"
                    f"'{target_stratum}'; using uniform weights"
                )

        results = await predict.prediction_function(
            text=text_w_mask,
            model=model,
            tokenizer=tokenizer,
            device=device,
            chunk_size=500,
            num_preds=5,
            task_id=task_id,
            progress_callback=progress_callback,
            cancellation_event=cancellation_event,
            text_type=text_type, 
            formular_concordance=concordance,
            formula_stratum_weights=formula_weights,
        )
        if results is None:
             logging.info(f"Task {task_id}: Cannot process text prediction.")
             return None

        await progress_callback(99.0, "Formatting results")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        orig_txt = text

        # format results for response class
        formatted_results = {}
        for masked_index, predictions in results.items():
            token_predictions = [
                prediction_schemas.TokenPrediction(token=clean_prediction_token(pred[0], model_lang), probability=pred[1])
                for pred in predictions
            ]
            formatted_results[masked_index] = prediction_schemas.MaskedIndexPredictions(
                predictions=token_predictions
            )

        # scansion payload for frontend display
        scansion_payload = None
        if text_type == "hexameter":
            raw_scansion = predict_utils.restored_text_scansion(
                text=text_w_mask,
                final_predictions=results,
                tokenizer=tokenizer,
                use_macronizer=False,
            )
            try:
                scansion_payload = [
                    prediction_schemas.ScansionLine(**entry) for entry in raw_scansion
                ]
            except ValidationError:
                logging.exception(
                    f"Task {task_id}: scansion payload failed validation; "
                    "omitting scansion from response"
                )
                scansion_payload = None

        final_response = prediction_schemas.PredictionResponse(
            predictions=formatted_results,
            origText=orig_txt,
            scansion=scansion_payload,
        )
        await progress_callback(100.0, "Word prediction complete.")

        response_dict = final_response.model_dump()
        logging.info(f"Response: {response_dict}")
        
        return response_dict

    except asyncio.CancelledError:
        logging.info(f"Task {task_id}: User cancelled word prediction task.")
        return None
    except Exception as e:
        logging.exception(f"Task {task_id}: Error during word prediction task: {e}")
        raise


async def run_char_prediction_task(
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
        request_data (PredictionRequest) -- data from front-end (contains text with '-' and model_name)
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
            (m for m in MODELS_CONFIG if m["name"] == frontend_selection), None
        )
        if not model_info:
            logging.warning(f"Task {task_id}: Unable to load {frontend_selection}")
            raise HTTPException(status_code=422, detail=f"{frontend_selection} not available.")
        
        # load model from HF Hub to device via config
        await progress_callback(5.0, f"Loading {frontend_selection} model")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        model_name = model_info["model_path"]
        model_type = model_info["type"]
        model_lang = model_info["lang"]

        text = request_data.text
        text_type = request_data.text_type  

        if model_lang == "la" and text_type == "hexameter":
            raise HTTPException(
                status_code=422,
                detail = "Latin BERT cannot be used with hexameter text type"
            )

        try:
            model, char_stoi, char_itos, mask_id = model_loader.load_character_mlm(model_name)
            # model = model_loader.patch_char_model_for_mps(model)
            # device, model = model_loader.load_device(model)
            # use cpu until MPS error resolved
            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            device = torch.device("cpu")
            logging.info(f"Using device {device}.")
            model = model.to(device)
        except Exception as e:
            logging.info(f"Task {task_id}: Unable to load model: {e}")
            raise HTTPException(status_code=500, detail="Unable to load model.") from e

        text = request_data.text
        text = clean_txt.normalize_grc_input(text)
        text_w_mask = re.sub(r"\-", "[MASK]", text)

        await progress_callback(10.0, "Initiating word prediction")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        results = await predict_char_auto.char_prediction_function(
            text=text_w_mask,
            model=model,
            char_stoi=char_stoi,
            char_itos=char_itos,
            mask_id=mask_id,
            device=device,
            chunk_size=2048,
            num_preds=5,
            task_id=task_id,
            progress_callback=progress_callback,
            cancellation_event=cancellation_event
        )
        if results is None:
             logging.info(f"Task {task_id}: Cannot process text prediction.")
             return None

        await progress_callback(99.0, "Formatting results")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        orig_txt = text

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

        final_response = prediction_schemas.PredictionResponse(
            predictions=formatted_results,
            origText=orig_txt,
        )
        await progress_callback(100.0, "Character prediction complete.")

        response_dict = final_response.model_dump()
        logging.info(f"Response: {response_dict}")
        
        return response_dict

    except asyncio.CancelledError:
        logging.info(f"Task {task_id}: User cancelled character prediction task.")
        return None
    except Exception as e:
        logging.exception(f"Task {task_id}: Error during character prediction task: {e}")
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
        - Passes items from DetectionRequest to detection_function()
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
            (m for m in MODELS_CONFIG if m["name"] == frontend_selection), None
        )
        if not model_info:
            logging.warning(f"Task {task_id}: Unable to load {frontend_selection}")
            raise HTTPException(status_code=422, detail=f"{frontend_selection} not available.")
        
        # load model from HF Hub to device via conifg
        await progress_callback(5.0, f"Loading {frontend_selection} model")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        model_name = model_info["model_path"]
        model_type = model_info["type"]
        model_lang = model_info["lang"]
        tokenizer_path = model_info["tokenizer_path"]
        trust_remote_code = model_info.get("trust_remote_code", False)
        lev_distance = request_data.lev_distance

        try:
            model, tokenizer = model_loader.load_encoder(
                model_name, 
                model_type, 
                tokenizer_path, 
                model_lang=model_lang,
                trust_remote_code=trust_remote_code)
            device, model = model_loader.load_device(model)
        except Exception as e:
            logging.info(f"Task {task_id}: Unable to load model: {e}")
            raise HTTPException(status_code=500, detail="Unable to load model.") from e
        
        # load .npy matrix from HF via config
        # await progress_callback(10.0, f"Loading Levenshtein filter {lev_distance}")
        # if await cancel.check_cancel_status(cancellation_event, task_id): return None

        # lev_filter_path = LEV_CONFIG_ENTRY.get(f"lev{lev_distance}")
        # if not lev_filter_path:
        #     logging.info(f"Task {task_id}: Unable to load URL for Lev filter {lev_distance}")
        #     raise HTTPException(status_code=422, detail=f"{lev_distance} not an available Levenshtein distance.")

        # try:
        #     logging.info(f"Lev {lev_distance} URL: {lev_filter_path}")
        #     lev_filter = lev_filter_loader.load_filter(lev_filter_path)
        # except Exception as e:
        #     logging.info(f"Task {task_id}: Unable to load Lev filter: {e}")
        #     raise HTTPException(status_code=500, detail="Unable to load Levenshtein filter.") from e

        # create instance of Logion class
        #logion_model = logion_class.Logion(model, tokenizer, lev_filter, device)
        logion_model = logion_class.Logion(model, tokenizer, device)

        text = request_data.text

        await progress_callback(10.0, "Initiating error detection")
        if await cancel.check_cancel_status(cancellation_event, task_id): return None

        results, ccr = await detect.detection_function(
            text=text,
            model=logion_model,
            tokenizer=tokenizer,
            device=device,
            chunk_size=500,
            lev=lev_distance,
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

        return final_response.model_dump()

    except asyncio.CancelledError:
        logging.info(f"Task {task_id}: User cancelled error detection task")
        return None
    except Exception as e:
        logging.exception(f"Task {task_id}: Error during error detection task: {e}")
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


                if message_type == "start_word_prediction":
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
                        await send_message(
                            websocket, 
                            ws_schemas.ServerErrorMsg(
                                type="error", 
                                task_id=task_id, 
                                detail=f"Invalid request data: {str(e)}",
                                status_code=422
                            ).model_dump()
                        )
                        continue

                    await send_message(websocket, ws_schemas.ServerAckMsg(type="ack", task_id=task_id, message="Word prediction task received").model_dump())

                    cancellation_event = asyncio.Event()

                    async def progress_callback(percentage: float, message: str):
                        await send_message(websocket, ws_schemas.ServerProgressMsg(type="progress", task_id=task_id, percentage=percentage, message=message).model_dump())

                    task = asyncio.create_task(
                        run_prediction_task(request, task_id, progress_callback, cancellation_event)
                    )
                    active_tasks[task_id] = (task, cancellation_event)
                    task.add_done_callback(
                        lambda t: handle_task_completion(t, task_id, websocket)
                    )

                elif message_type == "start_char_prediction":
                    """
                    Execute Character Prediction task
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
                        logging.info(f"Task {task_id}: Invalid character prediction request data: {e}")
                        await send_message(
                            websocket, 
                            ws_schemas.ServerErrorMsg(
                                type="error", 
                                task_id=task_id, 
                                detail=f"Invalid request data: {str(e)}",
                                status_code=422
                            ).model_dump()
                        )
                        continue

                    await send_message(websocket, ws_schemas.ServerAckMsg(type="ack", task_id=task_id, message="Character prediction task received").model_dump())

                    cancellation_event = asyncio.Event()

                    async def progress_callback(percentage: float, message: str):
                        await send_message(websocket, ws_schemas.ServerProgressMsg(type="progress", task_id=task_id, percentage=percentage, message=message).model_dump())

                    task = asyncio.create_task(
                        run_char_prediction_task(request, task_id, progress_callback, cancellation_event)
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
                        await send_message(
                            websocket, 
                            ws_schemas.ServerErrorMsg(
                                type="error", 
                                task_id=task_id, 
                                detail=f"Invalid request data: {str(e)}",
                                status_code=422
                            ).model_dump()
                        )
                        continue

                    await send_message(websocket, ws_schemas.ServerAckMsg(type="ack", task_id=task_id, message="Error detection task received").model_dump())

                    cancellation_event = asyncio.Event()

                    async def progress_callback(percentage: float, message: str):
                        await send_message(websocket, ws_schemas.ServerProgressMsg(type="progress", task_id=task_id, percentage=percentage, message=message).model_dump())

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
                

                # UNK msgs handled by pydantic, but fallback
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
    # clean when WebSocket disconnects
    if task_id in active_tasks:
        del active_tasks[task_id]
        logging.info(f"Task {task_id}: Removed from active tasks")

    try:
        if task.cancelled():
            logging.info(f"Task {task_id}: Cancelled")
            asyncio.create_task(send_message(websocket, ws_schemas.ServerCancelMsg(type="cancelled", task_id=task_id).model_dump()))

        elif task.exception():
            exc = task.exception()
            logging.info(f"Task {task_id} error: {exc}")
            error_detail = f"Task {task_id} failed: {exc}"
            status_code = 500

            if isinstance(exc, HTTPException):
                 error_detail = exc.detail
                 status_code = exc.status_code
            asyncio.create_task(
                send_message(
                    websocket, 
                    ws_schemas.ServerErrorMsg(
                        type="error", 
                        task_id=task_id, 
                        detail=error_detail, 
                        status_code=status_code
                    ).model_dump()
                )
            )

        else:
            result = task.result()
            # None = Cancelled
            if result is None:
                 logging.info(f"Task {task_id}: Cancelled")
                 asyncio.create_task(send_message(websocket, ws_schemas.ServerCancelMsg(type="cancelled", task_id=task_id).model_dump()))
            else:
                 logging.info(f"Task {task_id}: Completed")
                 asyncio.create_task(send_message(websocket, ws_schemas.ServerResultMsg(type="result", task_id=task_id, data=result).model_dump()))

    except Exception as e:
        logging.info(f"Task handler error for {task_id}: {e}")
        asyncio.create_task(send_message(websocket, ws_schemas.ServerErrorMsg(type="error", task_id=task_id, detail=f"Task handler internal error", status_code=500).model_dump()))



"""
Server status endpoint
"""
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_connections": len(active_connections), "active_tasks": len(active_tasks)}


"""
Model retrieval endpoint
"""
@app.get("/models")
async def get_models():
    """
    Returns list of available model names loaded at startup.
    To load list from updated resources_config.yaml, restart app.
    """
    if not MODELS_CONFIG:
        logging.info("Unable to access MODELS_CONFIG in /models endpoint.")
        raise HTTPException(
            status_code=503,
            detail="Model configurations currently unavailable."
        )
    try:
        models = [
            {
                "name": model["name"],
                "type": model["type"],
                "lang": model["lang"]
            }
            for model in MODELS_CONFIG
        ]
        return models
    except Exception as e:
        logging.info(f"Unexpected error in /models endpoint: {e}")
        raise HTTPException(status_code=500, detail="Unable to access models.")
    

class CachedStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        if "text/html" in (resp.media_type or ""):
            resp.headers["Cache-Control"] = "no-cache"
        else:
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp
# pull static dir from dir and log for debugging
frontend_build_dir = os.environ.get("STATIC_DIR")
if not frontend_build_dir or not os.path.isdir(frontend_build_dir):
    logging.info(f"Unable to find static at {frontend_build_dir}")
    sys.exit(1)
logging.info(f"Serving static from {frontend_build_dir}")
# mount static
app.mount("/", CachedStaticFiles(directory=frontend_build_dir, html=True), name="static")

def exeunt_with_electron(server: uvicorn.Server) -> None:
    """
    Shut down when Electron parent disappears
    Avoids lingering vnicorn run on force-quit
    Ignored w/ Singularity/Apptainer
    """
    if os.environ.get("LOGION_LAUNCHER") != "electron":
        logging.info("Not launched by Electron; stdin exit detection disabled.")
        return

    if sys.stdin is None or sys.stdin.closed:
        logging.info("stdin pipe unavailable. Server may continue after app closes.")
        return

    def _wait() -> None:
        try:
            while sys.stdin.readline():
                pass
        except Exception:
            pass
        logging.info("Electron process closed. Shutting down.")
        server.should_exit = True
        time.sleep(10)
        os._exit(1)

    threading.Thread(target=_wait, daemon=True).start()


if __name__ == "__main__":
    host = os.environ.get("LOGION_HOST") or "127.0.0.1"
    # assign free port if 8000 taken
    requested_port = int(os.environ.get("LOGION_PORT") or 0)
    # bind socket to prevent other process claiming port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, requested_port))
    actual_port = sock.getsockname()[1]

    print(f"__LOGION_PORT__={actual_port}", flush=True)
    logging.info(f"Spawning Logion server on http://{host}:{actual_port}")

    config = uvicorn.Config(app, timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    exeunt_with_electron(server)
    server.run(sockets=[sock])