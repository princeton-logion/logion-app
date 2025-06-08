import pytest
import torch
from src.backend.features import predict
from src.backend.models import model_loader
import asyncio
from typing import Any, Callable, Coroutine

"""
Universal variables
"""
WINDOW_SIZE = 512
WINDOW_OVERLAP = 128
NUM_PREDS = 5
TASK_ID = "05a7a1f9-6071-4467-9af2-52165657a685"

test_txts = [
    "",
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
    "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
    "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος." * 40,
    "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν [MASK], καὶ [MASK] ἦν ὁ λόγος.",
]

# progress callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]
async def progress_callback(progress: float, message: str) -> None:
    await asyncio.sleep(0)

# load model
@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_type = "bert"
    model_name = "princeton-logion/logion-bert-base"
    model, tokenizer_obj = model_loader.load_encoder(model_name, model_type)
    device, model = model_loader.load_device(model)
    return model, tokenizer_obj, device


"""
Integration tests for prediction_function
"""

@pytest.mark.asyncio
@pytest.mark.parametrize("text", test_txts)
async def test_predict_data(text: str, model_and_tokenizer):
    """
    Test input-output data behavior for prediction_function

    Asserts:
        1. Empty text returns empty dict
        2. Text w/ no [MASK] returns empty result
        3. Length of output matches number of [MASK]s
        4. Valid output data types
    """
    model, tokenizer, device = model_and_tokenizer
    cancellation_event = asyncio.Event()
    mask_count = text.count("[MASK]")

    final_predictions = await predict.prediction_function(
        text=text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=WINDOW_SIZE,
        overlap=WINDOW_OVERLAP,
        num_predictions=NUM_PREDS,
        task_id=TASK_ID,
        progress_callback=progress_callback,
        cancellation_event=cancellation_event
    )

    # check output data types
    assert isinstance(final_predictions, dict)
    assert len(final_predictions) == mask_count
    for mask_global_index, predictions_list in final_predictions.items():
        assert isinstance(mask_global_index, int)
        assert isinstance(predictions_list, list)
        assert len(predictions_list) == NUM_PREDS
        for predicted_token, probability_score in predictions_list:
            assert isinstance(predicted_token, str)
            assert isinstance(probability_score, float)

@pytest.mark.asyncio 
@pytest.mark.parametrize("text", test_txts)
async def test_predict_reproducibility(text: str, model_and_tokenizer):
    """
    Reproducibility test for prediction_function

    Asserts:
        1. Length of different prediction instances for same text are identical
        2. Key-value pairs of different prediction instances for same text are identical
    """
    model, tokenizer, device = model_and_tokenizer
    cancellation_event_1 = asyncio.Event()
    cancellation_event_2 = asyncio.Event()

    final_predictions_1 = await predict.prediction_function(
        text=text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=WINDOW_SIZE,
        overlap=WINDOW_OVERLAP,
        num_predictions=NUM_PREDS,
        task_id=TASK_ID + "_iter1",
        progress_callback=progress_callback,
        cancellation_event=cancellation_event_1
    )

    final_predictions_2 = await predict.prediction_function(
        text=text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=WINDOW_SIZE,
        overlap=WINDOW_OVERLAP,
        num_predictions=NUM_PREDS,
        task_id=TASK_ID + "_iter2",
        progress_callback=progress_callback,
        cancellation_event=cancellation_event_2
    )

    assert len(final_predictions_1) == len(final_predictions_2)

    items1 = sorted(final_predictions_1.items())
    items2 = sorted(final_predictions_2.items())
    assert items1 == items2

@pytest.mark.asyncio
async def test_window_size_and_overlap(model_and_tokenizer):
    """
    Test sliding window for text processing
    Test window/overlap settings return identical predictions same MASK text
    Does not use test_txts

    Asserts:
        1. Different sliding window and overlaps return equal number of key-value pairs
        2. Different sliding window and overlaps return identical key-value pairs
    """
    model, tokenizer, device = model_and_tokenizer
    cancellation_event_1 = asyncio.Event()
    cancellation_event_2 = asyncio.Event()

    large_txt = (
        "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος." * 40
    )
    mask_count = large_txt.count("[MASK]")

    final_predictions_1 = await predict.prediction_function(
        text=large_txt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=10,
        overlap=5,
        num_predictions=NUM_PREDS,
        task_id=TASK_ID + "_window1",
        progress_callback=progress_callback,
        cancellation_event=cancellation_event_1
    )
    final_predictions_2 = await predict.prediction_function(
        text=large_txt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=20,
        overlap=10,
        num_predictions=NUM_PREDS,
        task_id=TASK_ID + "_window2",
        progress_callback=progress_callback,
        cancellation_event=cancellation_event_2
    )

    assert len(final_predictions_1) == mask_count
    assert len(final_predictions_2) == mask_count
    assert set(final_predictions_1.keys()) == set(final_predictions_2.keys())