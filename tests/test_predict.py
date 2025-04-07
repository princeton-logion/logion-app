import pytest
import torch
from unittest.mock import patch, MagicMock
from src.backend.prediction import predict
from src.backend.models import model_loader


"""
Integration tests for prediction_function
Uses BERT model from HF
"""


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_type = "bert"
    model_name = "princeton-logion/LOGION-50k_wordpiece"
    model, tokenizer = model_loader.load_encoder(model_name, model_type)
    device, model = model_loader.load_device(model)
    return model, tokenizer, device


test_texts = [
    "",
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
    "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
    "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος." * 40,
    "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν [MASK], καὶ [MASK] ἦν ὁ λόγος.",
]


@pytest.mark.parametrize("text", test_texts)
def test_predict_data(text, model_and_tokenizer):
    """
    Test input-output data behavior for prediction_function

    Asserts:
        1. Empty text returns error
        2. Text w/ no [MASK] returns empty result
        3. Length of output matches number of [MASK]s
        4. Valid output data types
    """
    model, tokenizer, device = model_and_tokenizer

    mask_count = text.count("[MASK]")

    # check for empty input
    if len(text) == 0:
        with pytest.raises(ValueError):
            predict.prediction_function(text, model, tokenizer, device)
        return

    final_predictions_1 = predict.prediction_function(text, model, tokenizer, device)

    # check output data types
    assert isinstance(final_predictions_1, dict)
    assert len(final_predictions_1) == mask_count
    for mask_index, predictions in final_predictions_1.items():
        assert isinstance(mask_index, int)
        assert isinstance(predictions, list)
        assert len(predictions) == 5
        for predicted_token, probability_score in predictions:
            assert isinstance(predicted_token, str)
            assert isinstance(probability_score, float)


@pytest.mark.parametrize("text", test_texts)
def test_predict_reproducibility(text, model_and_tokenizer):
    """
    Reproducibility test for prediction_function

    Asserts:
        1. Length of different prediction instances for same text are identical
        2. Key-value pars of different prediction instances for same text are identical
    """
    model, tokenizer, device = model_and_tokenizer

    # check for empty input
    if len(text) == 0:
        with pytest.raises(ValueError):
            predict.prediction_function(text, model, tokenizer, device)
        return

    final_predictions_1 = predict.prediction_function(text, model, tokenizer, device)
    final_predictions_2 = predict.prediction_function(text, model, tokenizer, device)

    assert len(final_predictions_1) == len(final_predictions_2)

    for (key1, value1), (key2, value2) in zip(
        final_predictions_1.items(), final_predictions_2.items()
    ):
        assert key1 == key2
        assert [x[0] for x in value1] == [x[0] for x in value2]


def test_window_size_and_overlap(model_and_tokenizer):
    """
    Test sliding window for text processing
    Does not use test_texts

    Asserts:
        1. Different sliding window and overlaps return equal number of key-value pairs
        2. Different sliding window and overlaps return identical key-value pairs
    """
    model, tokenizer, device = model_and_tokenizer

    text = (
        "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος." * 40
    )

    mask_count = text.count("[MASK]")

    final_predictions_1 = predict.prediction_function(
        text, model, tokenizer, device, window_size=10, overlap=5
    )
    final_predictions_2 = predict.prediction_function(
        text, model, tokenizer, device, window_size=20, overlap=10
    )

    assert len(final_predictions_1) == mask_count
    assert len(final_predictions_2) == mask_count
    assert set(final_predictions_1.keys()) == set(final_predictions_2.keys())
