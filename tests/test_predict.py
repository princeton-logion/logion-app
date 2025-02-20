import pytest
from unittest.mock import patch
from ..src.backend.prediction import predict
from ..src.backend.models import model_loader


"""
Test for prediction_function given different inputs
"""


"""
TEST TXT:
Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.
"""

@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_type = 'bert'
    model_name = "princeton-logion/LOGION-50k_wordpiece" 
    model, tokenizer = model_loader.load_encoder(model_name, model_type)
    model = model_loader.load_device(model) 
    return model, tokenizer



def test_empty_text(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = ""
    with pytest.raises(ValueError):
      predictions = predict.prediction_function(text, model, tokenizer)



def test_short_text_no_mask(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
    predictions = predict.prediction_function(text, model, tokenizer)
    assert predictions == {}


    
def test_short_text_with_mask(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Ἐν [MASK] ἦν ὁ λόγος"
    predictions = predict. prediction_function(text, model, tokenizer)
    assert isinstance(predictions, dict)
    assert len(predictions) == 1
    for mask_index, preds in predictions.items():
      assert isinstance(preds, list)
      assert len(preds) == 5
      for word, prob in preds:
        assert isinstance(word, str)
        assert isinstance(prob, float)



def test_long_text_with_mask(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος." * 200
    predictions = predict.prediction_function(text, model, tokenizer, window_size=100, overlap=20)
    assert isinstance(predictions, dict)
    assert len(predictions) > 0
    for mask_index, preds in predictions.items():
      assert isinstance(preds, list)
      assert len(preds) == 5
      for word, prob in preds:
        assert isinstance(word, str)
        assert isinstance(prob, float)



def test_text_with_multiple_masks(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν [MASK], καὶ [MASK] ἦν ὁ λόγος."
    predictions = predict.prediction_function(text, model, tokenizer)
    assert isinstance(predictions, dict)
    assert len(predictions) == 3
    for mask_index, preds in predictions.items():
        assert isinstance(preds, list)
        assert len(preds) == 5
        for word, prob in preds:
          assert isinstance(word, str)
          assert isinstance(prob, float)



def test_window_size_and_overlap(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος." * 10
    predictions1 = predict.prediction_function(text, model, tokenizer, window_size=10, overlap=2)
    predictions2 = predict.prediction_function(text, model, tokenizer, window_size=20, overlap=5)
    assert isinstance(predictions1, dict)
    assert isinstance(predictions2, dict)
    assert len(predictions1) > 0
    assert len(predictions2) > 0
    assert set(predictions1.keys()) == set(predictions2.keys())



def test_num_predictions(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Ἐν ἀρχῇ ἦν ὁ [MASK], καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος."
    predictions_3 = predict.prediction_function(text, model, tokenizer, num_predictions=3)
    predictions_7 = predict.prediction_function(text, model, tokenizer, num_predictions=7)
    assert isinstance(predictions_3, dict)
    assert isinstance(predictions_7, dict)
    for mask_index, preds in predictions_3.items():
      assert len(preds) == 3
    for mask_index, preds in predictions_7.items():
      assert len(preds) == 7

      
    
def test_no_special_characters_in_text(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "ΕΡΤΥΘΙΟΠΛΚΞΗΓΦΔΣΑΖΧΨΩΒΝΜ [MASK] ερτυθιοπλκξηγφδςσαζχψωβνμ" 
    predictions = predict.prediction_function(text, model, tokenizer)
    assert isinstance(predictions, dict)
    for mask_index, preds in predictions.items():
        assert isinstance(preds, list)
        assert len(preds) == 5
        for word, prob in preds:
          assert isinstance(word, str)
          assert isinstance(prob, float)