import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from ..detection import detect
from ..detection import logion_class


"""
Test for detection_function given different inputs
"""


@pytest.fixture(scope="module")
def mock_model_and_tokenizer():

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_lev_filter = MagicMock()

    # mock tokenizer return value
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1,2,3]]), "attention_mask": torch.tensor([[1,1,1]])}
    # mock model return value, add last_hidden_state
    mock_model.return_value = MagicMock(last_hidden_state = torch.rand(1,3,768))
    # mock Tokenizer attribute
    mock_model.Tokenizer = MagicMock()
    # mock cls_token_id attribute as predefined value
    mock_model.Tokenizer.cls_token_id = 101  # 101 == [CLS]

    mock_lev_filter.data = torch.rand(100,100)
    mock_model.device = 'cpu'
    logion_model = logion_class.Logion(mock_model, mock_tokenizer, mock_lev_filter, mock_model.device)

    return logion_model, mock_tokenizer



test_texts = [
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
    "Ἐν ἀρχῇ ἦν ὁ λόγος",
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος." * 40,
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος" * 40,
    ""
]



@pytest.mark.parametrize("text", test_texts)
def test_detection_function(text, mock_model_and_tokenizer):
    logion_model, tokenizer = mock_model_and_tokenizer

    if len(text) == 0:
        with pytest.raises(ValueError):
            detect.detection_function(text, logion_model, tokenizer)
        return

    predictions, ccr = detect.detection_function(text, logion_model, tokenizer)

    assert isinstance(predictions, dict)
    for (original_word, chance_score, global_word_index), suggestions in predictions.items():
        assert isinstance(original_word, str)
        assert isinstance(chance_score, torch.Tensor) or isinstance(chance_score, float)
        assert isinstance(global_word_index, int)
        assert isinstance(suggestions, list)
        for suggestion, confidence_score in suggestions:
            assert isinstance(suggestion, str)
            assert isinstance(confidence_score, float)

    assert isinstance(ccr, list)
    for item in ccr:
      assert isinstance(item, float)



@pytest.mark.parametrize("text", test_texts)
def test_detection_function_no_beam(text, mock_model_and_tokenizer):
    logion_model, tokenizer = mock_model_and_tokenizer

    if len(text) == 0:
        with pytest.raises(ValueError):
            detect.detection_function(text, logion_model, tokenizer)
        return
    
    predictions, _ = detect.detection_function(text, logion_model, tokenizer, no_beam=True)

    assert isinstance(predictions, dict)
    for (original_word, chance_score, global_word_index), suggestions in predictions.items():
        assert isinstance(original_word, str)
        assert isinstance(chance_score, torch.Tensor) or isinstance(chance_score, float)
        assert isinstance(global_word_index, int)
        assert isinstance(suggestions, list)
        for suggestion, confidence_score in suggestions:
            assert isinstance(suggestion, str)
            assert isinstance(confidence_score, float)



def test_detection_function_reproducibility(mock_model_and_tokenizer):
  logion_model, tokenizer = mock_model_and_tokenizer
  text = "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος."
  predictions_1, ccr_1 = detect.detection_function(text, logion_model, tokenizer)
  predictions_2, ccr_2 = detect.detection_function(text, logion_model, tokenizer)

  assert len(predictions_1) == len(predictions_2)
  assert len(ccr_1) == len(ccr_2)

  for (key1, value1), (key2, value2) in zip(predictions_1.items(), predictions_2.items()):
     assert key1 == key2
     assert [x[0] for x in value1] == [x[0] for x in value2]

  for value1, value2 in zip(ccr_1, ccr_2):
     assert value1 == value2