import pytest
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    ElectraTokenizer,
    ElectraForMaskedLM,
)
from unittest.mock import patch
from src.backend.models import model_loader


"""
load_encoder tests
"""


def test_load_encoder_bert():
    model, tokenizer = model_loader.load_encoder(
        "princeton-logion/logion-bert-base", "bert"
    )
    assert isinstance(model, BertForMaskedLM)
    assert isinstance(tokenizer, BertTokenizer)


def test_load_encoder_electra():
    model, tokenizer = model_loader.load_encoder(
        "princeton-logion/logion-electra-base", "electra"
    )
    assert isinstance(model, ElectraForMaskedLM)
    assert isinstance(tokenizer, ElectraTokenizer)


def test_load_encoder_invalid_model():
    with pytest.raises(ValueError):
        model_loader.load_encoder("bert-base-uncased", "bert")


def test_load_encoder_invalid_model_type():
    with pytest.raises(ValueError):
        model_loader.load_encoder("princeton-logion/logion-bert-base", "invalid")


"""
load_device tests
"""


@patch("torch.cuda.is_available")
def cuda_available(mock_cuda_available):
    mock_cuda_available.return_value = True
    model, _ = model_loader.load_encoder(
        "princeton-logion/logion-bert-base", "bert"
    )
    model = model_loader.load_device(model)
    assert model.device.type == "cuda"


@patch("torch.backends.mps.is_available")
def mps_available(mock_mps_available):
    mock_mps_available.return_value = True
    model, _ = model_loader.load_encoder(
        "princeton-logion/logion-bert-base", "bert"
    )
    model = model_loader.load_device(model)
    assert model.device.type == "mps"


@patch("torch.cuda.is_available")
def cuda_not_available(mock_cuda_available):
    mock_cuda_available.return_value = False
    model, _ = model_loader.load_encoder(
        "princeton-logion/logion-bert-base", "bert"
    )
    model = model_loader.load_device(model)
    assert model.device.type == "cpu"
