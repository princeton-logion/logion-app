import pytest
import torch
from unittest.mock import MagicMock
from ..src.backend.detection import logion_class


"""
Tests for Logion class functions
"""


@pytest.fixture
def logion_instance():
    """
    Define fixture of Logion class instance for reuse.
    Use mock model, tokenizer, Lev filter, and device.
    """
    model = MagicMock()
    model.return_value.logits = torch.randn(1, 10, 1000)

    tokenizer = MagicMock()
    tokenizer.mask_token_id = 0
    tokenizer.convert_ids_to_tokens = lambda ids: [f"token_{id}" for id in ids]
    tokenizer.encode = lambda text, return_tensors: torch.tensor([[1, 2, 3, 0, 4, 5]])

    lev_filter = torch.ones(1000, 1000)

    device = torch.device("cpu")

    return logion_class.Logion(model, tokenizer, lev_filter, device)


def test_get_chance_probability(logion_instance):
    """
    Verify data types of "probabilities" and "underlying_token_ids" returned by _get_chance_probability().
    Uses example torch tensor.

    Assert:
        1. probabilities is tensor
        2. underlying_token_id is integer
    """
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    index_of_id_to_mask = 2

    probabilities, underlying_token_id = logion_instance._get_chance_probability(token_ids, index_of_id_to_mask)

    assert isinstance(probabilities, torch.Tensor), "Probabilities must be tensor."
    assert isinstance(underlying_token_id, int), "Underlying token ID must be integer."


def test_get_chance_scores(logion_instance):
    """
    Verify data type and length of "scores" returned by get_chance_scores().
    Uses example torch tensor.

    Assert:
        1. scores is list
        2. length of scores equals number of token_ids
    """
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    relevant_token_ids = [2, 4]

    scores = logion_instance.get_chance_scores(token_ids, relevant_token_ids)

    assert isinstance(scores, list), "Scores must be in a list."
    assert len(scores) == token_ids.shape[1], "Length of scores list must equal the number of tokens."


def test_get_mask_probabilities(logion_instance):
    """
    Verify data type and dimension of "probabilities" returned by _get_mask_probabilities().
    Uses example torch tensor.

    Assert:
        1. probabilities is tensor
        2. probabilities has 2 dimensions
    """
    masked_text_ids = torch.tensor([[1, 2, 0, 4, 5]])

    probabilities = logion_instance._get_mask_probabilities(masked_text_ids)

    assert isinstance(probabilities, torch.Tensor), "Probabilities must be tensor."
    assert probabilities.ndim == 2, "Probabilities must have two dimensions."


def test_argkmax(logion_instance):
    """
    Verify data type and length of "indicies" returned by _argkmax().
    Verify data type of entires in  type of entries in "indices".
    Uses example torch tensor.

    Assert:
        1. indices is tensor
        2. indices length equals k (i.e. 3)
        3. items in indices are integers
    """
    array = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    k = 3

    indices = logion_instance._argkmax(array, k)


    assert isinstance(indices, torch.Tensor), "Indicies must be tensor."
    assert len(indices) == k, "Should return k=3 indices."
    assert all(isinstance(i, int) for i in indices.tolist()), "Indices must be integers."


def test_display_sentence(logion_instance):
    """
    Verify data type "sentence" returned by _argkmax().
    Verify correct piecing together of subword tokens in "sentence".
    Uses example Greek from Psellos.

    Assert:
        1. sentence is string
        2. correct piecing together of subword tokens
    """
    toks = ["κινει", "##σθω", "δε", "παρ", "##´", "αλλου"]

    sentence = logion_instance.display_sentence(toks)

    assert isinstance(sentence, str), "Output must be string."
    assert sentence == "κινεισθω δε παρ´ αλλου", "Incorrect sentence display."


def test_argkmax_beam(logion_instance):
    array = torch.randn(1, 10)
    k = 3
    prefix = "tok"

    indices, new_prefixes = logion_instance._argkmax_beam(array, k, prefix)

    assert len(indices) == k, "Should return k=3 indices."
    assert len(new_prefixes) == k, "Should return k=3 prefixes."


def test_fill_the_blank_given_transmitted(logion_instance):
    pre_text = "Ἐν ἀρχῇ ἦν "
    post_text = ", καὶ θεὸς ἦν ὁ λόγος."
    transmitted_text = "λόγος"
    transmitted_tokens = [10]
    tokens = torch.tensor([[1, 2, 0, 4, 5]])

    suggestions = logion_instance.fill_the_blank_given_transmitted(
        pre_text, post_text, transmitted_text, transmitted_tokens, tokens
    )

    assert isinstance(suggestions, list), "Suggestions must be in a list."
    assert len(suggestions) >= 1, "Should returns 1 or more suggestions."


def test_display_word(logion_instance):
    toks = ["κινει", "##σθω"]

    word = logion_instance._display_word(toks)

    assert isinstance(word, str), "Output must be a string."
    assert word == "κινεισθω", "Incorrect word display."
