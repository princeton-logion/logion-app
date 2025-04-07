import torch
import numpy as np
from polyleven import levenshtein
import logging
from . import logion_class


def detection_function(
    text: str,
    model: logion_class.Logion,
    tokenizer,
    device,
    chunk_size: int = 500,
    lev: int = 1,
    no_beam=False
):
    """
    Masked language modeling for detecting and suggesting corrections to potential erroneous words in input text.
    Requires Logion class and .npy matrix.

    Parameters:
        text (str) -- input text
        model (Logion) -- instance of Logion (loaded model and tokenizer)
        tokenizer (str) -- tokenizer
        chunk_size (int) -- max number of tokens for window
        lev (int) -- max Levenshtein distance between original words and suggested replacements
        no_beam (bool) --

    Returns:
        final_predictions (dict) -- EX: {(original_word, chance_score): [(suggested_word_1, confidence_score_1), ...], ...}
        ccr (list) -- list chance-confidence scores
        *** list indices correspond to order of dict key entries ***
    """
    # set seed for reproducibiilty
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if text == "":
        raise ValueError("Input text cannot be empty.")

    final_predictions = {}
    ccr = []
    device = model.device
    start_token = model.Tokenizer.cls_token_id
    end_token = model.Tokenizer.sep_token_id

    tokens_full = tokenizer.encode(text, add_special_tokens=False)
    num_tokens_full = len(tokens_full)

    chunk_start = 0
    while chunk_start < num_tokens_full:
        chunk_end = min(chunk_start + chunk_size, num_tokens_full)
        chunk_tokens = tokens_full[chunk_start:chunk_end]

        # "." = end of chunk
        while (
            chunk_end > chunk_start
            and tokenizer.convert_ids_to_tokens(chunk_tokens[-1]) != "."
        ):
            chunk_end -= 1
            chunk_tokens = tokens_full[chunk_start:chunk_end]
        if chunk_end == chunk_start:
            # if no ".", start new chunk
            chunk_end = min(chunk_start + chunk_size, num_tokens_full)
            chunk_tokens = tokens_full[chunk_start:chunk_end]

        token_ids = (
            torch.tensor([start_token] + chunk_tokens + [end_token])
            .unsqueeze(0)
            .to(device)
        )

        # compute chance score for each token
        scores = model.get_chance_scores(token_ids)[1:-1]

        # convert token IDs to words
        tokens_decode = tokenizer.convert_ids_to_tokens(torch.tensor(chunk_tokens))
        logging.info(f"Tokens: {tokens_decode}")
        logging.info(f"Number of tokens: {len(tokens_decode)}")

        word_scores: list = []
        for i in range(len(tokens_decode)):
            if tokens_decode[i].startswith("##"):
                if scores[i] < word_scores[-1]:
                    word_scores[-1] = scores[i]
            else:
                word_scores.append(scores[i])

        words = []
        for i in range(len(tokens_decode)):
            if not tokens_decode[i].startswith("##"):
                words.append([token_ids[0, 1:-1][i].item()])
            else:
                words[-1] = words[-1] + [token_ids[0, 1:-1][i].item()]

        logging.info(f"Word scores: {word_scores}")
        logging.info(f"Words: {words}")
        logging.info(f"Number of words: {len(words)}")
        logging.info(f"Number of word scores: {len(word_scores)}")

        all_suggestions = []
        for word_ind, word_score in enumerate(word_scores):
            # separate tokens into [MASK] token, pre, and post tokens
            pre_tokens = sum(words[:word_ind], [])
            transmitted_tokens = words[word_ind]
            post_tokens = sum(words[word_ind + 1 :], [])

            masked_token_ids = torch.unsqueeze(
                torch.tensor(
                    [start_token]
                    + pre_tokens
                    + [tokenizer.mask_token_id]
                    + post_tokens
                    + [end_token]
                ),
                0,
            ).to(device)

            suggestions = model.fill_the_blank_given_transmitted(
                model.display_sentence(
                    tokenizer.convert_ids_to_tokens(pre_tokens)
                ),  # pre-text
                model.display_sentence(
                    tokenizer.convert_ids_to_tokens(post_tokens)
                ),  # post-text
                model.display_sentence(
                    tokenizer.convert_ids_to_tokens(transmitted_tokens)
                ),  # masked word
                transmitted_tokens,
                masked_token_ids,
                max_lev=lev,
                no_beam=no_beam,
            )
            all_suggestions.append(suggestions)

        chunk_predictions = {}
        chunk_ccr = []
        for word_index, (word, score) in enumerate(zip(words, word_scores)):
            global_word_index = chunk_start + word_index
            original_word = model.display_sentence(
                tokenizer.convert_ids_to_tokens(word)
            )

            chunk_predictions[(original_word, score, global_word_index)] = (
                all_suggestions[word_index]
            )

            first_suggestion_score = all_suggestions[word_index][0][1]
            ccr_value = (
                score / first_suggestion_score
                if first_suggestion_score != 0
                else float("inf")
            )
            chunk_ccr.append(ccr_value)

        logging.info(f"Chunk Predictions: {chunk_predictions}")
        logging.info(f"Chunk CCR: {chunk_ccr}")
        logging.info(f"Length Chunk Predictions: {len(chunk_predictions)}")
        logging.info(f"Length Chunk CCR: {len(chunk_ccr)}")

        final_predictions.update(chunk_predictions)
        ccr.extend(chunk_ccr)
        # set new chunk start at end of current chunk for next loop
        chunk_start = chunk_end

    logging.info(f"Final Predictions: {final_predictions}")
    logging.info(f"CCR: {ccr}")
    logging.info(f"Length Final Predictions: {len(final_predictions)}")
    logging.info(f"Length CCR: {len(ccr)}")

    return final_predictions, ccr
