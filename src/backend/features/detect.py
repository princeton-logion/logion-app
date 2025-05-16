import torch
import numpy as np
from polyleven import levenshtein
import logging
from . import logion_class, cancel
import asyncio
from typing import Callable, Coroutine, Any, Dict, List, Tuple

# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

async def detection_function(
    text: str,
    model: logion_class.Logion,
    tokenizer: str,
    device: torch.device,
    chunk_size: int,
    lev: int,
    no_beam: bool,
    task_id: str,
    progress_callback: ProgressCallback,
    cancellation_event: asyncio.Event,
) -> Tuple[Dict[Tuple[str, float, int], List[Tuple[str, float]]], List[float]]:
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
        Tuple comprised of:
            final_predictions (dict) -- EX: {(original_word, chance_score): [(suggested_word_1, confidence_score_1), ...], ...}
            ccr (list) -- list chance-confidence scores
            *** list indices correspond to order of dict key entries ***
    """

    logging.info(f"Task {task_id}: Begin error detection task.")

    # set seed for reproducibiilty
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if text == "":
        logging.info(f"Task {task_id}: Input text cannot be empty.")
        await progress_callback(100.0, "Input text cannot be empty.")
        return []

    final_predictions = {}
    ccr = []
    device = model.device
    start_token = model.Tokenizer.cls_token_id
    end_token = model.Tokenizer.sep_token_id

    await progress_callback(12.0, "Tokenizing text...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    tokens_full = tokenizer.encode(text, add_special_tokens=False)
    num_tokens_full = len(tokens_full)

    await progress_callback(15.0, "Chunking text...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    chunk_start = 0
    while chunk_start < num_tokens_full:
        await cancel.check_cancel_status(cancellation_event, task_id)

        progess_percent = 15.0 + (chunk_start / num_tokens_full) * 85.0
        # chunks processed / total (approx)
        total_chunks = (num_tokens_full + chunk_size - 1) // chunk_size # approx
        current_chunk_num = (chunk_start // chunk_size) + 1
        await progress_callback(progess_percent, f"Processing chunk {current_chunk_num}/{total_chunks}...")

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

        pre_chance_progress = progess_percent
        progress_range_chance = 20.0
        await progress_callback(pre_chance_progress, "Calculating chance scores...")

        # compute chance score for each token
        tkn_chance_scores = await model.get_chance_scores(
            token_ids,
            task_id=task_id,
            progress_callback=progress_callback,
            cancellation_event=cancellation_event,
            base_progress=pre_chance_progress,
            progress_range=progress_range_chance
        )

        post_chance_progress = min(pre_chance_progress + progress_range_chance, 50.0)
        await progress_callback(post_chance_progress, "Processing words for suggestions...")

        tkn_chance_scores = tkn_chance_scores[1:-1]

        tokens_decode = tokenizer.convert_ids_to_tokens(torch.tensor(chunk_tokens))
        #logging.info(f"Task {task_id}: Tokens: {tokens_decode}")
        logging.info(f"Task {task_id}: Number of tokens: {len(tokens_decode)}")

        # compute word-level chance scores
        word_chance_scores: list = []
        for i in range(len(tokens_decode)):
            if tokens_decode[i].startswith("##"):
                if tkn_chance_scores[i] < word_chance_scores[-1]:
                    word_chance_scores[-1] = tkn_chance_scores[i]
            else:
                word_chance_scores.append(tkn_chance_scores[i])

        words: List[List[int]] = []
        for i in range(len(tokens_decode)):
            if not tokens_decode[i].startswith("##"):
                words.append([token_ids[0, 1:-1][i].item()])
            else:
                words[-1] = words[-1] + [token_ids[0, 1:-1][i].item()]

        #logging.info(f"Task {task_id}: Word scores: {word_chance_scores}\nWords: {words}")
        logging.info(f"Task {task_id}: Words in chunk: {len(words)}\nWord scores in chunk: {len(word_chance_scores)}")

        all_suggestions = []
        words_in_chunk = len(words)

        pre_confidence_progress = post_chance_progress
        progress_range_confidence = 95.0 - pre_confidence_progress

        # generate suggestions with confidence scores
        for word_ind, word_score in enumerate(word_chance_scores):
            await cancel.check_cancel_status(cancellation_event, task_id)

            inner_loop_progress = (word_ind + 1) / words_in_chunk if words_in_chunk else 0
            current_suggestion_progress = pre_confidence_progress + (inner_loop_progress * progress_range_confidence)
            await progress_callback(current_suggestion_progress, f"Generating suggestions ({word_ind+1}/{words_in_chunk})...")

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

            suggestions = await model.fill_the_blank_given_transmitted(
                model.display_sentence(
                    tokenizer.convert_ids_to_tokens(pre_tokens)
                ), # pre-text
                model.display_sentence(
                    tokenizer.convert_ids_to_tokens(post_tokens)
                ), # post-text
                model.display_sentence(
                    tokenizer.convert_ids_to_tokens(transmitted_tokens)
                ), # masked word
                transmitted_tokens,
                masked_token_ids,
                max_lev=lev,
                no_beam=no_beam,
                task_id=task_id,
                progress_callback=progress_callback,
                cancellation_event=cancellation_event
            )
            all_suggestions.append(suggestions)


        # calculate chance-confidence scores
        chunk_predictions = {}
        chunk_ccr = []
        for word_index, (word, word_chance_score) in enumerate(zip(words, word_chance_scores)):
            global_word_index = chunk_start + word_index
            original_word = model.display_sentence(
                tokenizer.convert_ids_to_tokens(word)
            )

            chunk_predictions[(original_word, word_chance_score, global_word_index)] = (
                all_suggestions[word_index]
            )

            first_sug_confidence_score = all_suggestions[word_index][0][1]
            ccr_value = (
                word_chance_score / first_sug_confidence_score
                if first_sug_confidence_score != 0
                else float("inf")
            )
            chunk_ccr.append(ccr_value)

        #logging.info(f"Task {task_id}: Chunk Predictions: {chunk_predictions}\nChunk CCR: {chunk_ccr}")
        logging.info(f"Task {task_id}: Length Chunk Predictions: {len(chunk_predictions)}\nLength Chunk CCR: {len(chunk_ccr)}")

        final_predictions.update(chunk_predictions)
        ccr.extend(chunk_ccr)
        # set new chunk start at end of current chunk for next loop
        chunk_start = chunk_end

    await progress_callback(97.0, "Error detection finished.")
    await cancel.check_cancel_status(cancellation_event, task_id)

    #logging.info(f"Task {task_id}: Final Predictions: {final_predictions}\nCCR: {ccr}")
    logging.info(f"Task {task_id}: Length Final Predictions: {len(final_predictions)}\nLength CCR: {len(ccr)}")

    return final_predictions, ccr
