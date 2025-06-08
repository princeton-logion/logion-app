import torch
from collections import defaultdict
import logging
import numpy as np
import asyncio
from typing import Callable, Coroutine, Any, Dict, List, Tuple
from . import cancel

# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

async def prediction_function(
        text: str,
        model: torch.nn.Module,
        tokenizer: str,
        device: torch.device,
        window_size: int,
        overlap: int,
        num_predictions: int,
        task_id: str,
        progress_callback: ProgressCallback,
        cancellation_event: asyncio.Event
        ) -> Dict[int, List[Tuple[str, float]]]:
    """
    Masked language modeling inference for lacuna predictions using sliding window

    Parameters:
        text (str) -- input text
        model (str) -- encoder-only model
        tokenizer (str) -- tokenizer for model
        window_size (int) -- sliding window size
        overlap (int) -- sliding window overlap
        num_predictions (int) -- number of suggestions per word
        task_id (str) -- id for prediction task
        progress_callback --
        cancellation_event -- 

    Returns:
        final_predictions (dict) -- {mask_token_index_1: [(predicted_token_1, probability_score_1), ...], ...}
    """

    logging.info(f"Task {task_id}: Begin prediction task {task_id}")

    # set seed for reproducibiilty
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if text == "":
        logging.info(f"Task {task_id}: Input text cannot be empty.")
        await progress_callback(100.0, "Input text cannot be empty.")
        return {}

    await progress_callback(15.0, "Tokenizing text...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    all_predictions = defaultdict(list)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(tokens)

    await progress_callback(20.0, "Chunking text...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    step = window_size - overlap
    if step <= 0:
        step = max(1, window_size // 4)
        logging.info(f"Task {task_id}: Window size <= overlap. Using step {step}")

    # calculate total iterations for progress update
    num_iterations = (num_tokens + step -1) // step
    if num_iterations == 0 : num_iterations = 1 # account for div by 0 when num_tokens < step

    for iteration, i in enumerate(range(0, num_tokens, step)):
        await cancel.check_cancel_status(cancellation_event, task_id)

        progress_percent = 20.0 + (iteration / num_iterations) * 85.0
        await progress_callback(progress_percent, f"Processing chunk {iteration + 1}/{num_iterations}...")

        chunk_ids = tokens[i : min(i + window_size, num_tokens)]
        chunk_ids = chunk_ids[:512]
        chunk = tokenizer.decode(chunk_ids)
        chunk_inputs = tokenizer(
            chunk,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            max_length=512
        )


        chunk_inputs = {k: v.to(device) for k, v in chunk_inputs.items()}

        masked_indices = [
            index
            for index, token_id in enumerate(chunk_inputs["input_ids"][0])
            if token_id == tokenizer.mask_token_id
        ]
        logging.info(f"Task {task_id}: Chunk {i} Masks: {masked_indices}")

        with torch.no_grad():
            outputs = model(**chunk_inputs)
            predictions_logits = outputs.logits

        for masked_index in masked_indices:
            predicted_probs = predictions_logits[0, masked_index]
            sorted_preds, sorted_index = torch.sort(predicted_probs, descending=True)
            masked_predictions = []
            for k in range(num_predictions):
                predicted_index = sorted_index[k].item()
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                probability = torch.softmax(predicted_probs, dim=-1)[
                    predicted_index
                ].item()
                masked_predictions.append((predicted_token, probability))
            logging.info(f"Task {task_id}: Predictions for {i}.{masked_index}: {masked_predictions}") # i = chunk start
            all_predictions[masked_index + i].extend(masked_predictions) # masked_index + i = global mask index

    await progress_callback(95.0, "Gathering all predictions...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    final_predictions = {}
    for masked_index, prediction_list in all_predictions.items():
        # group subword predictions
        subword_groups: Dict[str, List[Tuple[str, float]]] = {}
        for token, prob in prediction_list:
            if token.startswith("##"):
                base_word = token[2:] # remove "##" prefix
                if base_word not in subword_groups:
                    subword_groups[base_word] = []
                subword_groups[base_word].append((token, prob))
            else: # whole word token
                subword_groups[token] = [(token, prob)]
        logging.info(f"Task {task_id}: Subword groups for mask {masked_index}: {subword_groups}")

        whole_word_predictions = []
        for base_word, subword_list in subword_groups.items():
            max_prob = 0.0
            for _subtoken, prob in subword_list:
                if prob > max_prob:
                    max_prob = prob
            whole_word_predictions.append((base_word, max_prob))

        # sort by prob
        sorted_predictions = sorted(
            whole_word_predictions, key=lambda x: x[1], reverse=True
        )
        # keep top num_predictions
        final_predictions[masked_index] = sorted_predictions[:num_predictions]

    #logging.info(f"Task {task_id}: Final predictions: {type(final_predictions)}\n{final_predictions}")

    await progress_callback(97.0, "Generated predictions.")
    return final_predictions
