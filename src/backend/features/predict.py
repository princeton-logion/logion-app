import torch
from collections import defaultdict
import logging
import numpy as np
import asyncio
from typing import Callable, Coroutine, Any, Dict, List, Tuple
from . import cancel, predict_utils

# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]


async def prediction_function(
        text: str,
        model: torch.nn.Module,
        tokenizer: str,
        device: torch.device,
        chunk_size: int,
        num_preds: int,
        task_id: str,
        progress_callback: ProgressCallback,
        cancellation_event: asyncio.Event
        ) -> Dict[int, List[Tuple[str, float]]]:
    """
    Masked language modeling inference for lacuna predictions using sliding window

    Parameters:
        text (str) -- input text with [MASK]s
        model (str) -- encoder-only model
        tokenizer (str) -- tokenizer for model
        device (torch.device) --
        chunk_size (int) -- max input length
        num_preds (int) -- number of suggestions per word
        task_id (str) -- id for prediction task
        progress_callback -- async callback for progress updates
        cancellation_event -- event check for task cancellation

    Returns:
        final_predictions (dict) -- {mask_token_index_1: [(predicted_token_1, probability_score_1), ...], ...}
    """
    
    logging.info(f"Task {task_id}: Begin word prediction task {task_id}")

    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if text == "":
        logging.info(f"Task {task_id}: Input text cannot be empty.")
        await progress_callback(100.0, "Input text cannot be empty.")
        return {}
    
    start_token = tokenizer.cls_token_id
    end_token = tokenizer.sep_token_id

    await progress_callback(12.0, "Preparing text for model processing...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    all_predictions = defaultdict(list)
    tokens_full = tokenizer.encode(text, add_special_tokens=False)
    num_tokens_full = len(tokens_full)

    # find [MASK]s across whole txt
    global_mask_indices_to_process = {
        i for i, token_id in enumerate(tokens_full) if token_id == tokenizer.mask_token_id
    }

    await progress_callback(15.0, "Dividing text into chunks...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    chunk_start = 0
    while chunk_start < num_tokens_full:
        await cancel.check_cancel_status(cancellation_event, task_id)

        progress_percent = 15.0 + (chunk_start / num_tokens_full) * 85.0
        # chunks processed / total (approx)
        total_chunks = (num_tokens_full + chunk_size - 1) // chunk_size # approx
        current_chunk_num = (chunk_start // chunk_size) + 1
        await progress_callback(progress_percent, f"Processing chunk {current_chunk_num}/{total_chunks}...")

        chunk_end = min(chunk_start + chunk_size, num_tokens_full)
        chunk_tokens = tokens_full[chunk_start:chunk_end]

        # "." = end of chunk
        while (
            chunk_end > chunk_start
            and tokenizer.convert_ids_to_tokens([chunk_tokens[-1]])[0] != "."
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

        # get positions of [MASK] tkns in chunk
        masked_indices_in_chunk = [
            idx + 1 for idx, token_id in enumerate(chunk_tokens) if token_id == tokenizer.mask_token_id
        ]
        if not masked_indices_in_chunk:
            chunk_start = chunk_end
            continue

        for mask_idx_in_chunk in masked_indices_in_chunk:
            global_mask_idx = chunk_start + mask_idx_in_chunk - 1

            if global_mask_idx in global_mask_indices_to_process:
                logging.info(f"Task {task_id}: Generating beam search predictions for [MASK] at {global_mask_idx}")
                
                suggestions = await predict_utils.generate_multi_token_suggestions(
                    input_token_ids=token_ids,
                    mask_idx_in_chunk=mask_idx_in_chunk,
                    model=model,
                    tokenizer=tokenizer,
                    num_preds=num_preds,
                    task_id=task_id,
                    cancellation_event=cancellation_event
                )

                logging.info(f"Task {task_id}: Predictions for mask {global_mask_idx}: {suggestions}")
                all_predictions[global_mask_idx].extend(suggestions)
                global_mask_indices_to_process.remove(global_mask_idx)

        chunk_start = chunk_end

    await progress_callback(95.0, "Gathering all predictions...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    # compile predictions and rmv duplicates
    final_predictions = {}
    for masked_index, prediction_list in all_predictions.items():
        sorted_predictions = sorted(prediction_list, key=lambda x: x[1], reverse=True)
        unique_preds = {}
        for word, prob in sorted_predictions:
            if word not in unique_preds:
                unique_preds[word] = prob
        final_predictions[masked_index] = list(unique_preds.items())[:num_preds]

    #logging.info(f"Task {task_id}: Final predictions: {type(final_predictions)}\n{final_predictions}")
    await progress_callback(97.0, "Generated predictions.")
    return final_predictions
