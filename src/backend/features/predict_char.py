import torch
from collections import defaultdict
import logging
import numpy as np
import asyncio
from typing import Callable, Coroutine, Any, Dict, List, Tuple
from . import cancel, clean_txt

# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]


async def char_prediction_function(
        text: str,
        model: torch.nn.Module,
        char_stoi: Dict[str, int],
        char_itos: Dict[int, str],
        mask_id: int,
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
        char_stoi (dict) -- 
        char_itos (dict) -- 
        mask_id (int) --
        device (torch.device) --
        chunk_size (int) -- max input length
        num_preds (int) -- number of suggestions per word
        task_id (str) -- id for prediction task
        progress_callback -- async callback for progress updates
        cancellation_event -- event check for task cancellation

    Returns:
        final_predictions (dict) -- {mask_token_index_1: [(predicted_token_1, probability_score_1), ...], ...}
    """
    
    logging.info(f"Task {task_id}: Begin character prediction task {task_id}")

    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if text == "":
        logging.info(f"Task {task_id}: Input text cannot be empty.")
        await progress_callback(100.0, "Input text cannot be empty.")
        return {}
    
    cls_id = char_stoi.get("[CLS]", None)
    sep_id = char_stoi.get("[SEP]", None)
    pad_id = char_stoi.get("[PAD]", 0)
    unk_id = char_stoi.get("[UNK]", None)

    await progress_callback(12.0, "Preparing text for model processing...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    text = clean_txt.clean_input_txt(text)

    all_predictions = defaultdict(list)
    
    # handle special tkns in txt-ID converstion
    tokens_full = []
    i = 0
    while i < len(text):
        if text[i:i+6] == "[MASK]":
            tokens_full.append(mask_id)
            i += 6
        elif text[i:i+5] == "[CLS]":
            tokens_full.append(cls_id)
            i += 5
        elif text[i:i+5] == "[SEP]":
            tokens_full.append(sep_id)
            i += 5
        elif text[i:i+5] == "[PAD]":
            tokens_full.append(pad_id)
            i += 5
        elif text[i:i+5] == "[UNK]":
            tokens_full.append(char_stoi["[UNK]"])
            i += 5
        else:
            # now go to actual txt
            char = text[i]
            if char in char_stoi:
                tokens_full.append(char_stoi[char])
            else:
                tokens_full.append(unk_id)
            i += 1
    
    num_tokens_full = len(tokens_full)
    logging.info(f"Task {task_id}: Converted {len(text)} chars to {num_tokens_full} tokens")

    # find [MASK]s across whole txt
    global_mask_indices_to_process = {
        i for i, token_id in enumerate(tokens_full) if token_id == mask_id
    }

    logging.info(f"Task {task_id}: {len(global_mask_indices_to_process)} [MASK] tokens in input")
    
    if not global_mask_indices_to_process:
        logging.info(f"Task {task_id}: No [MASK] tokens in input")
        await progress_callback(100.0, "No missing characters in input.")
        return {}

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
        period_id = char_stoi.get(".", None)
        while chunk_end > chunk_start and chunk_tokens[-1] != period_id:
            chunk_end -= 1
            chunk_tokens = tokens_full[chunk_start:chunk_end]
        if chunk_end == chunk_start:
            # if no ".", start new chunk
            chunk_end = min(chunk_start + chunk_size, num_tokens_full)
            chunk_tokens = tokens_full[chunk_start:chunk_end]

        # input tensor with CLS/SEP tkns
        token_ids = (
            torch.tensor([cls_id] + chunk_tokens + [sep_id])
            .unsqueeze(0)
            .to(device)
        )
        # adjust indxing per [CLS]
        cls_offset = 1

        # get positions of [MASK] tkns in chunk
        masked_indices_in_chunk = [
            idx for idx, token_id in enumerate(chunk_tokens) if token_id == mask_id
        ]
        if not masked_indices_in_chunk:
            chunk_start = chunk_end
            continue
        
        # create attn mask
        attention_mask = torch.ones_like(token_ids).to(device)

        # run inference for chunk
        with torch.no_grad():
            outputs = model(
                input_ids=token_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

        # process each mask in the chunk
        for mask_idx_in_chunk in masked_indices_in_chunk:
            global_mask_idx = chunk_start + mask_idx_in_chunk
            
            if global_mask_idx in global_mask_indices_to_process:
                # account for [CLS] w/ cls_offset
                mask_position_in_input = mask_idx_in_chunk + cls_offset
                mask_logits = logits[0, mask_position_in_input, :]  # Shape: (vocab_size,)
                
                probs = torch.nn.functional.softmax(mask_logits, dim=-1)
                
                # get 2x top_k preds to account for special tkns
                top_k = min(num_preds * 2, len(probs))
                top_probs, top_indices = torch.topk(probs, top_k)
                
                logging.info(f"Task {task_id}: Generating predictions for [MASK] at position {global_mask_idx}")
                
                suggestions = []
                for prob_val, token_id in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                    token_id = int(token_id)
                    
                     # ignore special tkn predictions
                    if token_id == mask_id:
                        continue
                    if token_id == pad_id:
                        continue
                    if token_id == cls_id:
                        continue
                    if token_id == sep_id:
                        continue
                    if token_id == unk_id:
                        continue
                    
                    predicted_char = char_itos.get(token_id, "?")
                    suggestions.append((predicted_char, float(prob_val)))
                    
                    if len(suggestions) >= num_preds:
                        break
                
                logging.info(f"Task {task_id}: Character predictions for mask {global_mask_idx}: {suggestions}")
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
        for char, prob in sorted_predictions:
            if char not in unique_preds:
                unique_preds[char] = prob
        final_predictions[masked_index] = list(unique_preds.items())[:num_preds]

    #logging.info(f"Task {task_id}: Final predictions: {type(final_predictions)}\n{final_predictions}")
    await progress_callback(97.0, "Generated predictions.")
    return final_predictions