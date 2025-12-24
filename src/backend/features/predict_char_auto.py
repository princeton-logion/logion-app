import torch
from collections import defaultdict
import logging
import numpy as np
import asyncio
from typing import Callable, Coroutine, Any, Dict, List, Tuple, Set
from . import cancel, clean_txt, predict_char_utils_auto

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
        cancellation_event: asyncio.Event,
        beam_size: int = 20
        ) -> Dict[int, List[Tuple[str, float]]]:

    
    logging.info(f"Task {task_id}: Begin character prediction task {task_id}")

    # Set seeds for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # Validate input
    if text == "":
        logging.info(f"Task {task_id}: Input text cannot be empty.")
        await progress_callback(100.0, "Input text cannot be empty.")
        return {}
    
    # Extract special token IDs from vocabulary
    cls_id = char_stoi.get("[CLS]", None)
    sep_id = char_stoi.get("[SEP]", None)
    pad_id = char_stoi.get("[PAD]", 0)
    unk_id = char_stoi.get("[UNK]", None)
    
    # Build set of special token IDs to exclude from predictions
    special_token_ids: Set[int] = {
        tid for tid in [mask_id, cls_id, sep_id, pad_id, unk_id] if tid is not None
    }

    await progress_callback(12.0, "Preparing text for model processing...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    # Clean input text
    text = clean_txt.clean_input_txt(text)

    # === TOKENIZATION ===
    # Convert text to token IDs, handling special token strings
    tokens_full: List[int] = []
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
            tokens_full.append(unk_id)
            i += 5
        else:
            # Regular character
            char = text[i]
            if char in char_stoi:
                tokens_full.append(char_stoi[char])
            else:
                tokens_full.append(unk_id)
            i += 1
    
    num_tokens_full = len(tokens_full)
    logging.info(f"Task {task_id}: Converted {len(text)} chars to {num_tokens_full} tokens")

    # === IDENTIFY MASK SPANS ===
    # Group consecutive [MASK] tokens into spans for joint prediction
    global_mask_spans = predict_char_utils_auto.identify_mask_spans(tokens_full, mask_id)
    
    logging.info(f"Task {task_id}: Found {len(global_mask_spans)} mask span(s)")
    for span_start, span_end in global_mask_spans:
        logging.info(f"Task {task_id}:   Span at positions {span_start}-{span_end-1} (length {span_end - span_start})")
    
    if not global_mask_spans:
        logging.info(f"Task {task_id}: No [MASK] tokens in input")
        await progress_callback(100.0, "No missing characters in input.")
        return {}

    # Track which spans still need processing
    spans_to_process: Set[Tuple[int, int]] = set(global_mask_spans)
    
    # Store predictions: maps global_char_index -> list of (span_string, probability)
    all_predictions: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

    await progress_callback(15.0, "Dividing text into chunks...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    # === CHUNK PROCESSING ===
    # Process text in chunks to handle sequences longer than model's max length
    chunk_start = 0
    while chunk_start < num_tokens_full:
        await cancel.check_cancel_status(cancellation_event, task_id)

        # Update progress
        progress_percent = 15.0 + (chunk_start / num_tokens_full) * 80.0
        total_chunks = (num_tokens_full + chunk_size - 1) // chunk_size
        current_chunk_num = (chunk_start // chunk_size) + 1
        await progress_callback(progress_percent, f"Processing chunk {current_chunk_num}/{total_chunks}...")

        # Determine chunk boundaries
        chunk_end = min(chunk_start + chunk_size, num_tokens_full)
        chunk_tokens = tokens_full[chunk_start:chunk_end]

        # Try to end chunk at sentence boundary (period)
        period_id = char_stoi.get(".", None)
        original_chunk_end = chunk_end
        while chunk_end > chunk_start and chunk_tokens[-1] != period_id:
            chunk_end -= 1
            chunk_tokens = tokens_full[chunk_start:chunk_end]
        
        # If no period found, use original chunk boundary
        if chunk_end == chunk_start:
            chunk_end = original_chunk_end
            chunk_tokens = tokens_full[chunk_start:chunk_end]

        # === FIND COMPLETE SPANS IN THIS CHUNK ===
        # Only process spans that are fully contained within this chunk
        # (to avoid partial predictions at chunk boundaries)
        spans_in_chunk: List[Tuple[int, int]] = []
        for span_start_global, span_end_global in spans_to_process:
            # Check if entire span is within chunk boundaries
            if span_start_global >= chunk_start and span_end_global <= chunk_end:
                spans_in_chunk.append((span_start_global, span_end_global))
        
        if not spans_in_chunk:
            chunk_start = chunk_end
            continue

        # === PREPARE INPUT TENSOR ===
        # Add [CLS] at start and [SEP] at end
        token_ids = (
            torch.tensor([cls_id] + chunk_tokens + [sep_id])
            .unsqueeze(0)
            .to(device)
        )
        # Offset for [CLS] token at position 0
        cls_offset = 1

        # === PROCESS EACH SPAN ===
        for span_start_global, span_end_global in spans_in_chunk:
            # Convert global positions to local positions within chunk
            span_start_local = span_start_global - chunk_start
            span_length = span_end_global - span_start_global
            
            # Account for [CLS] token offset in input tensor
            span_start_in_input = span_start_local + cls_offset
            
            logging.info(
                f"Task {task_id}: Generating beam search predictions for span at "
                f"global positions {span_start_global}-{span_end_global-1} "
                f"(length {span_length})"
            )
            
            # === RUN BEAM SEARCH FOR SPAN ===
            span_predictions = await predict_char_utils_auto.generate_span_predictions(
                token_ids=token_ids,
                span_start_in_input=span_start_in_input,
                span_length=span_length,
                model=model,
                char_itos=char_itos,
                special_token_ids=special_token_ids,
                device=device,
                num_preds=num_preds,
                task_id=task_id,
                cancellation_event=cancellation_event,
                beam_size=beam_size
            )
            
            logging.info(
                f"Task {task_id}: Span predictions for positions "
                f"{span_start_global}-{span_end_global-1}: {span_predictions}"
            )
            
            # === ASSIGN SPAN PREDICTIONS TO ALL CHARACTER POSITIONS ===
            # Each character in the span gets the same span-level predictions
            for global_idx in range(span_start_global, span_end_global):
                all_predictions[global_idx].extend(span_predictions)
            
            # Mark span as processed
            spans_to_process.discard((span_start_global, span_end_global))

        chunk_start = chunk_end

    await progress_callback(95.0, "Gathering all predictions...")
    await cancel.check_cancel_status(cancellation_event, task_id)

    # === COMPILE FINAL PREDICTIONS ===
    # Deduplicate and sort predictions for each position
    final_predictions: Dict[int, List[Tuple[str, float]]] = {}
    for masked_index, prediction_list in all_predictions.items():
        # Sort by probability (highest first)
        sorted_predictions = sorted(prediction_list, key=lambda x: x[1], reverse=True)
        
        # Remove duplicate span strings, keeping highest probability version
        unique_preds: Dict[str, float] = {}
        for span_str, prob in sorted_predictions:
            if span_str not in unique_preds:
                unique_preds[span_str] = prob
        
        # Store top num_preds predictions
        final_predictions[masked_index] = list(unique_preds.items())[:num_preds]

    logging.info(f"Task {task_id}: Generated predictions for {len(final_predictions)} character positions")
    await progress_callback(97.0, "Generated predictions.")
    return final_predictions