import torch
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Set
from . import cancel

"""
Helper functions for character-level span predictions using beam search.

This module provides utilities for predicting sequences of masked characters
as coherent spans rather than independent characters. It uses beam search
to find the most probable character sequences for contiguous [MASK] tokens.

Key Functions:
    - identify_mask_spans: Groups consecutive [MASK] tokens into spans
    - generate_span_predictions: Uses beam search to predict character sequences
"""

# Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def identify_mask_spans(
    tokens_full: List[int],
    mask_id: int
) -> List[Tuple[int, int]]:
    """
    Identify contiguous spans of consecutive [MASK] tokens in the token sequence.
    
    This function scans through the full token list and groups consecutive
    [MASK] tokens into spans. Each span is represented as a (start, end) tuple
    where 'start' is inclusive and 'end' is exclusive.
    
    Example:
        tokens = [1, MASK, MASK, MASK, 2, 3, MASK, MASK, 4]
        Result: [(1, 4), (6, 8)]
        - First span: positions 1, 2, 3 (3 consecutive masks)
        - Second span: positions 6, 7 (2 consecutive masks)
    
    Parameters:
        tokens_full (List[int]): Complete list of token IDs representing the input
        mask_id (int): The token ID that represents [MASK]
    
    Returns:
        List[Tuple[int, int]]: List of (start_index, end_index) tuples where:
            - start_index: First position of the mask span (inclusive)
            - end_index: Position after the last mask in span (exclusive)
            - Span length = end_index - start_index
    """
    spans = []
    i = 0
    n = len(tokens_full)
    
    while i < n:
        # Check if current position is a [MASK] token
        if tokens_full[i] == mask_id:
            span_start = i
            # Continue while we find consecutive [MASK] tokens
            while i < n and tokens_full[i] == mask_id:
                i += 1
            # Record the span (end is exclusive)
            spans.append((span_start, i))
        else:
            i += 1
    
    return spans


def _get_spans_in_chunk(
    chunk_start: int,
    chunk_end: int,
    global_spans: List[Tuple[int, int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Find which global spans overlap with the current chunk and compute local positions.
    
    For each global span that falls within or overlaps the chunk boundaries,
    this function calculates both the global indices and the corresponding
    local indices within the chunk.
    
    Parameters:
        chunk_start (int): Global start index of the current chunk
        chunk_end (int): Global end index of the current chunk (exclusive)
        global_spans (List[Tuple[int, int]]): List of (global_start, global_end) spans
    
    Returns:
        List[Tuple[int, int, int, int]]: List of tuples containing:
            (global_span_start, global_span_end, local_span_start, local_span_end)
            where local positions are relative to chunk_start
    """
    spans_in_chunk = []
    
    for global_start, global_end in global_spans:
        # Check if span overlaps with chunk
        # Span overlaps if: span_start < chunk_end AND span_end > chunk_start
        if global_start < chunk_end and global_end > chunk_start:
            # Clip span to chunk boundaries
            clipped_start = max(global_start, chunk_start)
            clipped_end = min(global_end, chunk_end)
            
            # Only include if the full span is within this chunk
            # (to avoid partial span predictions)
            if global_start >= chunk_start and global_end <= chunk_end:
                local_start = global_start - chunk_start
                local_end = global_end - chunk_start
                spans_in_chunk.append((global_start, global_end, local_start, local_end))
    
    return spans_in_chunk


async def _beam_search_chars(
    token_ids: torch.Tensor,
    model: torch.nn.Module,
    mask_positions: List[int],
    special_token_ids: Set[int],
    device: torch.device,
    beam_size: int,
    task_id: str,
    cancellation_event: asyncio.Event
) -> List[Tuple[List[int], float]]:
    """
    Perform beam search to find optimal character sequences for masked positions.
    
    This function implements beam search for character-level masked language modeling.
    It iteratively predicts each character position while maintaining the top-k
    (beam_size) most probable partial sequences at each step.
    
    Algorithm:
        1. Start with an empty sequence and probability 1.0
        2. For each mask position (left to right):
           a. For each beam, get model predictions at current mask position
           b. Expand each beam with top-k character predictions
           c. Score expanded beams by cumulative probability
           d. Keep only top beam_size candidates
        3. Return final beams sorted by probability
    
    Parameters:
        token_ids (torch.Tensor): Input tensor with [MASK] tokens at positions
                                  to be predicted. Shape: (1, seq_len)
        model (torch.nn.Module): Character-level MLM model that outputs logits
        mask_positions (List[int]): Ordered list of positions in token_ids that
                                    contain [MASK] tokens to be predicted
        special_token_ids (Set[int]): Token IDs to exclude from predictions
                                      (e.g., [PAD], [CLS], [SEP], [UNK], [MASK])
        device (torch.device): Device for model inference (CPU/GPU)
        beam_size (int): Number of top sequences to maintain at each step
        task_id (str): Identifier for logging and cancellation checking
        cancellation_event (asyncio.Event): Event to check for task cancellation
    
    Returns:
        List[Tuple[List[int], float]]: List of (character_id_sequence, probability)
            tuples sorted by probability in descending order. Each sequence
            has length equal to len(mask_positions).
    """
    num_masks = len(mask_positions)
    if num_masks == 0:
        return []
    
    # Initialize beams: (predicted_token_ids, cumulative_probability)
    # Start with empty prediction and probability 1.0
    current_beams: List[Tuple[List[int], float]] = [([], 1.0)]
    
    # Process each mask position sequentially (left to right)
    for step, mask_pos in enumerate(mask_positions):
        await cancel.check_cancel_status(cancellation_event, task_id)
        
        all_candidates: List[Tuple[List[int], float]] = []
        
        # Expand each current beam
        for beam_ids, beam_prob in current_beams:
            # Create working copy and fill in previously predicted characters
            working_ids = token_ids.clone()
            for i, pred_id in enumerate(beam_ids):
                working_ids[0, mask_positions[i]] = pred_id
            
            # Run model inference
            with torch.no_grad():
                attention_mask = torch.ones_like(working_ids).to(device)
                outputs = model(input_ids=working_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Extract logits for current mask position and compute probabilities
            mask_logits = logits[0, mask_pos, :]
            probs = torch.nn.functional.softmax(mask_logits, dim=-1)
            
            # Get top-k predictions (extra to account for filtered special tokens)
            k = min(beam_size * 2, probs.shape[0])
            top_probs, top_indices = torch.topk(probs, k)
            
            # Generate candidate expansions
            for prob_val, token_id in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                token_id_int = int(token_id)
                
                # Skip special tokens (they shouldn't be predicted as characters)
                if token_id_int in special_token_ids:
                    continue
                
                # Create new beam by appending this character
                new_ids = beam_ids + [token_id_int]
                # Cumulative probability is product of individual probabilities
                new_prob = beam_prob * float(prob_val)
                all_candidates.append((new_ids, new_prob))
        
        # Prune: keep only top beam_size candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        current_beams = all_candidates[:beam_size]
        
        # Early termination if no valid candidates remain
        if not current_beams:
            break
    
    return current_beams


async def generate_span_predictions(
    token_ids: torch.Tensor,
    span_start_in_input: int,
    span_length: int,
    model: torch.nn.Module,
    char_itos: Dict[int, str],
    special_token_ids: Set[int],
    device: torch.device,
    num_preds: int,
    task_id: str,
    cancellation_event: asyncio.Event,
    beam_size: int = 20
) -> List[Tuple[str, float]]:
    """
    Generate span-level predictions for a sequence of consecutive [MASK] tokens.
    
    This is the main entry point for span prediction. Given a tensor containing
    a span of consecutive [MASK] tokens, it uses beam search to find the most
    probable character sequences and returns them as strings with probabilities.
    
    The function:
        1. Identifies the mask positions within the input tensor
        2. Runs beam search to find top character sequences
        3. Converts token IDs to characters and joins them into strings
        4. Removes duplicate predictions (keeping highest probability)
        5. Returns top num_preds unique predictions
    
    Parameters:
        token_ids (torch.Tensor): Input tensor containing the chunk with [MASK]
                                  tokens. Shape: (1, seq_len). Should include
                                  [CLS] at start and [SEP] at end.
        span_start_in_input (int): Position in token_ids where the span begins
                                   (accounting for [CLS] offset)
        span_length (int): Number of consecutive [MASK] tokens in the span
        model (torch.nn.Module): Character-level masked language model
        char_itos (Dict[int, str]): Mapping from token IDs to character strings
        special_token_ids (Set[int]): Set of special token IDs to exclude from
                                      predictions (e.g., [MASK], [PAD], [CLS],
                                      [SEP], [UNK])
        device (torch.device): Device for model inference
        num_preds (int): Number of top predictions to return
        task_id (str): Task identifier for logging and cancellation
        cancellation_event (asyncio.Event): Event for checking task cancellation
        beam_size (int): Number of beams for beam search. Higher values explore
                         more possibilities but increase computation. Default: 20
    
    Returns:
        List[Tuple[str, float]]: List of (predicted_span_string, probability)
            tuples, sorted by probability in descending order. Length is at
            most num_preds. Each string has length equal to span_length.
    
    Example:
        If span_length=3 and the model predicts "ing" as most likely:
        Returns: [("ing", 0.85), ("ion", 0.08), ("ive", 0.03), ...]
    """
    # Build list of mask positions in the input tensor
    mask_positions = list(range(span_start_in_input, span_start_in_input + span_length))
    
    # Execute beam search to find optimal character sequences
    beam_results = await _beam_search_chars(
        token_ids=token_ids,
        model=model,
        mask_positions=mask_positions,
        special_token_ids=special_token_ids,
        device=device,
        beam_size=beam_size,
        task_id=task_id,
        cancellation_event=cancellation_event
    )
    
    # Convert token ID sequences to character strings
    span_predictions: List[Tuple[str, float]] = []
    for char_ids, prob in beam_results:
        # Map each character ID to its string representation and concatenate
        span_str = ''.join(char_itos.get(cid, '?') for cid in char_ids)
        span_predictions.append((span_str, prob))
    
    # Remove duplicate strings, keeping the one with highest probability
    # (Duplicates can occur if beam search finds same sequence via different paths)
    unique_preds: Dict[str, float] = {}
    for span_str, prob in span_predictions:
        if span_str not in unique_preds:
            unique_preds[span_str] = prob
    
    # Sort by probability descending and return top predictions
    sorted_preds = sorted(unique_preds.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds[:num_preds]