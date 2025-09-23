
import torch
import numpy as np
import asyncio
from typing import Callable, Coroutine, Any
from . import cancel, blacklist

"""
Helper functions for gap-filling with beam search
"""

# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

# set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)


# convert list of sub-tokens into word
def _display_word(toks, tokenizer):
    s = ''
    first_tok = True
    for tok_id in toks:
        # convert tkn ID to string
        tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
        if not isinstance(tok, str): tok = str(tok)

        # reconstruct words per '##' prefix
        is_suffix = tok.startswith('##')
        if is_suffix:
            # rmv '##'
            tok = tok[2:]
        elif not first_tok:
            pass
        s += tok
        first_tok = False
    return s


# get top K predictions via logits
def _argkmax_beam(array, k, tokenizer, dim=1):
    array_cpu = array.cpu()
    _, topk_ids = torch.topk(array_cpu, k, dim=dim, largest=True)
    return topk_ids.squeeze(0)


def _get_n_predictions_batch(
    token_ids, model, tokenizer, n, masked_ind, fill_inds_list, cur_probs
):
    # retrieve all [MASK]s from input
    mask_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()

    # create batch of token_ids from original set of token_ids
    batch_size = len(fill_inds_list)
    batch_token_ids = token_ids.repeat(batch_size, 1)
    
    # for each text sequence inside batch, fill preceding predicted [MASK] tkns
    for i in range(batch_size):
        for j in range(len(fill_inds_list[i])):
            batch_token_ids[i, mask_positions[j]] = fill_inds_list[i][j]

    # get remaining [MASK] pred for all text sequences in batch
    logits = model(batch_token_ids.to(model.device)).logits
    # focus on logits for current [MASK] under consideration
    mask_logits = logits[:, masked_ind]
    probabilities = torch.nn.functional.softmax(mask_logits, dim=-1)

    # process batch
    all_candidates = []
    # retrieve top "n" tkns for each [MASK] sequence
    for i in range(batch_size):
        # top "n" tkn ID predictions
        suggestion_ids_tensor = _argkmax_beam(probabilities[i:i+1], n, tokenizer, dim=1)
        suggestion_ids = suggestion_ids_tensor.tolist()
        
        # for handling if only one suggestion returned
        if not isinstance(suggestion_ids, list):
            suggestion_ids = [suggestion_ids]

        # retrieve probs of top "n" tkns
        n_probs_tensor = probabilities[i, suggestion_ids]
        # calculate total prob for each sequence
        n_probs = torch.mul(n_probs_tensor, cur_probs[i]).tolist()
        # append new tkn ID for new sequence
        new_fill_inds = [fill_inds_list[i] + [j] for j in suggestion_ids]
        all_candidates.extend(zip(new_fill_inds, n_probs))

    return all_candidates



async def _beam_search(
    token_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: Any,
    beam_size: int,
    task_id: str,
    cancellation_event: asyncio.Event,
    breadth: int = 100
):
    # retrieve all [MASK]s from input
    mask_positions = (token_ids.detach().clone().squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()
    num_masked = len(mask_positions)
    if num_masked == 0:
        return []
    
    # initial empty pred w/ prob = 1.0
    initial_fill_inds = [[]]
    initial_probs = [1.0]

    # prediction for first [MASK]
    cur_preds_tuples = _get_n_predictions_batch(
        token_ids.detach().clone(), model, tokenizer, beam_size, mask_positions[0], initial_fill_inds, initial_probs
    )
    # cur_preds needs to be list of lists and tuples for next step
    cur_preds = [([item[0]], item[1]) for item in cur_preds_tuples]

    # predict remaining [MASK]
    for i in range(num_masked - 1):
        await cancel.check_cancel_status(cancellation_event, task_id)
        
        fill_inds_list = [pred[0][0] for pred in cur_preds] # pred[0] is the list of lists of ids
        cur_probs = [pred[1] for pred in cur_preds]
        
        # use batch processing to predict all potential sequences
        candidates_tuples = _get_n_predictions_batch(
            token_ids.detach().clone(), model, tokenizer, breadth, mask_positions[i + 1], fill_inds_list, cur_probs
        )
        
        candidates = [(c[0], c[1]) for c in candidates_tuples]
        # to find highest ranked sequences across beams, sort candidates by prob
        candidates.sort(key=lambda k: k[1], reverse=True)
        
        # keep only highest beam_size sequences
        top_candidates = candidates[:beam_size]
        cur_preds = [([ids], prob) for ids, prob in top_candidates]


    # prepare final result in list-tuple format: [(ids, probability)]
    final_results = []
    for ids, prob in cur_preds:
        # flatten when nested list
        final_results.append((ids[0], prob))
    final_results.sort(key=lambda k: k[1], reverse=True)
    return final_results

async def generate_multi_token_suggestions(
    input_token_ids: torch.Tensor,
    mask_idx_in_chunk: int,
    model: torch.nn.Module,
    tokenizer: Any,
    num_preds: int,
    task_id: str,
    cancellation_event: asyncio.Event,
    max_tokens: int = 3,
    beam_size: int = 20
) -> list[tuple[str, float]]:
    """
    For each single [MASK] token in input, use beam search to retrieve most probable multi-token predictions

    Parameters:


    Returns:
        list[tuple[str, float]]
    """
    overall_sugs = []
    original_token_list = input_token_ids.squeeze().tolist()

    # replace original single [MASK] with 1, 2, 3 [MASK] tkns
    for num_masks in range(1, max_tokens + 1):
        # create new tkn list w/ 'num_masks' [MASK]s replacing original [MASK]
        temp_token_list = (
            original_token_list[:mask_idx_in_chunk]
            + [tokenizer.mask_token_id] * num_masks
            + original_token_list[mask_idx_in_chunk + 1:]
        )
        
        # truncate if sequence exceeds model limits
        # (check with main pred_function for handling 512+)
        if len(temp_token_list) > tokenizer.model_max_length:
            continue
        # prepare [MASK] sequences as tensor of tkn IDs for beam search
        beam_search_input = torch.tensor([temp_token_list]).to(model.device)

        # beam search to fill [MASK]s in input
        sugs = await _beam_search(
            beam_search_input,
            model,
            tokenizer,
            beam_size=beam_size,
            breadth=num_preds,
            task_id=task_id,
            cancellation_event=cancellation_event,
        )

        # save retrieved tkn IDs as strings
        for suggestion_ids, probability in sugs:
            candidate_word = _display_word(suggestion_ids, tokenizer)
            overall_sugs.append((candidate_word, probability))

    # sort suggestions by probability for all runs (1-, 2-, and 3-[MASK])
    sorted_list = sorted(overall_sugs, key=lambda x: x[1], reverse=True)
    
    # rmv duplicate strings, keep copy w/ highest prob
    unique_preds = {}
    for word, prob in sorted_list:
        if word not in unique_preds:
            unique_preds[word] = prob
    
    # return top num_preds predictions as list of tuples
    return list(unique_preds.items())[:num_preds]