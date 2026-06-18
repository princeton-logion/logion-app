import torch
import numpy as np
import asyncio
import logging
from typing import Callable, Coroutine, Any, Dict, List, Tuple
from . import cancel, hex_filter
from itertools import product

"""
Helper functions for gap-filling with beam search
"""

# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

# set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# default pred value if hex_filter removes all model preds
NO_HEX_PRED = "omnia contra metrum"
def _pseudo_prediction() -> List[Tuple[str, float]]:
    """
    Returns single-item list to signify no metrically accurate model predictions
    """
    return [(NO_HEX_PRED, 0.0)]


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
            s += ' '
        s += tok
        first_tok = False
    return s


# get top K predictions via logits
def _argkmax_beam(array, k, tokenizer, dim=1):
    array_cpu = array.cpu()
    _, topk_ids = torch.topk(array_cpu, k, dim=dim, largest=True)
    return topk_ids.squeeze(0)


def _get_n_predictions_batch(
    token_ids, model, tokenizer, n, masked_ind, fill_inds_list, cur_probs,
    target_positions=None,
):
    if target_positions is None:
        target_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()

    batch_size = len(fill_inds_list)
    batch_token_ids = token_ids.repeat(batch_size, 1)
    
    for i in range(batch_size):
        for j in range(len(fill_inds_list[i])):
            batch_token_ids[i, target_positions[j]] = fill_inds_list[i][j]

    logits = model(batch_token_ids.to(model.device)).logits
    mask_logits = logits[:, masked_ind]
    probabilities = torch.nn.functional.softmax(mask_logits, dim=-1)

    # batch processing
    all_candidates = []
    for i in range(batch_size):
        suggestion_ids_tensor = _argkmax_beam(probabilities[i:i+1], n, tokenizer, dim=1)
        suggestion_ids = suggestion_ids_tensor.tolist()
        
        if not isinstance(suggestion_ids, list):
            suggestion_ids = [suggestion_ids]

        n_probs_tensor = probabilities[i, suggestion_ids]
        n_probs = torch.mul(n_probs_tensor, cur_probs[i]).tolist()
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
    breadth: int = 100,
    target_positions: List[int] = None,
):
    """
    Beam-search fill for only [MASK]s in target_positions

    Parameters:

    
    Returns:
        final results ( list(tuple[int, float]) ) --
            [(id_1, prob_1), (id_n, prob_n), ...]
    """
    if target_positions is None:
        # fill every [MASK] in input
        target_positions = (token_ids.detach().clone().squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()
    num_masked = len(target_positions)
    if num_masked == 0:
        return []
    
    # initial empty pred w/ prob = 1.0
    initial_fill_inds = [[]]
    initial_probs = [1.0]

    # pred for 1st [MASK]
    cur_preds_tuples = _get_n_predictions_batch(
        token_ids.detach().clone(), model, tokenizer, beam_size, target_positions[0], initial_fill_inds, initial_probs,
        target_positions=target_positions,
    )
    # cur_preds must be list of lists and tuples for next step
    cur_preds = [([item[0]], item[1]) for item in cur_preds_tuples]

    for i in range(num_masked - 1):
        await cancel.check_cancel_status(cancellation_event, task_id)
        
        fill_inds_list = [pred[0][0] for pred in cur_preds]
        cur_probs = [pred[1] for pred in cur_preds]
        
        candidates_tuples = _get_n_predictions_batch(
            token_ids.detach().clone(), model, tokenizer, breadth, target_positions[i + 1], fill_inds_list, cur_probs,
            target_positions=target_positions,
        )
        
        candidates = [(c[0], c[1]) for c in candidates_tuples]
        # to find highest ranked across beams, sort preds by prob
        candidates.sort(key=lambda k: k[1], reverse=True)
        top_candidates = candidates[:beam_size]
        cur_preds = [([ids], prob) for ids, prob in top_candidates]

    final_results = []
    for ids, prob in cur_preds:
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
    beam_size: int = 20,
    pool_size: int = None,
) -> list[tuple[str, float]]:
    """
    For each [MASK], beam search retrieves highest ranked multi-token preds

    Parameters:
        pool_size (int) -- num of top candidates to return; defaults to
            num_preds

    Returns:
        list[tuple[str, float]]
    """
    overall_sugs = []
    original_token_list = input_token_ids.squeeze().tolist()

    # replace orig [MASK] w/ 1, 2, 3 [MASK] tkns
    for num_masks in range(1, max_tokens + 1):
        temp_token_list = (
            original_token_list[:mask_idx_in_chunk]
            + [tokenizer.mask_token_id] * num_masks
            + original_token_list[mask_idx_in_chunk + 1:]
        )
        
        if len(temp_token_list) > tokenizer.model_max_length:
            continue
        # [MASK] sequences -> tensor of tkn IDs for beam search
        beam_search_input = torch.tensor([temp_token_list]).to(model.device)

        target_positions = list(range(mask_idx_in_chunk, mask_idx_in_chunk + num_masks))

        # beam search for only current gap [MASK]s
        sugs = await _beam_search(
            beam_search_input,
            model,
            tokenizer,
            beam_size=beam_size,
            breadth=num_preds,
            task_id=task_id,
            target_positions=target_positions,
            cancellation_event=cancellation_event,
        )

        for suggestion_ids, probability in sugs:
            candidate_word = _display_word(suggestion_ids, tokenizer)
            overall_sugs.append((candidate_word, probability))

    # sort all 1, 2, 3-[MASK] preds by probs
    sorted_list = sorted(overall_sugs, key=lambda x: x[1], reverse=True)

    # rmv duplicates, keep copy w/ highest prob
    unique_preds = {}
    for word, prob in sorted_list:
        if word not in unique_preds:
            unique_preds[word] = prob
    
    limit = num_preds if pool_size is None else pool_size
    return list(unique_preds.items())[:limit]

"""
Hexameter filter
"""
def _fill_verse_line(
    fragments: List[str],
    line_ordinals: List[int],
    sorted_keys: List[int],
    final_predictions: Dict[int, List[Tuple[str, float]]],
    target_ord: int,
    target_word: str,
) -> str:
    """
    Reconstruct a hex line w/ each [MASK] filled

    Parameters:
        fragments ( List[str] ) --
            lines text pre- and post-[MASK]
        line_ordinals ( List[int] ) --
            global ordinals (left->right) for all [MASK]s in line
        sorted_keys ( List[int] ) --
            [MASK] indices, key from final_predictions
        final_predictions ( Dict[int, List[Tuple[str, float]]] ) --
            pred dict from prediction_function()
        target_ord (int) --
            global ordinal for current [MASK]
         target_word (str) -- predicted word for target [MASK]

    Returns:
        rebuilt (str) -- reassembled hex line w/ [MASK]s filled
    """
    rebuilt = fragments[0]
    for j, ord_j in enumerate(line_ordinals):
        if ord_j == target_ord:
            fill = target_word
        else:
            other_preds = final_predictions.get(sorted_keys[ord_j])
            fill = other_preds[0][0] if other_preds else ""
        rebuilt += fill + fragments[j + 1]
    return rebuilt


def filter_predictions_hexameter(
    text: str,
    final_predictions: Dict[int, List[Tuple[str, float]]],
    tokenizer: Any,
    use_macronizer: bool = True,
    max_combos: int = 200,
    max_revet_passes: int = 3,
    num_preds: int = None,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Pass final_predictions through hex_filter to remove [MASK] predictions that don't fit hexameter rules defined in hex_filter. Hexameter lines demarcated by \n in raw input txt.

    Parameters:
        text (str) -- input text w/ [MASK]s
        final_predictions ( Dict[int, List[Tuple[str, float]]] ) --
            pred dict from prediction_function()
        tokenizer () -- model tokenizer
        use_macronizer (bool) -- resolve ambiguous vowels (α, ι, υ) w/ grc_macronizer?
        max_combos (int) -- per-line max exhaustive metrical vetting
        max_revet_passes (int) -- max per-line re-vetting iterations
        num_preds (int) -- max suggestions per word

    Returns:
        filtered ( Dict[int, List[Tuple[str, float]]] ) --
            identical to final_predictions, sans non-metrical preds

        N.b. if hex_filter rmvs all preds, returns "NO_HEX_PRED" value
    """

    if not final_predictions:
        return final_predictions

    mask_str = tokenizer.mask_token
    lines = text.split("\n")

    sorted_keys = sorted(final_predictions.keys())

    total_mask_literals = sum(line.count(mask_str) for line in lines)
    if total_mask_literals != len(sorted_keys):
        logging.warning(
            "Skipping hexameter filter",
            total_mask_literals, mask_str, len(sorted_keys),
        )
        if num_preds is not None:
            return {k: v[:num_preds] for k, v in final_predictions.items()}
        return final_predictions

    filtered = dict(final_predictions)

    degraded_keys = set()

    def _line_scans(fragments: List[str], fills: List[str]) -> bool:
        rebuilt = fragments[0]
        for j, fill in enumerate(fills):
            rebuilt += fill + fragments[j + 1]
        return hex_filter.line_matches_hexameter(
            rebuilt, use_macronizer=use_macronizer
        )

    mask_ordinal = 0
    for line in lines:
        n_line_masks = line.count(mask_str)
        if n_line_masks == 0:
            continue

        line_ordinals = list(range(mask_ordinal, mask_ordinal + n_line_masks))
        mask_ordinal += n_line_masks

        fragments = line.split(mask_str)

        keys = [sorted_keys[o] for o in line_ordinals]
        cand_lists = [final_predictions[k] for k in keys]

        participants = [i for i, c in enumerate(cand_lists) if c]
        if not participants:
            continue

        n_combos = 1
        for i in participants:
            n_combos *= len(cand_lists[i])
            if n_combos > max_combos:
                break

        if n_combos <= max_combos:

            valid_words: Dict[int, set] = {i: set() for i in participants}
            for combo in product(*(cand_lists[i] for i in participants)):
                fills = [""] * n_line_masks
                for slot, (word, _prob) in zip(participants, combo):
                    fills[slot] = word
                if _line_scans(fragments, fills):
                    for slot, (word, _prob) in zip(participants, combo):
                        valid_words[slot].add(word)

            for slot in participants:
                key = keys[slot]
                kept = [(w, p) for (w, p) in cand_lists[slot]
                        if w in valid_words[slot]]
                if kept:
                    filtered[key] = kept
                else:
                    degraded_keys.add(key)
                    logging.warning(
                        f"No metrical predictions for gap {key}. Flagging as '{NO_HEX_PRED}'",
                        key,
                    )
        else:

            degraded: set = set()
            for _pass in range(max_revet_passes):
                changed = False
                for ti in participants:
                    key = keys[ti]
                    target_ord = line_ordinals[ti]
                    kept = [
                        (word, prob)
                        for word, prob in cand_lists[ti]
                        if hex_filter.line_matches_hexameter(
                            _fill_verse_line(
                                fragments, line_ordinals, sorted_keys,
                                filtered, target_ord, word,
                            ),
                            use_macronizer=use_macronizer,
                        )
                    ]
                    if kept:
                        degraded.discard(key)
                        new_list = kept
                    else:
                        degraded.add(key)
                        new_list = cand_lists[ti]
                    if ([w for w, _ in new_list]
                            != [w for w, _ in filtered[key]]):
                        filtered[key] = new_list
                        changed = True
                if not changed:
                    break

            degraded_keys.update(degraded)
            for key in sorted(degraded):
                logging.warning(
                    f"No metrical predictions for gap {key}. Flagging as '{NO_HEX_PRED}'",
                    key,
                )

    if num_preds is not None:
        filtered = {k: v[:num_preds] for k, v in filtered.items()}

    for key in degraded_keys:
        filtered[key] = _pseudo_prediction()

    return filtered