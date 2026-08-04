import torch
import numpy as np
import asyncio
import logging
from typing import Callable, Coroutine, Any, Dict, List, Tuple
from . import cancel, hex_filter, blacklist
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

_LEADING_ELISION_MARKS = "\u02bc\u2019\u1fbd'"


OBELUS = "\u2020"
def _pseudo_prediction() -> List[Tuple[str, float]]:
    """
    Returns single-item list to signify no metrically accurate model predictions
    """
    return [(NO_HEX_PRED, 0.0)]


# convert list of sub-tokens into word
def _display_word(toks, tokenizer):
    is_latin_subword = getattr(tokenizer, "name_or_path", "") == "latincy/latin-bert"
    s = ''
    for i, tok_id in enumerate(toks):
        # convert tkn ID to string
        tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
        if not isinstance(tok, str): tok = str(tok)

        if is_latin_subword:
            ends_word = tok.endswith('_')
            if ends_word:
                # rmv '_'
                tok = tok[:-1]
            s += tok 
            if ends_word and i != len(toks) - 1:
                s += ' '

        # reconstruct words per '##' prefix
        else: 
            is_suffix = tok.startswith('##')
            if is_suffix:
                # rmv '##'
                tok = tok[2:]
            s += tok
    return s


# get top K predictions via logits
def _argkmax_beam(array, k, tokenizer, dim=1):
    array_cpu = array.cpu()
    _, topk_ids = torch.topk(array_cpu, k, dim=dim, largest=True)
    return topk_ids.squeeze(0)


def _get_n_predictions_batch(
    token_ids, model, tokenizer, n, masked_ind, prediction_inds_list, cur_probs,
    target_positions=None,
):
    if target_positions is None:
        target_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()

    batch_size = len(prediction_inds_list)
    batch_token_ids = token_ids.repeat(batch_size, 1)
    
    for i in range(batch_size):
        for j in range(len(prediction_inds_list[i])):
            batch_token_ids[i, target_positions[j]] = prediction_inds_list[i][j]

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
        new_prediction_inds = [prediction_inds_list[i] + [j] for j in suggestion_ids]
        all_candidates.extend(zip(new_prediction_inds, n_probs))

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
    Beam-search prediction for only [MASK]s in target_positions

    Parameters:

    
    Returns:
        final results ( list(tuple[int, float]) ) --
            [(id_1, prob_1), (id_n, prob_n), ...]
    """
    if target_positions is None:
        # prediction every [MASK] in input
        target_positions = (token_ids.detach().clone().squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()
    num_masked = len(target_positions)
    if num_masked == 0:
        return []
    
    # initial empty pred w/ prob = 1.0
    initial_prediction_inds = [[]]
    initial_probs = [1.0]

    # pred for 1st [MASK]
    cur_preds_tuples = _get_n_predictions_batch(
        token_ids.detach().clone(), model, tokenizer, beam_size, target_positions[0], initial_prediction_inds, initial_probs,
        target_positions=target_positions,
    )
    # cur_preds must be list of lists and tuples for next step
    cur_preds = [([item[0]], item[1]) for item in cur_preds_tuples]

    for i in range(num_masked - 1):
        await cancel.check_cancel_status(cancellation_event, task_id)
        
        prediction_inds_list = [pred[0][0] for pred in cur_preds]
        cur_probs = [pred[1] for pred in cur_preds]
        
        candidates_tuples = _get_n_predictions_batch(
            token_ids.detach().clone(), model, tokenizer, breadth, target_positions[i + 1], prediction_inds_list, cur_probs,
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

        is_latin_subword = getattr(tokenizer, "name_or_path", "") == "latincy/latin-bert"
        if is_latin_subword and getattr(tokenizer, "blacklist_ids", None) is None:
            tokenizer.blacklist_ids = blacklist.get_latin_blacklist_ids(tokenizer)

        for suggestion_ids, probability in sugs:
            if getattr(tokenizer, "blacklist_ids", None) and any(
                tid in tokenizer.blacklist_ids for tid in suggestion_ids
            ):
                continue
            candidate_word = _display_word(suggestion_ids, tokenizer)

            candidate_word = candidate_word.lstrip(_LEADING_ELISION_MARKS)
            if not candidate_word:
                continue
            overall_sugs.append((candidate_word, probability))

    # sort all 1, 2, 3-[MASK] preds by probs
    sorted_list = sorted(overall_sugs, key=lambda x: x[1], reverse=True)

    unique_preds: Dict[str, float] = {}
    for word, prob in sorted_list:
        unique_preds[word] = unique_preds.get(word, 0.0) + prob
    merged = sorted(unique_preds.items(), key=lambda x: x[1], reverse=True)

    limit = num_preds if pool_size is None else pool_size
    return merged[:limit]

"""
Hexameter filter
"""
def _fill_verse_blank_given_transmitted(
    text_segments: List[str],
    line_ordinals: List[int],
    sorted_keys: List[int],
    final_predictions: Dict[int, List[Tuple[str, float]]],
    target_ord: int,
    target_word: str,
) -> str:
    """
    Reconstruct a hex line w/ each [MASK] predicted

    Parameters:
        text_segments ( List[str] ) --
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
        rebuilt (str) -- reassembled hex line w/ [MASK]s predicted
    """
    rebuilt = text_segments[0]
    for j, ord_j in enumerate(line_ordinals):
        if ord_j == target_ord:
            prediction = target_word
        else:
            other_preds = final_predictions.get(sorted_keys[ord_j])
            prediction = other_preds[0][0] if other_preds else ""
        rebuilt += prediction + text_segments[j + 1]
    return rebuilt


def filter_predictions_hexameter(
    text: str,
    final_predictions: Dict[int, List[Tuple[str, float]]],
    tokenizer: Any,
    use_macronizer: bool = True,
    max_combos: int = 200,
    max_revet_iters: int = 3,
    num_preds: int = None,
    attested_predictions: Dict[int, List[str]] = None,
    max_obelus: int = 5,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Pass final_predictions through hex_filter to remove [MASK] predictions that don't fit hexameter rules defined in hex_filter
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

    def _line_scans(text_segments: List[str], predictions: List[str]) -> bool:
        rebuilt = text_segments[0]
        for j, prediction in enumerate(predictions):
            rebuilt += prediction + text_segments[j + 1]
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

        text_segments = line.split(mask_str)

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
                predictions = [""] * n_line_masks
                for slot, (word, _prob) in zip(participants, combo):
                    predictions[slot] = word
                if _line_scans(text_segments, predictions):
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
                        f"No metrical predictions for gap {key}. Flagging as '{NO_HEX_PRED}'"
                    )
        else:

            degraded: set = set()
            for _pass in range(max_revet_iters):
                changed = False
                for ti in participants:
                    key = keys[ti]
                    target_ord = line_ordinals[ti]
                    kept = [
                        (word, prob)
                        for word, prob in cand_lists[ti]
                        if hex_filter.line_matches_hexameter(
                            _fill_verse_blank_given_transmitted(
                                text_segments, line_ordinals, sorted_keys,
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
                    f"No metrical predictions for gap {key}. Flagging as '{NO_HEX_PRED}'"
                )

    if num_preds is not None:
        filtered = {k: v[:num_preds] for k, v in filtered.items()}

    for key in degraded_keys:
        recovery: List[Tuple[str, float]] = []
        attested = set((attested_predictions or {}).get(key, []))
        if attested:
            # in case filter rejects attested n-grams
            logging.warning(
                "Gap %d: scanner rejected ttested %s .\nReturning OBELUS-tagged candidates.",
                key, sorted(attested),
            )
            candidates = final_predictions.get(key, [])
            # first attestations, then rank by score
            ordered = sorted(
                candidates,
                key=lambda wp: (wp[0] not in attested, -wp[1]),
            )
            recovery = [
                (OBELUS + word, prob)
                for word, prob in ordered[:max_obelus]
            ]
        filtered[key] = _pseudo_prediction() + recovery

    return filtered


"""
Scansion display
"""

def _top_metrical_candidate_display(preds: List[Tuple[str, float]]) -> str:
    """
    Top-ranked prediction of gap for scansion display line reconstruction
    """
    if not preds:
        return ""
    if _is_pseudo_prediction(preds):
        if len(preds) > 1:
            return preds[1][0].lstrip(OBELUS)
        return ""
    return preds[0][0]


def restored_text_scansion(
    text: str,
    final_predictions: Dict[int, List[Tuple[str, float]]],
    tokenizer: Any,
    use_macronizer: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build  per-line scansion payload for frontend scansion display
    """
    if not final_predictions:
        return []

    mask_str = tokenizer.mask_token
    lines = text.split("\n")
    sorted_keys = sorted(final_predictions.keys())

    # same guard as filter_predictions_hexameter
    total_mask_literals = sum(line.count(mask_str) for line in lines)
    if total_mask_literals != len(sorted_keys):
        logging.warning(
            "Skipping scansion payload: %d '%s' tokens in text, %d predicted positions",
            total_mask_literals, mask_str, len(sorted_keys),
        )
        return []

    payload: List[Dict[str, Any]] = []
    mask_ordinal = 0
    for line in lines:
        n_line_masks = line.count(mask_str)
        text_segments = line.split(mask_str)

        restored_parts: List[Tuple[str, bool]] = [(text_segments[0], False)]
        for j in range(n_line_masks):
            key = sorted_keys[mask_ordinal + j]
            restored_parts.append(
                (_top_metrical_candidate_display(final_predictions.get(key, [])), True)
            )
            restored_parts.append((text_segments[j + 1], False))
        mask_ordinal += n_line_masks
        restored = "".join(piece for piece, _ in restored_parts)

        if not restored.strip():
            payload.append(
                {
                    "line": restored, "syllables": [], "markers": [],
                    "word_breaks": [], "segments": [],
                    "prediction_syllables": [],
                }
            )
            continue

        scan = hex_filter.scan_line_display(
            restored, use_macronizer=use_macronizer,
        )

        skeleton_flags: List[bool] = []
        for piece, is_prediction in restored_parts:
            skeleton_flags.extend(
                [is_prediction] * len(hex_filter._base_letter_skeleton(piece))
            )

        unit_flags: List[bool] = []
        pos = 0
        for unit_len in scan.unit_skeleton_lens:
            unit_flags.append(any(skeleton_flags[pos:pos + unit_len]))
            pos += unit_len
        if pos != len(skeleton_flags):
            # fallback: display scansion w/out highlighting rather than misattribute preds
            logging.warning(
                "Scansion prediction alignment mismatch on line %r (%d syllable "
                "letters vs %d line letters); omitting prediction highlighting",
                restored, pos, len(skeleton_flags),
            )
            unit_flags = [False] * len(scan.unit_skeleton_lens)

        # per-syllable character segments
        segments: List[List[List[Any]]] = []
        prediction_syllables: List[int] = []
        for i, composition in enumerate(scan.syllable_units):
            syllable_segments: List[List[Any]] = []
            for unit_text, ordinal in composition:
                flag = unit_flags[ordinal]
                if syllable_segments and syllable_segments[-1][1] == flag:
                    syllable_segments[-1][0] += unit_text
                else:
                    syllable_segments.append([unit_text, flag])
            if any(flag for _chars, flag in syllable_segments):
                prediction_syllables.append(i)
            segments.append(syllable_segments)

        payload.append(
            {
                "line": restored,
                "syllables": scan.syllables,
                "markers": scan.markers,
                "word_breaks": scan.word_breaks,
                "segments": segments,
                "prediction_syllables": prediction_syllables,
            }
        )
    return payload


"""
Pseudo-log-likelihood (PLL) rescoring of filtered predictions
"""

def _is_pseudo_prediction(preds: List[Tuple[str, float]]) -> bool:
    """
    True iff preds is the NO_HEX_PRED sentinel list from _pseudo_prediction()
    """
    return bool(preds) and preds[0][0] == NO_HEX_PRED


def _word_l2r_mask_span(toks: List[str], pos: int, n: int, is_latin_subword: bool) -> int:
    """
    End index (exclusive) of within-word subword span starting at pos
    """
    j = pos + 1
    if is_latin_subword:
        # keep consuming while the PREVIOUS token has not closed the word
        while j < n - 1 and not toks[j - 1].endswith('_'):
            j += 1
    else:
        while j < n - 1 and toks[j].startswith('##'):
            j += 1
    return j


@torch.no_grad()
def _pll_score_line(
    line_text: str,
    model: torch.nn.Module,
    tokenizer: Any,
    batch_size: int = 64,
) -> float:
    """
    Length-normalised pseudo-log-likelihood (PLL) of whole verse line
    """
    token_ids = tokenizer.encode(line_text, add_special_tokens=True)
    n = len(token_ids)

    if n <= 2 or n > tokenizer.model_max_length:
        return float("-inf")

    ids = torch.tensor(token_ids)
    toks = tokenizer.convert_ids_to_tokens(token_ids)
    is_latin_subword = getattr(tokenizer, "name_or_path", "") == "latincy/latin-bert"

    # positions 1 .. n-2: skip special tkns at the edges
    positions = list(range(1, n - 1))

    # one masked copy per scored position (word-l2r span masking)
    masked_inputs = []
    for pos in positions:
        row = ids.clone()
        end = _word_l2r_mask_span(toks, pos, n, is_latin_subword)
        row[pos:end] = tokenizer.mask_token_id
        masked_inputs.append(row)

    total_log_prob = 0.0
    for start in range(0, len(masked_inputs), batch_size):
        batch = torch.stack(masked_inputs[start:start + batch_size]).to(model.device)
        logits = model(batch).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        for i, pos in enumerate(positions[start:start + batch_size]):
            total_log_prob += log_probs[i, pos, token_ids[pos]].item()

    return total_log_prob / len(positions)


def rescore_predictions_pll(
    text: str,
    final_predictions: Dict[int, List[Tuple[str, float]]],
    model: torch.nn.Module,
    tokenizer: Any,
    num_preds: int = None,
    batch_size: int = 64,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Re-rank each gap's metrical predictions by PLL of whole line
    """
    if not final_predictions:
        return final_predictions

    mask_str = tokenizer.mask_token
    lines = text.split("\n")
    sorted_keys = sorted(final_predictions.keys())

    # same guard as filter_predictions_hexameter
    total_mask_literals = sum(line.count(mask_str) for line in lines)
    if total_mask_literals != len(sorted_keys):
        logging.warning(
            "Skipping PLL rescoring: %d '%s' tokens in text, %d predicted positions",
            total_mask_literals, mask_str, len(sorted_keys),
        )
        if num_preds is not None:
            return {k: v[:num_preds] for k, v in final_predictions.items()}
        return final_predictions

    context_preds = {
        k: ([] if _is_pseudo_prediction(v) else v)
        for k, v in final_predictions.items()
    }

    rescored = dict(final_predictions)
    # handle idnetical predicted lines
    line_score_cache: Dict[str, float] = {}

    mask_ordinal = 0
    for line in lines:
        n_line_masks = line.count(mask_str)
        if n_line_masks == 0:
            continue

        line_ordinals = list(range(mask_ordinal, mask_ordinal + n_line_masks))
        mask_ordinal += n_line_masks
        text_segments = line.split(mask_str)
        keys = [sorted_keys[o] for o in line_ordinals]

        for ti, key in enumerate(keys):
            cand_list = final_predictions[key]
            if not cand_list or _is_pseudo_prediction(cand_list):
                continue

            scored = []
            for word, _prob in cand_list:
                predicted = _fill_verse_blank_given_transmitted(
                    text_segments, line_ordinals, sorted_keys,
                    context_preds, line_ordinals[ti], word,
                )
                if predicted not in line_score_cache:
                    line_score_cache[predicted] = _pll_score_line(
                        predicted, model, tokenizer, batch_size=batch_size,
                    )
                scored.append((word, float(np.exp(line_score_cache[predicted]))))

            scored.sort(key=lambda x: x[1], reverse=True)
            rescored[key] = scored

    if num_preds is not None:
        rescored = {
            k: (v if _is_pseudo_prediction(v) else v[:num_preds])
            for k, v in rescored.items()
        }

    return rescored


"""
Formula-index candidate augmentation
"""

def _formula_collation_key(word: str) -> str:
    """
    Normalised comparison for duplicate detection beam-search v concordance formulae
    """
    import unicodedata
    return unicodedata.normalize("NFC", word).casefold().replace(" ", "")


def _formula_deaccented_key(word: str) -> str:
    """
    Diacritic-stripped key for beam-search v concordance cross-ref
    """
    import unicodedata
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", word)
        if not unicodedata.combining(c)
    )
    return _formula_collation_key(stripped)


def formular_concordance_lookup(
    text: str,
    final_predictions: Dict[int, List[Tuple[str, float]]],
    tokenizer: Any,
    concordance: Any,
    stratum_weights: Dict[str, float] = None,
    max_attested_suggestions: int = 10,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Issert corpus-attested fomrulae into gap candidate pool before metrical filtering

    Parameters:
        text (str) -- input text w/ [MASK]s
        final_predictions ( Dict[int, List[Tuple[str, float]]] ) --
            pred dict from prediction_function()
        tokenizer () -- model tokenizer (for the [MASK] literal)
        concordance ( formular_concordance.FormularConcordance ) -- built/loaded concordance
        stratum_weights ( Dict[str, float] ) -- stratum -> interpolation
            weight (see formular_concordance.stratum_weights); None = uniform
            over all strata in the concordance
        max_attested_suggestions (int) -- max predictions appended per gap
    """
    if not final_predictions or concordance is None:
        return final_predictions, {}

    if stratum_weights is None:
        names = list(getattr(concordance, "strata", {}).keys())
        if not names:
            return final_predictions, {}
        stratum_weights = {n: 1.0 / len(names) for n in names}

    mask_str = tokenizer.mask_token
    lines = text.split("\n")
    sorted_keys = sorted(final_predictions.keys())

    # same guard as filter_predictions_hexameter
    total_mask_literals = sum(line.count(mask_str) for line in lines)
    if total_mask_literals != len(sorted_keys):
        logging.warning(
            "Skipping formula injection: %d '%s' tokens in text, %d predicted positions",
            total_mask_literals, mask_str, len(sorted_keys),
        )
        return final_predictions, {}

    augmented = dict(final_predictions)
    attested: Dict[int, List[str]] = {}

    mask_ordinal = 0
    for line in lines:
        n_line_masks = line.count(mask_str)
        if n_line_masks == 0:
            continue

        line_ordinals = list(range(mask_ordinal, mask_ordinal + n_line_masks))
        mask_ordinal += n_line_masks
        text_segments = line.split(mask_str)

        for ti, ord_i in enumerate(line_ordinals):
            key = sorted_keys[ord_i]

            # context words: only from the text_segments IMMEDIATELY
            # adjacent to this mask -- words beyond a neighbouring
            # [MASK] are separated by unknown material
            pre_lacuna = text_segments[ti].split()
            post_lacuna = text_segments[ti + 1].split()

            touch_start = (ti == 0 and not pre_lacuna)
            touch_end = (ti == n_line_masks - 1 and not post_lacuna)

            if not pre_lacuna and not post_lacuna:
                continue

            attested_suggestions = concordance.query(
                pre_lacuna=pre_lacuna,
                post_lacuna=post_lacuna,
                line_initial=touch_start,
                line_final=touch_end,
                stratum_weights=stratum_weights,
                num_attestations=max_attested_suggestions,
            )
            if not attested_suggestions:
                continue

            # provenance-aware dedup:
            pool = list(augmented[key])
            beam_by_deaccented = {}
            for pos, (w, _p) in enumerate(pool):
                beam_by_deaccented.setdefault(_formula_deaccented_key(w), pos)
            attested_seen = set()
            upgraded_positions = set()
            upgraded = []
            newly_attested = []
            for prediction, score in attested_suggestions:
                sensitive_key = _formula_collation_key(prediction)
                if sensitive_key in attested_seen:
                    continue
                attested_seen.add(sensitive_key)
                deaccented_key = _formula_deaccented_key(prediction)
                pos = beam_by_deaccented.get(deaccented_key)
                if pos is not None:
                    if pool[pos][0] == prediction:
                        continue
                    if pos in upgraded_positions:
                        newly_attested.append((prediction, score))
                        continue
                    pool[pos] = (prediction, pool[pos][1])
                    upgraded_positions.add(pos)
                    upgraded.append(prediction)
                    continue
                newly_attested.append((prediction, score))
            if newly_attested or upgraded:
                if upgraded:
                    logging.info(
                        "Formula orthographic upgrade for gap %d: %s",
                        key, upgraded,
                    )
                if newly_attested:
                    logging.info(
                        "Formula attestation for gap %d: %s",
                        key, newly_attested,
                    )
                augmented[key] = pool + newly_attested
                attested[key] = upgraded + [
                    word for word, _score in newly_attested
                ]

    return augmented, attested
