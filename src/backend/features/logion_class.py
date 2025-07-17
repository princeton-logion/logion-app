import torch
import numpy as np
from polyleven import levenshtein
import asyncio
from typing import Callable, Coroutine, Any
from . import cancel


# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]


class Logion:

    def __init__(self, Model, Tokenizer, Device):
        # set seed for reproducibility
        seed_value = 42
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

        self.Model = Model
        self.Tokenizer = Tokenizer
        #self.lev_filter = Levenshtein_Filter
        self.device = Device
        self.sm = torch.nn.Softmax(dim=1)
        torch.set_grad_enabled(False)

        self.blacklist = {
    14: '.', 12: ',', 26: ':', 27: ';', 31: '?', 5: '!', 8: '(', 9: ')', 
    58: '·', 62: '»', 54: '«', 6: '\"', 7: '\'', 10: '*', 11: '+', 13: '-',
    81: 'α', 82: 'β', 83: 'γ', 84: 'δ', 85: 'ε', 
    86: 'ζ', 87: 'η', 88: 'θ', 89: 'ι', 90: 'κ', 91: 'λ', 92: 'μ', 93: 'ν', 
    94: 'ξ', 95: 'ο', 96: 'π', 97: 'ρ', 98: 'ς', 99: 'σ', 100: 'τ', 
    101: 'υ', 102: 'φ', 103: 'χ', 104: 'ψ', 105: 'ω', 1: '[UNK]', 16: '0',
    17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9'
}
        self.blacklist_ids = set(self.blacklist.keys())
        self.blacklist_chars = set(self.blacklist.values())

    
    def _get_chance_probability(
            self,
            token_ids,
            index_of_id_to_mask):
        
        underlying_token_id = token_ids[0, index_of_id_to_mask].item()
        token_ids[0, index_of_id_to_mask] = self.Tokenizer.mask_token_id
        logits = self.Model(token_ids).logits
        mask_logits = logits[:, index_of_id_to_mask, :]
        probabilities = self.sm(mask_logits).flatten()
        token_ids[0, index_of_id_to_mask] = underlying_token_id

        return probabilities, underlying_token_id

    async def get_chance_scores(
            self,
            token_ids,
            task_id: str,
            progress_callback: ProgressCallback,
            cancellation_event: asyncio.Event,
            base_progress: float,
            progress_range: float,
            relevant_token_ids=None):
        
        num_tokens = token_ids.shape[1]
        scores = []

        await cancel.check_cancel_status(cancellation_event, task_id)

        for index_of_id_to_mask in range(num_tokens):
            
            await cancel.check_cancel_status(cancellation_event, task_id)
            internal_progress = (index_of_id_to_mask + 1) / num_tokens
            current_overall_progress = base_progress + (internal_progress * progress_range)
            await progress_callback(current_overall_progress, f"Chance score for ({index_of_id_to_mask + 1}/{num_tokens})...")

            underlying_token_id = token_ids[0, index_of_id_to_mask].item()

            if not relevant_token_ids or underlying_token_id in relevant_token_ids:
                probabilities, underlying_token_id = self._get_chance_probability(
                    token_ids, index_of_id_to_mask
                )
                scores.append(probabilities[underlying_token_id].item())
            else:
                scores.append(-1)

        return scores

    
    def _get_mask_probabilities(self, masked_text_ids):
        mask_positions = (
            (masked_text_ids.squeeze() == self.Tokenizer.mask_token_id)
            .nonzero()
            .flatten()
            .tolist()
        )
        logits = self.Model(masked_text_ids).logits.squeeze(0)
        mask_logits = logits[mask_positions]
        probabilities = self.sm(mask_logits)
        return probabilities
    

    
    def _argkmax(self, array, k, dim=0, prefix=None):
        if not prefix:
            indices = []
            for i in range(1, k + 1):
                indices.append(
                    torch.kthvalue(-array, i, dim=dim).indices.tolist()
                )
        else:
            indices = []
            i = 1
            while len(indices) < k:
                index = (
                    torch.kthvalue(-array, i, dim=dim).indices.tolist()
                )
                if self.Tokenizer.convert_ids_to_tokens(index)[0].startswith(prefix):
                    indices.append(index)
                i += 1
        return torch.tensor(indices)
    

    
    def display_sentence(self, toks):
        s = ""
        first_tok = True

        for tok in toks:
            if tok.startswith("##"):
                tok = tok[2:]
            elif tok in ["´", ",", ".", ";"]:
                pass
            elif first_tok:
                first_tok = False
            else:
                tok = " " + tok

            s += tok
        return s
    

    
    def _argkmax_beam(self, array, k, prefix="", dim=1):
        array_cpu = array.cpu()

        topk_vals, topk_ids = torch.topk(array_cpu, k, dim=dim, largest=True)

        filtered_ids = []
        new_prefixes = []
        for tok_id in topk_ids.squeeze().tolist():
            tok_str = self.Tokenizer.convert_ids_to_tokens([tok_id])[0].lstrip("##")
            if prefix == "" or tok_str.startswith(prefix):
                filtered_ids.append(tok_id)
                new_prefixes.append(
                    prefix[len(tok_str):] if len(tok_str) < len(prefix) else ""
                )
            if len(filtered_ids) == k:
                break

        return torch.tensor(filtered_ids, dtype=torch.long), new_prefixes
    

    
    def _get_n_predictions(
        self, token_ids, n, prefix, masked_ind, fill_inds, cur_prob=1
    ):
        mask_positions = (
            (token_ids.squeeze() == self.Tokenizer.mask_token_id)
            .nonzero()
            .flatten()
            .tolist()
        )

        for i in range(len(fill_inds)):
            token_ids.squeeze()[mask_positions[i]] = fill_inds[i]

        model_id = min(len(mask_positions) - len(fill_inds) - 1, 4)
        logits = self.Model(token_ids).logits.squeeze(0)
        mask_logits = logits[[[masked_ind]]]
        probabilities = self.sm(mask_logits)
        arg1, prefixes = self._argkmax_beam(probabilities, n, prefix, dim=1)
        suggestion_ids = arg1.squeeze().tolist()
        n_probs = probabilities.squeeze()[suggestion_ids]
        n_probs = torch.mul(n_probs, cur_prob).tolist()
        new_fill_inds = [fill_inds + [i] for i in suggestion_ids]
        return tuple(zip(new_fill_inds, n_probs, prefixes))

    def _get_n_predictions_batch(
        self, token_ids, n, prefixes, masked_ind, fill_inds_list, cur_probs
    ):
        
        mask_positions = (
            (token_ids.squeeze() == self.Tokenizer.mask_token_id)
            .nonzero()
            .flatten()
            .tolist()
        )

        # create batch of token_ids w/ filled positions
        batch_size = len(fill_inds_list)
        batch_token_ids = token_ids.repeat(batch_size, 1)
        
        for i in range(batch_size):
            for j in range(len(fill_inds_list[i])):
                batch_token_ids[i, mask_positions[j]] = fill_inds_list[i][j]

        # get predictions for all items in batch
        logits = self.Model(batch_token_ids).logits
        mask_logits = logits[:, masked_ind]
        probabilities = self.sm(mask_logits)

        # process each item in batch
        all_candidates = []
        for i in range(batch_size):
            arg1, new_prefixes = self._argkmax_beam(probabilities[i:i+1], n, prefixes[i], dim=1)
            suggestion_ids = arg1.squeeze().tolist()
            n_probs = probabilities[i, suggestion_ids]
            n_probs = torch.mul(n_probs, cur_probs[i]).tolist()
            new_fill_inds = [fill_inds_list[i] + [j] for j in suggestion_ids]
            all_candidates.extend(zip(new_fill_inds, n_probs, new_prefixes))

        return all_candidates

    async def _beam_search(
        self,
        token_ids,
        beam_size,
        task_id: str,
        cancellation_event: asyncio.Event,
        prefix='',
        breadth=100,
    ):
        
        mask_positions = (token_ids.detach().clone().squeeze() == self.Tokenizer.mask_token_id).nonzero().flatten().tolist()
        num_masked = len(mask_positions)

        # get initial predictions
        cur_preds = self._get_n_predictions(token_ids.detach().clone(), beam_size, prefix, mask_positions[0], [])

        # process remaining positions in batches
        for i in range(num_masked - 1):
            await cancel.check_cancel_status(cancellation_event, task_id)
            # prepare batch input
            fill_inds_list = [pred[0] for pred in cur_preds]
            prefixes = [pred[2] for pred in cur_preds]
            cur_probs = [pred[1] for pred in cur_preds]
            
            # get predictions for all items in batch
            candidates = self._get_n_predictions_batch(
                token_ids.detach().clone(),
                breadth,
                prefixes,
                mask_positions[i + 1],
                fill_inds_list,
                cur_probs
            )
            
            # sort and select top candidates
            candidates.sort(key=lambda k: k[1], reverse=True)
            if i != num_masked - 2:
                cur_preds = candidates[:beam_size]
            else:
                cur_preds = candidates[:breadth]

        return cur_preds
    

    
    # def _suggest_filtered(self,
    #                       tokens,
    #                       ground_token_id,
    #                       filter):
    #     probs = self._get_mask_probabilities(tokens).cpu().squeeze()
    #     filtered_probs = probs * filter[ground_token_id]
    #     suggestion = self._argkmax(filtered_probs, 1)
    #     return suggestion, probs[suggestion].item()
    

    
    async def fill_the_blank_given_transmitted(
        self,
        pre_text,
        post_text,
        transmitted_text,
        transmitted_tokens,
        tokens,
        task_id: str,
        progress_callback: ProgressCallback,
        cancellation_event: asyncio.Event,
        max_masks=3,
        depth=20,
        breadth=20,
        max_lev=1,
        min_lev=0,
        no_beam=False
    ):

        if len(transmitted_tokens) == 1 and transmitted_tokens[0] in self.blacklist_ids:
            return [(transmitted_text, 0.0)]

        best_suggestion = (transmitted_text, 0.0)

        if no_beam:
            return [best_suggestion]
    
        # iterate [MASK] nums, 1->max
        for num_masks in range(1, max_masks + 1):
            await cancel.check_cancel_status(cancellation_event, task_id)
    
            # determine number of [MASK]s for blank
            text_to_mask = (
                pre_text
                + f"".join([f"{self.Tokenizer.mask_token}"] * num_masks)
                + post_text
            )
            beam_tokens = self.Tokenizer.encode(text_to_mask, return_tensors="pt").to(self.device)
    
            sugs = await self._beam_search(
                beam_tokens,
                beam_size=depth,
                breadth=breadth,
                task_id=task_id,
                cancellation_event=cancellation_event
            )
    
            # filter search results via lev dist
            for suggestion_ids, probability, _ in sugs:
                converted_tokens = self.Tokenizer.convert_ids_to_tokens(suggestion_ids)
                candidate_word = self._display_word(converted_tokens)

                # calculate lev dist w/out matrix for candidate words
                dist = levenshtein(candidate_word, transmitted_text, max_lev)
                if min_lev <= dist <= max_lev:
                    if probability > best_suggestion[1]:
                        best_suggestion = (candidate_word, probability)

        # if none better than orig or blacklisted, return orig
        filtered_suggestion = best_suggestion[0]
        if (filtered_suggestion != transmitted_text and filtered_suggestion not in self.blacklist_chars):
            return [best_suggestion]
        else:
            return [(transmitted_text, 0.0)]
    

    
    def _display_word(self, toks):
        s = ''
        first_tok = True
        for tok in toks:
            if not isinstance(tok, str): tok = str(tok)
            is_suffix = tok.startswith('##')
            if is_suffix: tok = "" + tok[2:]
            elif not first_tok:
                pass
            s += tok
            first_tok = False
        return s 