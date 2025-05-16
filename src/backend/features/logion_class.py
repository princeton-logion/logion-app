import torch
import numpy as np
from polyleven import levenshtein
import asyncio
from typing import Callable, Coroutine, Any
from . import cancel

"""
Chance-confidence algorithm created by
Charlie Cowen-Breen and Creston Brooks
"""

# type hint for callback
ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

class Logion:

    def __init__(self, Model, Tokenizer, Levenshtein_Filter, Device):
        # set seed for reproducibility
        seed_value = 42
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

        self.Model = Model
        self.Tokenizer = Tokenizer
        self.lev_filter = Levenshtein_Filter
        self.device = Device
        self.sm = torch.nn.Softmax(dim=1)
        torch.set_grad_enabled(False)


    def _get_chance_probability(self, token_ids, index_of_id_to_mask):
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
                    torch.kthvalue(-array, i, dim=dim).indices.cpu().numpy().tolist()
                )
        else:
            indices = []
            i = 1
            while len(indices) < k:
                index = (
                    torch.kthvalue(-array, i, dim=dim).indices.cpu().numpy().tolist()
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
        indices = []
        new_prefixes = []
        added = 0
        ind = 1

        while added < k:
            if ind > len(array[0]):
                break
            array = array.cpu()
            val = torch.kthvalue(-array, ind, dim=dim).indices.numpy().tolist()
            if prefix != "":
                cur_tok = self.Tokenizer.convert_ids_to_tokens([val[0]])[0].replace(
                    "##", ""
                )
                trunc_prefix = prefix[: min(len(prefix), len(cur_tok))]
                if not cur_tok.startswith(trunc_prefix):
                    ind += 1
                    continue
            else:
                cur_tok = ""
            indices.append(val)
            if len(cur_tok) >= len(prefix):
                new_prefixes.append("")
            else:
                new_prefixes.append(prefix[len(cur_tok) :])
            ind += 1
            added += 1

        return torch.tensor(indices), new_prefixes

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

    async def _beam_search(
        self,
        token_ids,
        beam_size,
        task_id: str,
        cancellation_event: asyncio.Event,
        prefix='',
        breadth=100
    ):

        mask_positions = (token_ids.detach().clone().squeeze() == self.Tokenizer.mask_token_id).nonzero().flatten().tolist()
        num_masked = len(mask_positions)

        cur_preds = self._get_n_predictions(token_ids.detach().clone(), beam_size, prefix, mask_positions[0], [])

        for i in range(num_masked - 1):
            
            await cancel.check_cancel_status(cancellation_event, task_id)

            candidates = []
            for j in range(len(cur_preds)):
                await cancel.check_cancel_status(cancellation_event, task_id)
                candidates += self._get_n_predictions(
                    token_ids.detach().clone(),
                    breadth,
                    cur_preds[j][2],
                    mask_positions[i + 1],
                    cur_preds[j][0],
                    cur_preds[j][1],
                )
            candidates.sort(key=lambda k: k[1], reverse=True)
            if i != num_masked - 2:
                cur_preds = candidates[:beam_size]
            else:
                cur_preds = candidates[:breadth]

        return cur_preds

    def _suggest_filtered(self, tokens, ground_token_id, filter):
        probs = self._get_mask_probabilities(tokens).cpu().squeeze()
        filtered_probs = probs * filter[ground_token_id]
        suggestion = self._argkmax(filtered_probs, 1)
    
        return suggestion, probs[suggestion].item()


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

        filtered_suggestions = {'?': 0.0}

        for num_masks in range(1, max_masks + 1):
            await cancel.check_cancel_status(cancellation_event, task_id)
            if num_masks == 1 and len(transmitted_tokens) == 1:

                sug, prob = self._suggest_filtered(tokens.clone(), transmitted_tokens[0], self.lev_filter)
                word = self._display_word(self.Tokenizer.convert_ids_to_tokens(sug))
                filtered_suggestions[word] = prob
                continue

            if no_beam:
                continue

            text = (
                pre_text
                + f"".join([f"{self.Tokenizer.mask_token}"] * num_masks)
                + post_text
            )
            beam_tokens = self.Tokenizer.encode(text, return_tensors="pt").to(self.device)

            sugs = await self._beam_search(
                beam_tokens,
                depth if num_masks>1 or len(transmitted_tokens)>1 else len(self.Tokenizer.vocab),
                breadth=breadth if num_masks>1 or len(transmitted_tokens)>1 else len(self.Tokenizer.vocab),
                task_id=task_id,
                cancellation_event=cancellation_event
            )

            # process beam search results
            for suggestion_ids, probability, _ in sugs:
                await cancel.check_cancel_status(cancellation_event, task_id)

                converted_tokens = self.Tokenizer.convert_ids_to_tokens(suggestion_ids)
                word = self._display_word(converted_tokens)

                word_str = str(word)
                transmitted_text_str = str(transmitted_text)
                d = levenshtein(word_str, transmitted_text_str, max_lev)

                # apply Lev filter
                if d > max_lev or d < min_lev:
                    continue

                # update sugs if better prob
                if (word not in filtered_suggestions) or (
                    word in filtered_suggestions and filtered_suggestions[word] < probability):
                    filtered_suggestions[word] = probability
                    break

        sorted_filtered_suggestions = sorted(filtered_suggestions.items(), key=lambda x: x[1], reverse=True)

        # return only top sug
        # if none better than ? return default
        if len(sorted_filtered_suggestions) > 1 and sorted_filtered_suggestions[0][0] != '?':
            return sorted_filtered_suggestions[:1]
        elif len(sorted_filtered_suggestions) == 1 and sorted_filtered_suggestions[0][0] != '?':
             return sorted_filtered_suggestions[:1]
        else:
             return [('?', 0.0)]


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
