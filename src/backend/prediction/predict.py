import torch
from collections import defaultdict
import logging
import numpy as np


def prediction_function(
    text: str,
    model,
    tokenizer,
    device,
    window_size: int = 512,
    overlap: int = 128,
    num_predictions: int = 5,
):
    """
    Masked language modeling inference for lacuna predictions using sliding window

    Parameters:
        text (str) -- input text
        model (str) -- encoder-only model
        tokenizer (str) -- tokenizer for model
        window_size (int) -- sliding window size
        overlap (int) -- sliding window overlap
        num_predictions (int) -- number of suggestions per word

    Returns
        final_predictions (dict) -- {mask_token_index_1: [(predicted_token_1, probability_score_1), ...], ...}
    """
    # set seed for reproducibiilty
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if text == "":
        raise ValueError("Input text cannot be empty.")

    all_predictions = defaultdict(list)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(tokens)

    # overlapping window loop to process text beyond 512 tokens
    for i in range(0, num_tokens, window_size - overlap):
        chunk_ids = tokens[i : min(i + window_size, num_tokens)]
        chunk_ids = chunk_ids[:512]
        chunk = tokenizer.decode(chunk_ids)
        chunk_inputs = tokenizer(
            chunk,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )

        chunk_inputs = {k: v.to(device) for k, v in chunk_inputs.items()}

        with torch.no_grad():
            outputs = model(**chunk_inputs)
            predictions = outputs.logits

        masked_indices = [
            i
            for i, token_id in enumerate(chunk_inputs["input_ids"][0])
            if token_id == tokenizer.mask_token_id
        ]
        logging.info(masked_indices)

        for masked_index in masked_indices:
            predicted_probs = predictions[0, masked_index]
            sorted_preds, sorted_idx = torch.sort(predicted_probs, descending=True)
            masked_predictions = []
            for k in range(num_predictions):
                predicted_index = int(sorted_idx[k].item())
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                probability = torch.softmax(predicted_probs, dim=-1)[
                    predicted_index
                ].item()
                masked_predictions.append((predicted_token, probability))
            logging.info(f"Predictions for {masked_index}: {masked_predictions}")
            all_predictions[masked_index + i].extend(masked_predictions)
    logging.info(f"All predictions: {all_predictions}")

    final_predictions = {}
    for masked_index, prediction_list in all_predictions.items():
        # group subword predictions
        subword_groups: dict = {}
        for token, prob in prediction_list:
            if token.startswith("##"):
                base_word = token[2:]  # remove "##" prefix
                if base_word not in subword_groups:
                    subword_groups[base_word] = []
                subword_groups[base_word].append((token, prob))
            else:  # whole word token
                subword_groups[token] = [(token, prob)]
        logging.info(f"Subword groups: {subword_groups}")

        whole_word_predictions = []
        for base_word, subword_list in subword_groups.items():
            max_prob = 0.0
            for subtoken, prob in subword_list:
                if prob > max_prob:
                    max_prob = prob

            whole_word_predictions.append((base_word, max_prob))

        # sort by prob
        sorted_predictions = sorted(
            whole_word_predictions, key=lambda x: x[1], reverse=True
        )
        # keep top num_predictions
        final_predictions[masked_index] = sorted_predictions[:num_predictions]

    logging.info(type(final_predictions))
    logging.info(f"Final predictions: {final_predictions}")

    return final_predictions
