import re
import string

_VALID_SINGLE_CHAR_WORDS = {"a", "e", "i", "o"}
_SPECIAL_TOKENS = re.compile(r"(\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\])")

def _is_blacklisted_subtoken(raw_key: str) -> bool:
    """
    True if a raw t2t vocab entry (before the +5 special-token offset) decodes
    to punctuation, a digit, or a single character that isn't a real word on its own.
    """
    ends_word = raw_key.endswith("_")
    if ends_word:
        raw_key = raw_key[:-1]
    if not raw_key:
        return True
    if len(raw_key) == 1:
        return raw_key.lower() not in _VALID_SINGLE_CHAR_WORDS
    if all(ch in string.punctuation or ch.isdigit() for ch in raw_key):
        return True
    return False

class LatinTokenizer:
    def __init__(self, encoder):
        self.vocab = {}
        self.reverseVocab = {}
        self.encoder = encoder

        self.vocab["[PAD]"] = 0
        self.vocab["[UNK]"] = 1
        self.vocab["[CLS]"] = 2
        self.vocab["[SEP]"] = 3
        self.vocab["[MASK]"] = 4

        self.blacklist_ids = {0, 1} 
        for key in self.encoder._subtoken_string_to_id:
            token_id = self.encoder._subtoken_string_to_id[key] + 5
            self.vocab[key] = token_id
            self.reverseVocab[token_id] = key
            if _is_blacklisted_subtoken(key):
                self.blacklist_ids.add(token_id)

    def convert_tokens_to_ids(self, tokens):
        wp_tokens = []
        for token in tokens:
            if token == "[PAD]":
                wp_tokens.append(0)
            elif token == "[UNK]":
                wp_tokens.append(1)
            elif token == "[CLS]":
                wp_tokens.append(2)
            elif token == "[SEP]":
                wp_tokens.append(3)
            elif token == "[MASK]":
                wp_tokens.append(4)
            else:
                wp_tokens.append(self.vocab[token])
        return wp_tokens

    def tokenize(self, text):
        tokens = text.split(" ")
        wp_tokens = []
        for token in tokens:
            if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
                wp_tokens.append(token)
            else:
                wp_toks = self.encoder.encode(token)
                for wp in wp_toks:
                    wp_tokens.append(self.reverseVocab[wp + 5])
        return wp_tokens


class LatinTokenizerAdapter(LatinTokenizer):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        text = _SPECIAL_TOKENS.sub(r" \1 ", text)
        words = text.split(" ")
        toks = []
        for w in words:
            if not w:
                continue
            if w in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
                toks.append(w)
            else:
                toks.extend(self.tokenize(w.lower()))
        return self.convert_tokens_to_ids(toks)

    def convert_ids_to_tokens(self, ids):
        special = {0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]", 4: "[MASK]"}
        return [special.get(i, self.reverseVocab.get(i, "[UNK]")) for i in ids]