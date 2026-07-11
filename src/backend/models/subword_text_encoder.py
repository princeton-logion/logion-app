import sys
import unicodedata

_ESCAPE_CHARS = set(u"\\_u;0123456789")

_ALPHANUMERIC_CHAR_SET = set(
    chr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(chr(i)).startswith("L") or
        unicodedata.category(chr(i)).startswith("N"))
)

def _tokenizer_encode(text):
    """Split a unicode string into alternating alnum / non-alnum tokens."""
    if not text:
        return []
    ret = []
    token_start = 0
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != " " or token_start == 0:
                ret.append(token)
            token_start = pos
    ret.append(text[token_start:])
    return ret


def _escape_token(token, alphabet):
    """Escape underscores/OOV chars and append the trailing '_' word marker."""
    token = token.replace("\\", "\\\\").replace("_", "\\u")
    ret = [c if c in alphabet and c != "\n" else r"\%d;" % ord(c) for c in token]
    return "".join(ret) + "_"


class SubwordTextEncoder:
    """Inference-only reimplementation of t2t's SubwordTextEncoder.

    Loads a vocab file produced by the original tensor2tensor trainer and
    reproduces its greedy longest-match subword encoding exactly.
    """

    def __init__(self, filename):
        self._alphabet = set()
        self.filename = filename
        self._load_from_file(filename)

    def encode(self, s):
        """Converts a native string to a list of subtoken ids."""
        return self._tokens_to_subtoken_ids(_tokenizer_encode(s))

    @property
    def vocab_size(self):
        return len(self._all_subtoken_strings)

    def _tokens_to_subtoken_ids(self, tokens):
        ret = []
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        return ret

    def _token_to_subtoken_ids(self, token):
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value
        ret = self._escaped_token_to_subtoken_ids(
            _escape_token(token, self._alphabet))
        self._cache[cache_location] = (token, ret)
        return ret

    def _escaped_token_to_subtoken_strings(self, escaped_token):
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in range(min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._subtoken_string_to_id:
                    ret.append(subtoken)
                    start = end
                    break
            else:
                raise ValueError(
                    "Token substring '%s' not found in subtoken vocabulary." % escaped_token)
        return ret

    def _escaped_token_to_subtoken_ids(self, escaped_token):
        return [
            self._subtoken_string_to_id[subtoken]
            for subtoken in self._escaped_token_to_subtoken_strings(escaped_token)
        ]

    def _init_subtokens_from_list(self, subtoken_strings, reserved_tokens=None):
        reserved_tokens = reserved_tokens or []
        if reserved_tokens:
            self._all_subtoken_strings = reserved_tokens + subtoken_strings
        else:
            self._all_subtoken_strings = subtoken_strings
        self._max_subtoken_len = max(len(s) for s in subtoken_strings)
        self._subtoken_string_to_id = {
            s: i + len(reserved_tokens)
            for i, s in enumerate(subtoken_strings) if s
        }
        self._cache_size = 2 ** 20
        self._cache = [(None, None)] * self._cache_size

    def _init_alphabet_from_tokens(self, tokens):
        self._alphabet = {c for token in tokens for c in token}
        self._alphabet |= _ESCAPE_CHARS

    def _load_from_file_object(self, f):
        subtoken_strings = []
        for line in f:
            s = line.rstrip("\n")
            if ((s.startswith("'") and s.endswith("'")) or
                    (s.startswith('"') and s.endswith('"'))):
                s = s[1:-1]
            subtoken_strings.append(s)
        self._init_subtokens_from_list(subtoken_strings)
        self._init_alphabet_from_tokens(subtoken_strings)

    def _load_from_file(self, filename):
        with open(filename, encoding="utf-8") as f:
            self._load_from_file_object(f)