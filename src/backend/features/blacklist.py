import string

greek_blacklist = {
    # punctuation and special characters
    14: '.', 12: ',', 26: ':', 27: ';', 31: '?', 5: '!', 8: '(', 9: ')', 
    58: '·', 62: '»', 54: '«', 6: '\"', 7: '\'', 10: '*', 11: '+', 13: '-',
    28: '<', 29: '=', 30: '>', 32: '[', 33: '\\', 34: ']', 35: '`', 41: '{',
    42: '|', 43: '}', 44: '~',
    # latin characters
    36: 'a', 37: 'e', 38: 'i', 39: 'o', 40: 'u',
    # greek single characters (sans articles)
    82: 'β', 83: 'γ', 84: 'δ', 86: 'ζ', 88: 'θ', 89: 'ι', 90: 'κ', 91: 'λ',
    92: 'μ', 93: 'ν', 94: 'ξ', 96: 'π', 97: 'ρ', 98: 'ς', 99: 'σ', 100: 'τ',
    101: 'υ', 102: 'φ', 103: 'χ', 104: 'ψ',
    # [UNK]
    1: '[UNK]',
    # digits 
    16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7',
    24: '8', 25: '9'
}

latin_whitelist = {'a', 'e', 'i', 'o'}

def is_latin_blacklisted(raw_token: str) -> bool:
    if raw_token.endswith("_"):
        raw_token = raw_token[:-1]
    if not raw_token:
        return True
    if len(raw_token) == 1:
        return raw_token.lower() not in latin_whitelist
    if all(ch in string.punctuation or ch.isdigit() for ch in raw_token):
        return True
    return False

def get_latin_blacklist_ids(tokenizer) -> set:
    return {
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if is_latin_blacklisted(token)
    }