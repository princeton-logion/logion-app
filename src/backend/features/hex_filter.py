"""
Hexameter filter for word-level MLM predictions

Based on hexameter scansion described in Barbara Graziosi & Johannes Haubold (eds.), Homer: Iliad, Book VI, Cambridge Greek and Latin Classics, Cambridge UP, 2015.

This module validates predicted sub/word spans per premodern Greek hexameter conventions.  After a model generates predictions, hex_filter confirms whether each prediction yields a metrically valid line and removes metrically invalid predictions.


grc_macronizer:
    optionally use grc_macronizer (https://github.com/Urdatorn/grc-macronizer) to scan ambiguous vowels (α, ι, υ).  Sans grc_macronizer, filter scans those vowels as flexible (X) for higher recall

Disclaimers:
    - assumes prehandling of crasis (kagw etc. arrive pre-fused in text)
    - elision: written elision (') handled by input text; *unwritten* elision
      (e.g. unelided MLM fills at gap junctions) tried as alternative
      readings via _elision_candidates
    - synizesis: word-internal V.V merges tried as alternative readings
    - correption (cross-word only): scans flexible (X)
    - muta cum liquida (plosive + liquid/nasal onset): scans flexible (X)
"""

import unicodedata
import logging
import re
from functools import lru_cache
from typing import Dict, List, NamedTuple, Tuple, Set, Optional
from collections import defaultdict

try:
    from grc_macronizer import Macronizer as _MacronizerCls
    HAS_MACRONIZER = True
except ImportError:
    _MacronizerCls = None
    HAS_MACRONIZER = False
    logging.info(
        "grc_macronizer not installed. Will use permissive heuristics for ambiguous vowels."
    )


_macronizer = None

_odycy_load_patched = False

USE_MACRONIZER = False


def set_macronizer_enabled(enabled: bool) -> None:
    global USE_MACRONIZER
    USE_MACRONIZER = bool(enabled)


"""
phonological constants
"""
LONG_VOWELS: frozenset = frozenset("ηω")
SHORT_VOWELS: frozenset = frozenset("εο")
AMBIGUOUS_VOWELS: frozenset = frozenset("αιυ")
ALL_VOWELS: frozenset = LONG_VOWELS | SHORT_VOWELS | AMBIGUOUS_VOWELS

DIPHTHONG_LIST: Tuple[str, ...] = (
    "αι", "αυ", "ει", "ευ", "οι", "ου", "ηυ", "υι",
)
DIPHTHONG_SET: frozenset = frozenset(DIPHTHONG_LIST)

CONSONANTS: frozenset = frozenset("βγδζθκλμνξπρσςτφχψ\u03DD")

PLOSIVES: frozenset = frozenset("πβφτδθκγχ")

LIQUIDS_NASALS: frozenset = frozenset("λρμν")

DRAWN_OUT_SONORANTS: frozenset = LIQUIDS_NASALS | frozenset("σ")

DOUBLE_CONSONANTS: frozenset = frozenset("ζξψ")

def _normalize_digamma_entry(w: str) -> str:
    return w.replace("\u03c2", "\u03c3")


DIGAMMA_WORDS: frozenset = frozenset(map(_normalize_digamma_entry, {
    # \u03dd\u1f79\u03c2/\u1f11\u1f79\u03c2 (possessive)
    "\u03bf\u03bd", "\u03b7\u03bd", "\u03c9\u03bd", "\u03bf\u03c2", "\u03bf\u03b9\u03c3\u03b9", "\u03bf\u03b9\u03c2",
    "\u03b5\u03bf\u03bd", "\u03b5\u03b7\u03bd", "\u03b5\u03b7\u03c2", "\u03b5\u03c9", "\u03b5\u03b7", "\u03b5\u03c9\u03bd", "\u03b5\u03b1", "\u03b5\u03b1\u03c2", "\u03b5\u03bf\u03b9\u03bf",
    # \u03dd\u1f71\u03bd\u03b1\u03be
    "\u03b1\u03bd\u03b1\u03be", "\u03b1\u03bd\u03b1\u03ba\u03c4\u03bf\u03c2", "\u03b1\u03bd\u03b1\u03ba\u03c4\u03b9", "\u03b1\u03bd\u03b1\u03ba\u03c4\u03b1", "\u03b1\u03bd\u03b1\u03ba\u03c4\u03b5\u03c2", "\u03b1\u03bd\u03b1\u03ba\u03c4\u03c9\u03bd",
    # \u03dd\u1f71\u03c3\u03c4\u03c5
    "\u03b1\u03c3\u03c4\u03c5", "\u03b1\u03c3\u03c4\u03b5\u03bf\u03c2", "\u03b1\u03c3\u03c4\u03b5\u03b9", "\u03b1\u03c3\u03c4\u03b5\u03b1",
    # \u03dd\u03b5\u1fd6\u03b4\u03bf\u03c2 / \u03dd\u03b5\u1f77\u03b4\u03bf\u03bc\u03b1\u03b9
    "\u03b5\u03b9\u03b4\u03bf\u03c2", "\u03b5\u03b9\u03b4\u03b5\u03b1", "\u03b5\u03b9\u03b4\u03c9\u03c2", "\u03b5\u03b9\u03b4\u03bf\u03c4\u03b5\u03c2",
    # \u03dd\u03b5\u1fd6\u03c0\u03bf\u03bd / \u03dd\u03b5\u1f76\u03c0\u03b5\u1fd6\u03bd
    "\u03b5\u03b9\u03c0\u03bf\u03bd", "\u03b5\u03b9\u03c0\u03b5", "\u03b5\u03b9\u03c0\u03b5\u03bd", "\u03b5\u03b9\u03c0\u03b5\u03c2", "\u03b5\u03b9\u03c0\u03c9", "\u03b5\u03b9\u03c0\u03b7\u03c2", "\u03b5\u03b9\u03c0\u03b7",
    "\u03b5\u03b9\u03c0\u03b5\u03b9\u03bd", "\u03b5\u03b9\u03c0\u03c9\u03bd", "\u03b5\u03b9\u03c0\u03bf\u03c5\u03c3\u03b1", "\u03b5\u03b9\u03c0\u03bf\u03b9",
    # \u03dd\u03b5\u03ba\u03ce\u03bd, \u03dd\u03b5\u03ba\u03b1\u03c2, \u03dd\u03b5\u03ba\u03b1\u03c3\u03c4-, \u03dd\u03b5\u03ba\u03b7\u03b2\u03bf\u03bb-
    "\u03b5\u03ba\u03c9\u03bd", "\u03b5\u03ba\u03bf\u03c5\u03c3\u03b1", "\u03b5\u03ba\u03b7\u03c4\u03b9", "\u03b5\u03ba\u03b1\u03c2",
    "\u03b5\u03ba\u03b1\u03c3\u03c4\u03bf\u03c2", "\u03b5\u03ba\u03b1\u03c3\u03c4\u03bf\u03bd", "\u03b5\u03ba\u03b1\u03c3\u03c4\u03bf\u03c5", "\u03b5\u03ba\u03b1\u03c3\u03c4\u03c9", "\u03b5\u03ba\u03b1\u03c3\u03c4\u03bf\u03b9", "\u03b5\u03ba\u03b1\u03c3\u03c4\u03b1",
    "\u03b5\u03ba\u03b1\u03c3\u03c4\u03b7", "\u03b5\u03ba\u03b1\u03c3\u03c4\u03b7\u03bd", "\u03b5\u03ba\u03b1\u03c3\u03c4\u03b7\u03c2",
    "\u03b5\u03ba\u03b7\u03b2\u03bf\u03bb\u03bf\u03c2", "\u03b5\u03ba\u03b7\u03b2\u03bf\u03bb\u03bf\u03c5", "\u03b5\u03ba\u03b7\u03b2\u03bf\u03bb\u03c9", "\u03b5\u03ba\u03b7\u03b2\u03bf\u03bb\u03bf\u03bd",
    # \u03dd\u1f73\u03c0\u03bf\u03c2
    "\u03b5\u03c0\u03bf\u03c2", "\u03b5\u03c0\u03b5\u03b1", "\u03b5\u03c0\u03b5\u03c3\u03c3\u03b9", "\u03b5\u03c0\u03b5\u03c9\u03bd", "\u03b5\u03c0\u03b5\u03b5\u03c3\u03c3\u03b9",
    # NB "\u03b5\u03c0\u03b5\u03b9" (dat. of \u03dd\u1f73\u03c0\u03bf\u03c2) deliberately omitted: bare
    # skeleton collides w/ the digamma-less conjunction \u1f10\u03c0\u03b5\u1f77
    # \u03dd\u03b5\u03c1\u1f73\u03c9 (fut. of \u03b5\u1f34\u03c1\u03c9, "speak")
    "\u03b5\u03c1\u03b5\u03c9", "\u03b5\u03c1\u03b5\u03b5\u03b9\u03c2", "\u03b5\u03c1\u03b5\u03b5\u03b9", "\u03b5\u03c1\u03b5\u03b5\u03b9\u03bd", "\u03b5\u03c1\u03b5\u03bf\u03c5\u03c3\u03b9", "\u03b5\u03c1\u03b5\u03bf\u03bd\u03c4\u03b5\u03c2",
    # \u03dd\u1f73\u03c1\u03b3\u03bf\u03bd
    "\u03b5\u03c1\u03b3\u03bf\u03bd", "\u03b5\u03c1\u03b3\u03b1", "\u03b5\u03c1\u03b3\u03c9\u03bd", "\u03b5\u03c1\u03b3\u03c9", "\u03b5\u03c1\u03b3\u03bf\u03c5", "\u03b5\u03c1\u03b3\u03bf\u03b9\u03c3\u03b9", "\u03b5\u03c1\u03b3\u03bf\u03b9\u03c2",
    # \u03dd\u1f73\u03c3\u03c0\u03b5\u03c1\u03bf\u03c2
    "\u03b5\u03c3\u03c0\u03b5\u03c1\u03bf\u03c2", "\u03b5\u03c3\u03c0\u03b5\u03c1\u03b9\u03bf\u03c2",
    # \u03dd\u1f73\u03c4\u03bf\u03c2
    "\u03b5\u03c4\u03bf\u03c2", "\u03b5\u03c4\u03b5\u03b1", "\u03b5\u03c4\u03b5\u03c9\u03bd",
    # \u03dd\u03b9\u03b4\u03b5\u1fd6\u03bd / \u03dd\u03b5\u1fd6\u03b4\u03bf\u03bd
    "\u03b9\u03b4\u03b5\u03b9\u03bd", "\u03b9\u03b4\u03c9\u03bd", "\u03b9\u03b4\u03bf\u03c5\u03c3\u03b1", "\u03b9\u03b4\u03bf\u03bd\u03c4\u03b5\u03c2", "\u03b9\u03b4\u03b5\u03bd", "\u03b9\u03b4\u03b5", "\u03b9\u03b4\u03b5\u03c3\u03b8\u03b1\u03b9",
    "\u03b5\u03b9\u03b4\u03bf\u03bd", "\u03b5\u03b9\u03b4\u03b5",
    # \u03dd\u1fd6\u03c3\u03bf\u03c2
    "\u03b9\u03c3\u03bf\u03c2", "\u03b9\u03c3\u03bf\u03bd", "\u03b9\u03c3\u03b7", "\u03b9\u03c3\u03b7\u03bd", "\u03b9\u03c3\u03b1", "\u03b9\u03c3\u03bf\u03b9",
    # \u03dd\u1fd6\u03c6\u03b9
    "\u03b9\u03c6\u03b9",
    # \u03dd\u03bf\u1fd6\u03b4\u03b1
    "\u03bf\u03b9\u03b4\u03b1", "\u03bf\u03b9\u03b4\u03b5", "\u03b9\u03c3\u03bc\u03b5\u03bd", "\u03b9\u03b4\u03bc\u03b5\u03bd",
    # \u03dd\u03bf\u1fd6\u03ba\u03bf\u03c2
    "\u03bf\u03b9\u03ba\u03bf\u03c2", "\u03bf\u03b9\u03ba\u03bf\u03bd", "\u03bf\u03b9\u03ba\u03bf\u03c5", "\u03bf\u03b9\u03ba\u03c9", "\u03bf\u03b9\u03ba\u03bf\u03b9", "\u03bf\u03b9\u03ba\u03b1\u03b4\u03b5", "\u03bf\u03b9\u03ba\u03bf\u03bd\u03b4\u03b5",
    # \u03dd\u03bf\u1fd6\u03bd\u03bf\u03c2
    "\u03bf\u03b9\u03bd\u03bf\u03c2", "\u03bf\u03b9\u03bd\u03bf\u03bd", "\u03bf\u03b9\u03bd\u03bf\u03c5", "\u03bf\u03b9\u03bd\u03c9", "\u03bf\u03b9\u03bd\u03bf\u03b9\u03bf", "\u03bf\u03b9\u03bd\u03bf\u03b9\u03c3\u03b9",
    # omit monosyllabic pronouns for collision safety 
}))
# scansion marks: L=long, S=short, X=flexible
L = "L"
S = "S"
X = "X"


"""
Unicode helpers
"""
def _base_char(char: str) -> str:
    nfd = unicodedata.normalize("NFD", char)
    return nfd[0].lower() if nfd else char.lower()

def _has_iota_subscript(char: str) -> bool:
    return "\u0345" in unicodedata.normalize("NFD", char)

def _has_macron(char: str) -> bool:
    return "\u0304" in unicodedata.normalize("NFD", char)

def _has_breve(char: str) -> bool:
    return "\u0306" in unicodedata.normalize("NFD", char)

def _has_circumflex(char: str) -> bool:
    return "\u0342" in unicodedata.normalize("NFD", char)

def _has_diaeresis(char: str) -> bool:
    return "\u0308" in unicodedata.normalize("NFD", char)


"""
grc_macronizer helpers

    -currently obsolete, keep for testing grc_macronizer integration
    -currently bypass grc_macronizer for initial test and higher recall
"""

def _patch_odycy_loader_to_memoize():
    global _odycy_load_patched
    if _odycy_load_patched:
        return
    try:
        import grc_odycy_joint_trf
    except Exception as exc:
        logging.warning(
            "Cannot import grc_odycy_joint_trf.", exc
        )
        return

    _original_load = grc_odycy_joint_trf.load
    _nlp_holder = {}

    def _memoized_load(*args, **kwargs):
        if "nlp" not in _nlp_holder:
            _nlp_holder["nlp"] = _original_load(*args, **kwargs)
        return _nlp_holder["nlp"]

    grc_odycy_joint_trf.load = _memoized_load
    _odycy_load_patched = True


def _get_macronizer():
    global _macronizer
    if _macronizer is None and HAS_MACRONIZER:
        try:
            import spacy
            spacy.require_cpu()
        except Exception as exc:
            logging.warning("Cannot pin odyCy to CPU.", exc)
        _patch_odycy_loader_to_memoize()
        _macronizer = _MacronizerCls()
    return _macronizer


@lru_cache(maxsize=4096)
def _macronize_cached(text: str) -> Optional[str]:
    macronizer = _get_macronizer()
    if macronizer is None:
        return None
    try:
        import torch
        with torch.inference_mode():
            return macronizer.macronize(text)
    except Exception as exc:
        logging.debug("Macroniser failed on input: %s", exc)
        return None


def macronize(text: str) -> Optional[str]:
    if not (HAS_MACRONIZER and USE_MACRONIZER):
        return None
    return _macronize_cached(text)


"""
Phonology parser
"""

class _PhoneticUnit:
    """
    minimal phonological unit, vowel/diphthong (V) or consonant (C)

    Attributes:
        kind (str) -- V == vowel/diphthong, C == consonant
        text (str) -- orig characters w/ diacritics/accents
        base (str) -- characters sans diacritics/accents
        is_diphthong (bool) -- is unit diphthong?
        has_iota_sub (bool) -- does vowel have iota subscript?
        has_macron (bool) -- does vowel have macron? (per grc_macronizer)
        has_breve (bool) -- does vowel have breve? (per grc_macronizer)
        is_double (bool) --is unit double consonant (ζ, ξ, ψ)?
        word_initial (bool) -- is unit 1st letter of word?
    """
    __slots__ = (
        "kind", "text", "base", "is_diphthong",
        "has_iota_sub", "has_macron", "has_breve", "is_double",
        "word_initial",
    )

    def __init__(self, kind, text, base, **flags):
        self.kind = kind
        self.text = text
        self.base = base
        self.is_diphthong = flags.get("is_diphthong", False)
        self.has_iota_sub = flags.get("has_iota_sub", False)
        self.has_macron = flags.get("has_macron", False)
        self.has_breve = flags.get("has_breve", False)
        self.is_double = flags.get("is_double", False)
        self.word_initial = flags.get("word_initial", False)

    def __repr__(self):
        return f"_PhoneticUnit({self.kind!r}, {self.text!r})"


def _parse_phon_units(text: str) -> List[_PhoneticUnit]:
    """
    Parse grc txt into list of phonological units

    Parameters:
         text (str) -- raw grc txt

    Returns:
         units (List[_PhoneticUnit]) --
    """
    text = unicodedata.normalize("NFC", text)
    units: List[_PhoneticUnit] = []
    i = 0
    n = len(text)
    at_word_start = True

    while i < n:
        char = text[i]
        base = _base_char(char)

        # diphthongs
        if base in ALL_VOWELS and i + 1 < n:
            nxt = text[i + 1]
            nxt_base = _base_char(nxt)
            pair_base = base + nxt_base
            if pair_base in DIPHTHONG_SET and not _has_diaeresis(nxt):
                if i + 2 < n:
                    third = text[i + 2]
                    third_base = _base_char(third)
                    if (
                        nxt_base + third_base in DIPHTHONG_SET
                        and _has_circumflex(third)
                        and not _has_diaeresis(third)
                    ):
                        units.append(_PhoneticUnit(
                            "V", char, base,
                            is_diphthong=False,
                            has_iota_sub=_has_iota_subscript(char),
                            has_macron=_has_macron(char),
                            has_breve=_has_breve(char),
                            word_initial=at_word_start,
                        ))
                        at_word_start = False
                        i += 1
                        continue
                units.append(_PhoneticUnit(
                    "V", text[i:i + 2], pair_base,
                    is_diphthong=True,
                    has_iota_sub=False,
                    has_macron=False,
                    has_breve=False,
                    word_initial=at_word_start,
                ))
                at_word_start = False
                i += 2
                continue

        # single vowels
        if base in ALL_VOWELS:
            units.append(_PhoneticUnit(
                "V", char, base,
                is_diphthong=False,
                has_iota_sub=_has_iota_subscript(char),
                has_macron=_has_macron(char),
                has_breve=_has_breve(char),
                word_initial=at_word_start,
            ))
            at_word_start = False
            i += 1
            continue

        # consonants
        if base in CONSONANTS:
            units.append(_PhoneticUnit(
                "C", char, base,
                is_double=(base in DOUBLE_CONSONANTS),
                word_initial=at_word_start,
            ))
            at_word_start = False
            i += 1
            continue

        # skip non-alphabetic chars (== word boundary)
        if not unicodedata.combining(char) and not char.isalpha():
            at_word_start = True
        i += 1

    return units

"""
Syllabification
"""
class SyllableUnit:
    """
    Represents single syllable of grc

    Attributes:
        text (str) -- syllable composed of alphabetic characters
        nucleus (_PhoneticUnit) -- vowel / diphthong nucleus
        onset (List[_PhoneticUnit]) -- consonant(s) pre-nucleus
        coda (List[_PhoneticUnit]) -- consonant(s) post-nucleus
        has_muta_cum_liquida_after (bool) -- does code begin with plosive + liquid/nasal?
        quantity (str) -- L, S, X
    """

    def __init__(self, nucleus: _PhoneticUnit):
        self.nucleus: _PhoneticUnit = nucleus
        self.onset: List[_PhoneticUnit] = []
        self.coda: List[_PhoneticUnit] = []
        self.has_muta_cum_liquida_after: bool = False
        self.quantity: str = X
        self.text: str = ""

    def __repr__(self):
        return (
            f"SyllableUnit(text={self.text!r}, q={self.quantity}, "
            f"nuc={self.nucleus.text!r})"
        )


def _split_consonant_cluster(
    cluster: List[_PhoneticUnit],
) -> Tuple[List[_PhoneticUnit], List[_PhoneticUnit], bool]:
    """
    categorize consonant clusters coda/onset

    Parameters:
        cluster ( List[_PhoneticUnit] ) -- inter-vowel consonants

    Returns:
        coda ( List[_PhoneticUnit] ) -- close previous syllable
        onset ( List[_PhoneticUnit] ) -- open subsequent syllable
        muta_cum_liquida (bool) -- muta cum liquida?
    """
    if not cluster:
        return [], [], False

    total_sounds = sum(2 if u.is_double else 1 for u in cluster)

    if total_sounds == 1 and len(cluster) == 1:
        return [], cluster, False

    n = len(cluster)
    muta_cum_liquida = False

    # muta cum liquida check
    if n >= 2:
        penult_base = cluster[-2].base
        last_base = cluster[-1].base
        if penult_base in PLOSIVES and last_base in LIQUIDS_NASALS:
            coda = cluster[:-2]
            onset = cluster[-2:]
            muta_cum_liquida = True
            return coda, onset, muta_cum_liquida

    # double-consonant check
    if n == 1 and cluster[0].is_double:
        return [cluster[0]], [cluster[0]], False

    # default -- all but last consonant close prior syllable
    return cluster[:-1], cluster[-1:], False


def syllabify_line(line: str) -> List[SyllableUnit]:
    """
    Syllabify str of grc

    N.b. Ignores word boundaries, elision assumed

    Parameters:
        line (str) -- raw text of grc

    Returns:
        syllables ( List[SyllableUnit] ) --
            syllables w/ onset, nucleus, coda, + quantity assignments
    """
    # clean txt for processing
    cleaned = line.replace("\u2019", "").replace("\u0027", "").replace("\u2018", "")

    units = _parse_phon_units(cleaned)
    if not units:
        return []

    vowel_indices = [i for i, u in enumerate(units) if u.kind == "V"]
    if not vowel_indices:
        return []

    # construct syllables
    syllables: List[SyllableUnit] = []

    for v_pos, vi in enumerate(vowel_indices):
        syllable = SyllableUnit(nucleus=units[vi])

        if v_pos == 0:
            syllable.onset = [u for u in units[:vi] if u.kind == "C"]
        else:
            prev_vi = vowel_indices[v_pos - 1]
            cluster = [u for u in units[prev_vi + 1: vi] if u.kind == "C"]
            coda, onset, muta_cum_liquida = _split_consonant_cluster(cluster)

            if syllables:
                syllables[-1].coda = coda
                if muta_cum_liquida:
                    syllables[-1].has_muta_cum_liquida_after = True
            syllable.onset = onset

        syllables.append(syllable)

    if vowel_indices:
        last_vi = vowel_indices[-1]
        trailing = [u for u in units[last_vi + 1:] if u.kind == "C"]
        syllables[-1].coda = trailing

    # display syllables
    for syllable in syllables:
        parts = [u.text for u in syllable.onset]
        parts.append(syllable.nucleus.text)
        parts.extend(u.text for u in syllable.coda)
        syllable.text = "".join(parts)

    # assign vowel lengths
    _assign_quantities(syllables)

    return syllables


"""
Vowel length assignment
"""
def _intrinsic_vowel_quantity(nuc: _PhoneticUnit) -> str:
    """
    Mark vowel length

    Rules:
        - η, ω == L
        - ε, ο == S
        - diphthongs (αι, αυ, ει, ευ, οι, ου, ηυ, υι) == L
        - iota subscript (ᾳ, ῃ, ῳ) == L
        - α, ι, υ w/ macron == L
        - α, ι, υ w/ breve == S
        - α, ι, υ else == X

    Parameters:
        nuc (_PhoneticUnit) -- vowel

    Returns:
        str -- vowel length (L,S,X)
    """
    base = nuc.base

    # diphthong
    if nuc.is_diphthong:
        return L

    # iota subscript
    if nuc.has_iota_sub:
        return L

    # long
    if base in LONG_VOWELS:
        return L

    # short
    if base in SHORT_VOWELS:
        return S

    # flexible (w/ macornizer)
    if base in AMBIGUOUS_VOWELS:
        if nuc.has_macron:
            return L
        if nuc.has_breve:
            return S
        return X

    # in case of remainders
    return X


def _is_syllable_closed(syllable: SyllableUnit) -> bool:
    """
    """
    if not syllable.coda:
        return False
    # dbl consonant in coda closed
    return True


def _next_word_skeleton(syllables: List[SyllableUnit], i: int) -> str:
    """

    """
    if i + 1 >= len(syllables):
        return ""
    nxt = syllables[i + 1]
    if not (nxt.nucleus.word_initial or any(u.word_initial for u in nxt.onset)):
        return ""

    skeleton: List[str] = []
    last: Optional[_PhoneticUnit] = None
    started = False
    for k in range(i + 1, len(syllables)):
        syllable = syllables[k]
        for unit in (*syllable.onset, syllable.nucleus, *syllable.coda):
            if unit is last:
                continue
            last = unit
            if not started:
                if not unit.word_initial:
                    continue
                started = True
            elif unit.word_initial:
                return "".join(skeleton).replace("\u03c2", "\u03c3")
            skeleton.append(unit.base)
    return "".join(skeleton).replace("\u03c2", "\u03c3")


def _assign_quantities(syllables: List[SyllableUnit]) -> None:
    """
    Assign scan markets (i.e. L, S, X) to each syllable

    Parameters:
        syllables ( List[SyllableUnit] ) ---
    """
    n = len(syllables)
    for i, syllable in enumerate(syllables):
        intrinsic = _intrinsic_vowel_quantity(syllable.nucleus)

        # finl syllable = anceps
        if i == n - 1:
            syllable.quantity = X
            continue

        # closed syllable
        if _is_syllable_closed(syllable):
            # sigma + plosive
            if (
                len(syllable.coda) == 1
                and syllable.coda[0].base == "σ"
                and syllable.coda[0].word_initial
                and i + 1 < n
                and syllables[i + 1].onset
                and syllables[i + 1].onset[0].base in PLOSIVES
                and intrinsic != L
            ):
                syllable.quantity = X
            else:
                syllable.quantity = L
            continue

        # muta cum liquida check
        if syllable.has_muta_cum_liquida_after:
            syllable.quantity = L if intrinsic == L else X
            continue

        # correption
        if intrinsic == L and i + 1 < n:
            nxt_syl = syllables[i + 1]
            if not nxt_syl.onset and nxt_syl.nucleus.word_initial:
                syllable.quantity = X
                continue

        # liquid/nasal/σ check
        if intrinsic != L and i + 1 < n:
            nxt_syl = syllables[i + 1]
            if (
                len(nxt_syl.onset) == 1
                and nxt_syl.onset[0].word_initial
                and nxt_syl.onset[0].base in DRAWN_OUT_SONORANTS
            ):
                syllable.quantity = X
                continue

        # digamma check
        if intrinsic != L and i + 1 < n:
            if _next_word_skeleton(syllables, i) in DIGAMMA_WORDS:
                syllable.quantity = X
                continue

        syllable.quantity = intrinsic

"""
Hexameter verse check
"""

def _word_break_positions(syllables: List["SyllableUnit"]) -> Set[int]:
    """

    """
    breaks: Set[int] = set()
    for i in range(len(syllables) - 1):
        nxt = syllables[i + 1]
        if nxt.nucleus.word_initial or any(u.word_initial for u in nxt.onset):
            breaks.add(i)
    return breaks


def _quantity_check(actual: str, required: str) -> bool:
    """
    
    """
    if actual == X:
        return True
    return actual == required


def matches_hexameter(quantities: List[str],
                      word_breaks: Optional[Set[int]] = None,
                      return_resolved: bool = False):
    """
    Test if syllable quantity sequence is valid hexametere

    Parameters:
        quantities ( List[str] ) -- sequence of scan markers (1/syllable)
        word_breaks (Set[int]) --
        return_resolved (bool) -- 

    Returns:
        bool -- (default) does sequence scan as hexameter?
        Optional[List[str]] -- (return_resolved=True) resolved markers,
            1/syllable, or None
    """
    n = len(quantities)
    if n < 12 or n > 17:
        return None if return_resolved else False

    winning_starts: List[List[int]] = []

    def _caesura_ok(foot_starts: List[int]) -> bool:
        """
        Check if fem, masc, or hephthemimeral caesura
        """
        if word_breaks is None:
            return True
        s3 = foot_starts[2]
        s4 = foot_starts[3]
        if s3 in word_breaks:
            return True
        if s4 - s3 == 3 and (s3 + 1) in word_breaks:
            return True
        if s4 in word_breaks:
            return True
        return False

    def _match(syl_idx: int, foot: int, foot_starts: List[int]) -> bool:
        """
        
        """
        if foot == 6:
            return syl_idx == n and _caesura_ok(foot_starts)

        remaining_feet = 6 - foot
        if n - syl_idx < remaining_feet * 2:
            return False

        if foot < 5:
            # try dactyl
            if syl_idx + 3 <= n:
                if (_quantity_check(quantities[syl_idx], L) and
                        _quantity_check(quantities[syl_idx + 1], S) and
                        _quantity_check(quantities[syl_idx + 2], S)):
                    if _match(syl_idx + 3, foot + 1,
                              foot_starts + [syl_idx + 3]):
                        return True
            # try spondee
            if syl_idx + 2 <= n:
                if (_quantity_check(quantities[syl_idx], L) and
                        _quantity_check(quantities[syl_idx + 1], L)):
                    if _match(syl_idx + 2, foot + 1,
                              foot_starts + [syl_idx + 2]):
                        return True

        else:
            if syl_idx + 2 == n:
                if (_quantity_check(quantities[syl_idx], L)
                        and _caesura_ok(foot_starts)):
                    winning_starts.append(foot_starts)
                    return True

        return False

    matched = _match(0, 0, [0])
    if not return_resolved:
        return matched
    if not matched:
        return None

    # rebuild winning pattern syllable sequence
    starts = winning_starts[0] + [n]
    resolved: List[str] = []
    for foot in range(6):
        span = starts[foot + 1] - starts[foot]
        if foot < 5:
            resolved.extend([L, S, S] if span == 3 else [L, L])
        else:
            resolved.extend([L, X])
    return resolved


"""
Full line scansion
"""

def _base_letter_skeleton(text: str) -> str:
    """
    regularize grc line
    """
    out = []
    for char in unicodedata.normalize("NFC", text):
        base = _base_char(char)
        if base in ALL_VOWELS or base in CONSONANTS:
            out.append(base)
    return "".join(out)


def _synizesis_candidates(
    syllables: List["SyllableUnit"],
    word_breaks: Set[int],
    max_merges: int = 2,
):
    """

    """
    from itertools import combinations

    base = [s.quantity for s in syllables]
    yield base, word_breaks, [[i] for i in range(len(base))]

    mergeable = [
        i for i in range(len(syllables) - 1)
        if not syllables[i].coda
        and not syllables[i + 1].onset
        and not syllables[i + 1].nucleus.word_initial
    ]
    for r in range(1, min(max_merges, len(mergeable)) + 1):
        for subset in combinations(mergeable, r):
            chosen = set(subset)
            merged: List[str] = []
            merged_breaks: Set[int] = set()
            merged_groups: List[List[int]] = []
            i = 0
            while i < len(base):
                if i in chosen and i + 1 < len(base):
                    correptible = (
                        not syllables[i + 1].coda
                        and i + 2 < len(syllables)
                        and not syllables[i + 2].onset
                    )
                    merged.append(X if correptible else L)
                    merged_groups.append([i, i + 1])
                    if (i + 1) in word_breaks:
                        merged_breaks.add(len(merged) - 1)
                    i += 2
                else:
                    merged.append(base[i])
                    merged_groups.append([i])
                    if i in word_breaks:
                        merged_breaks.add(len(merged) - 1)
                    i += 1
            yield merged, merged_breaks, merged_groups


def _elision_candidates(
    syllables: List["SyllableUnit"],
    word_breaks: Set[int],
    max_elisions: int = 2,
):
    """
    Yield alternative readings w/ word-final short vowels elided before vowel-initial words

    Parameters:
        syllables ( List[SyllableUnit] ) --
        word_breaks ( Set[int] ) -- word-break positions (break after syllable i)
        max_elisions (int) -- max simultaneous elisions per line
    """
    from itertools import combinations

    base = [s.quantity for s in syllables]
    elidable = [
        i for i in range(len(syllables) - 1)
        if not syllables[i].coda
        and not syllables[i + 1].onset
        and syllables[i + 1].nucleus.word_initial
        and not syllables[i].nucleus.is_diphthong
        and _intrinsic_vowel_quantity(syllables[i].nucleus) != L
    ]
    for r in range(1, min(max_elisions, len(elidable)) + 1):
        for subset in combinations(elidable, r):
            chosen = set(subset)
            out: List[str] = []
            out_breaks: Set[int] = set()
            out_groups: List[List[int]] = []
            pending: List[int] = []
            for i, q in enumerate(base):
                if i in chosen:
                    if i in word_breaks and out:
                        out_breaks.add(len(out) - 1)
                    pending.append(i)
                    continue
                out.append(q)
                out_groups.append(pending + [i])
                pending = []
                if i in word_breaks:
                    out_breaks.add(len(out) - 1)
            yield out, out_breaks, out_groups


def _prepare_scan_syllables(line: str, use_macronizer: bool = True) -> List["SyllableUnit"]:
    """
    Macronize (optionally) + syllabify a line for scanning
    """
    scan_input = line
    if use_macronizer:
        macronized = macronize(line)
        if macronized is not None:
            if _base_letter_skeleton(macronized) == _base_letter_skeleton(line):
                scan_input = macronized
            else:
                logging.warning(
                    "Macronizer altered the letter skeleton of a line", line,
                )
    return syllabify_line(scan_input)


def _scan_readings(line: str, use_macronizer: bool = True):
    """

    """
    syllables = _prepare_scan_syllables(line, use_macronizer)
    word_breaks = _word_break_positions(syllables)
    yield from _synizesis_candidates(syllables, word_breaks)
    yield from _elision_candidates(syllables, word_breaks)


def scan_line(line: str, use_macronizer: bool = True) -> List[str]:
    """
    Scan line, return list of syllable scan markers

    Parameters:
        line (str) --
        use_macronizer (bool)--

    Returns:
        List[str]
            1 scan marker per syllable
    """
    base: Optional[List[str]] = None
    for quantities, breaks, _groups in _scan_readings(line, use_macronizer):
        if base is None:
            base = quantities
        if matches_hexameter(quantities, breaks):
            return quantities
    return base if base is not None else []


def line_matches_hexameter(line: str, use_macronizer: bool = True) -> bool:
    """
    test if line scans as valid hexameter

    Parameters:
        line (str) --
        use_macronizer (bool) --
    """
    return any(
        matches_hexameter(quantities, breaks)
        for quantities, breaks, _groups in _scan_readings(line, use_macronizer)
    )


class LineScansion(NamedTuple):
    """
    Display-ready verse line scansion
    """
    syllables: List[str]
    markers: List[str]
    word_breaks: List[int]
    syllable_units: List[List[Tuple[str, int]]]
    unit_skeleton_lens: List[int]


def scan_line_display(
    line: str, use_macronizer: bool = True,
) -> LineScansion:
    """
    Scan line for frontend scansion display

    Parameters:
        line (str) -- restored verse line
        use_macronizer (bool) -- resolve ambiguous vowels w/ grc_macronizer?

    Returns:
        LineScansion --
    """
    syllables = _prepare_scan_syllables(line, use_macronizer)
    if not syllables:
        return LineScansion([], [], [], [], [])
    word_breaks = _word_break_positions(syllables)

    unit_ordinals: Dict[int, int] = {}
    unit_skeleton_lens: List[int] = []
    per_syllable_units: List[List[Tuple[str, int]]] = []
    for syllable in syllables:
        composition: List[Tuple[str, int]] = []
        for unit in (*syllable.onset, syllable.nucleus, *syllable.coda):
            ordinal = unit_ordinals.get(id(unit))
            if ordinal is None:
                ordinal = len(unit_skeleton_lens)
                unit_ordinals[id(unit)] = ordinal
                unit_skeleton_lens.append(len(unit.base))
            composition.append((unit.text, ordinal))
        per_syllable_units.append(composition)

    from itertools import chain

    fallback: Optional[LineScansion] = None
    for quantities, breaks, groups in chain(
        _synizesis_candidates(syllables, word_breaks),
        _elision_candidates(syllables, word_breaks),
    ):
        texts = ["".join(syllables[j].text for j in g) for g in groups]
        group_units = [
            [pair for j in g for pair in per_syllable_units[j]] for g in groups
        ]
        if fallback is None:
            fallback = LineScansion(
                texts, list(quantities), sorted(breaks),
                group_units, unit_skeleton_lens,
            )
        resolved = matches_hexameter(quantities, breaks, return_resolved=True)
        if resolved is not None:
            return LineScansion(
                texts, resolved, sorted(breaks),
                group_units, unit_skeleton_lens,
            )
    return fallback if fallback is not None else LineScansion([], [], [], [], [])
