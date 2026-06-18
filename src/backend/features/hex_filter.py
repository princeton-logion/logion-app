"""
Hexameter filter for word-level MLM predictions

Based on hexameter scansion described in Barbara Graziosi & Johannes Haubold (eds.), Homer: Iliad, Book VI, Cambridge Greek and Latin Classics, Cambridge UP, 2015.

This module validates predicted sub/word spans per premodern Greek hexameter conventions.  After a model generates predictions, hex_filter confirms whether each prediction yields a metrically valid line and removes metrically invalid predictions.


grc_macronizer:
    optionally use grc_macronizer (https://github.com/Urdatorn/grc-macronizer) to scan ambiguous vowels (α, ι, υ).  Sans grc_macronizer, filter scans those vowels as flexible (X) for higher recall

Disclaimers:
    - assumes prehandling of elision + synizesis/crasis
    - correption: scans flexible (X)
    - muta cum liquida (plosive + liquid/nasal onset): scans flexible (X)
"""

import unicodedata
import logging
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional
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
    """
    Toggle grc_macronizer for module
    """
    global USE_MACRONIZER
    USE_MACRONIZER = bool(enabled)


"""
Premodern Greek phonological constants
"""
# vowels
LONG_VOWELS: frozenset = frozenset("ηω")
SHORT_VOWELS: frozenset = frozenset("εο")
AMBIGUOUS_VOWELS: frozenset = frozenset("αιυ")
ALL_VOWELS: frozenset = LONG_VOWELS | SHORT_VOWELS | AMBIGUOUS_VOWELS

# diphthongs
DIPHTHONG_LIST: Tuple[str, ...] = (
    "αι", "αυ", "ει", "ευ", "οι", "ου", "ηυ", "υι",
)
DIPHTHONG_SET: frozenset = frozenset(DIPHTHONG_LIST)

# consonants (+ digamma)
CONSONANTS: frozenset = frozenset("βγδζθκλμνξπρσςτφχψ\u03DD")

# plosives
PLOSIVES: frozenset = frozenset("πβφτδθκγχ")

# liquids + nasals
LIQUIDS_NASALS: frozenset = frozenset("λρμν")

# sonorants (λ, μ, ν, ρ, σ)
DRAWN_OUT_SONORANTS: frozenset = LIQUIDS_NASALS | frozenset("σ")

# dbl consonants
DOUBLE_CONSONANTS: frozenset = frozenset("ζξψ")

# select frequent digamma (ϝ) wrds
DIGAMMA_WORDS: frozenset = frozenset({
    # ϝός/ἑός
    "ον", "ην", "ων", "οσ", "οισι", "οισ",
    "εον", "εην", "εης", "εω", "εη", "εων", "εα", "εασ", "εοιο",
    # ϝάναξ
    "αναξ", "ανακτοσ", "ανακτι", "ανακτα", "ανακτεσ", "ανακτων",
    # ϝέργον
    "εργον", "εργα", "εργων", "εργω", "εργου", "εργοισι", "εργοισ",
    # ϝοἶνος
    "οινοσ", "οινον", "οινου", "οινω", "οινοιο", "οινοισι",
    # ϝοἶκος
    "οικοσ", "οικον", "οικου", "οικω", "οικοι", "οικαδε", "οικονδε",
    # ϝιδεῖν/ϝεἶδον
    "ιδειν", "ιδων", "ιδεν", "ιδε", "ιδεσθαι", "ειδον", "ειδε",
    # ϝέπος
    "επος", "επεα", "επεσσι", "επεων", "επεεσσι",
    # ϝοἶδα
    "οιδα", "οιδε", "ισμεν", "ιδμεν",
})

# scansion markers
L = "L"   # long
S = "S"   # short
X = "X"   # flexible


#  Unicode Helpers

def _base_char(char: str) -> str:
    """
    Lowercase + de-accent Grc character
    """
    nfd = unicodedata.normalize("NFD", char)
    return nfd[0].lower() if nfd else char.lower()


def _has_iota_subscript(char: str) -> bool:
    """
    Check if character has iota subscript (U+0345)
    Makes vowel long
    """
    return "\u0345" in unicodedata.normalize("NFD", char)


def _has_macron(char: str) -> bool:
    """
    Check if character has combining macron (U+0304)
    Makes flexible vowel long
    For grc_macronizer
    """
    return "\u0304" in unicodedata.normalize("NFD", char)


def _has_breve(char: str) -> bool:
    """
    Check if character has combining breve (U+0306)
    Makes flexible vowel short
    For grc_macronizer
    """
    return "\u0306" in unicodedata.normalize("NFD", char)


def _has_circumflex(char: str) -> bool:
    """
    Check if character has circumflex (U+0342)
    """
    return "\u0342" in unicodedata.normalize("NFD", char)


def _has_diaeresis(char: str) -> bool:
    """
    Check if character has diaeresis (U+0308)
    Makes two adjacent vowels separate
    """
    return "\u0308" in unicodedata.normalize("NFD", char)


"""
grc_macronizer helpers

-currently obsolete, keep for testing grc_macronizer integration
-currently bypass grc_macronizer for initial test and higher recall
"""

def _patch_odycy_loader_to_memoize():
    """
    Aims to load odyCy once for grc_macronizer
    """
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
    """
    Instantiate grc_macronizer on first run

    Returns None if grc_macronizer not installed
    """
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
    """
    Macronize one verse line
    """
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
    """
    macronize text w/ grc_macronizer if enabled

    Parameters:
        text (str) -- grc text

    Returns:
        str or None
    """
    if not (HAS_MACRONIZER and USE_MACRONIZER):
        return None
    return _macronize_cached(text)


"""
Phonology parser
"""

class _PhonUnit:
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
        return f"_PhonUnit({self.kind!r}, {self.text!r})"


def _parse_phon_units(text: str) -> List[_PhonUnit]:
    """
    Parse grc txt into list of phonological units

    Parameters:
         text (str) -- raw grc txt

    Returns:
         units (List[_PhonUnit]) --
    """
    text = unicodedata.normalize("NFC", text)
    units: List[_PhonUnit] = []
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
                        units.append(_PhonUnit(
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
                units.append(_PhonUnit(
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
            units.append(_PhonUnit(
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
            units.append(_PhonUnit(
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
class SyllableInfo:
    """
    Represents single syllable of grc

    Attributes:
        text (str) -- syllable composed of alphabetic characters
        nucleus (_PhonUnit) -- vowel / diphthong nucleus
        onset (List[_PhonUnit]) -- consonant(s) pre-nucleus
        coda (List[_PhonUnit]) -- consonant(s) post-nucleus
        has_muta_cum_liquida_after (bool) -- does code begin with plosive + liquid/nasal?
        quantity (str) -- L, S, X
    """

    def __init__(self, nucleus: _PhonUnit):
        self.nucleus: _PhonUnit = nucleus
        self.onset: List[_PhonUnit] = []
        self.coda: List[_PhonUnit] = []
        self.has_muta_cum_liquida_after: bool = False
        self.quantity: str = X
        self.text: str = ""

    def __repr__(self):
        return (
            f"SyllableInfo(text={self.text!r}, q={self.quantity}, "
            f"nuc={self.nucleus.text!r})"
        )


def _split_consonant_cluster(
    cluster: List[_PhonUnit],
) -> Tuple[List[_PhonUnit], List[_PhonUnit], bool]:
    """
    categorize consonant clusters coda/onset

    Parameters:
        cluster ( List[_PhonUnit] ) -- inter-vowel consonants

    Returns:
        coda ( List[_PhonUnit] ) -- close previous syllable
        onset ( List[_PhonUnit] ) -- open subsequent syllable
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


def syllabify_line(line: str) -> List[SyllableInfo]:
    """
    Syllabify str of grc

    N.b. Ignores word boundaries, elision assumed

    Parameters:
        line (str) -- raw text of grc

    Returns:
        syllables ( List[SyllableInfo] ) --
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
    syllables: List[SyllableInfo] = []

    for v_pos, vi in enumerate(vowel_indices):
        syl = SyllableInfo(nucleus=units[vi])

        if v_pos == 0:
            syl.onset = [u for u in units[:vi] if u.kind == "C"]
        else:
            prev_vi = vowel_indices[v_pos - 1]
            cluster = [u for u in units[prev_vi + 1: vi] if u.kind == "C"]
            coda, onset, muta_cum_liquida = _split_consonant_cluster(cluster)

            if syllables:
                syllables[-1].coda = coda
                if muta_cum_liquida:
                    syllables[-1].has_muta_cum_liquida_after = True
            syl.onset = onset

        syllables.append(syl)

    if vowel_indices:
        last_vi = vowel_indices[-1]
        trailing = [u for u in units[last_vi + 1:] if u.kind == "C"]
        syllables[-1].coda = trailing

    # display syllables
    for syl in syllables:
        parts = [u.text for u in syl.onset]
        parts.append(syl.nucleus.text)
        parts.extend(u.text for u in syl.coda)
        syl.text = "".join(parts)

    # assign vowel lengths
    _assign_quantities(syllables)

    return syllables


"""
Vowel length assignment
"""

def _intrinsic_vowel_quantity(nuc: _PhonUnit) -> str:
    """
    Mark length of vowel

    Rules:
        - η, ω == L
        - ε, ο == S
        - diphthongs (αι, αυ, ει, ευ, οι, ου, ηυ, υι) == L
        - iota subscript (ᾳ, ῃ, ῳ) == L
        - α, ι, υ w/ macron == L
        - α, ι, υ w/ breve == S
        - α, ι, υ else == X

    Parameters:
        nuc (_PhonUnit) -- vowel

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


def _is_syllable_closed(syl: SyllableInfo) -> bool:
    """
    """
    if not syl.coda:
        return False
    # dbl consonant in coda closed
    return True


def _next_word_skeleton(syllables: List[SyllableInfo], i: int) -> str:
    """

    """
    if i + 1 >= len(syllables):
        return ""
    nxt = syllables[i + 1]
    if not (nxt.nucleus.word_initial or any(u.word_initial for u in nxt.onset)):
        return ""

    skel: List[str] = []
    last: Optional[_PhonUnit] = None
    started = False
    for k in range(i + 1, len(syllables)):
        syl = syllables[k]
        for unit in (*syl.onset, syl.nucleus, *syl.coda):
            if unit is last:
                continue
            last = unit
            if not started:
                if not unit.word_initial:
                    continue
                started = True
            elif unit.word_initial:
                return "".join(skel).replace("\u03c2", "\u03c3")
            skel.append(unit.base)
    return "".join(skel).replace("\u03c2", "\u03c3")


def _assign_quantities(syllables: List[SyllableInfo]) -> None:
    """
    Assign scan markets (i.e. L, S, X) to each syllable

    Parameters:
        syllables ( List[SyllableInfo] ) ---
    """
    n = len(syllables)
    for i, syl in enumerate(syllables):
        intrinsic = _intrinsic_vowel_quantity(syl.nucleus)

        # finl syllable = anceps
        if i == n - 1:
            syl.quantity = X
            continue

        # closed syllable
        if _is_syllable_closed(syl):
            # sigma + plosive
            if (
                len(syl.coda) == 1
                and syl.coda[0].base == "σ"
                and syl.coda[0].word_initial
                and i + 1 < n
                and syllables[i + 1].onset
                and syllables[i + 1].onset[0].base in PLOSIVES
                and intrinsic != L
            ):
                syl.quantity = X
            else:
                syl.quantity = L
            continue

        # muta cum liquida check
        if syl.has_muta_cum_liquida_after:
            syl.quantity = L if intrinsic == L else X
            continue

        # correption
        if intrinsic == L and i + 1 < n:
            nxt_syl = syllables[i + 1]
            if not nxt_syl.onset:
                syl.quantity = X
                continue

        # liquid/nasal/σ check
        if intrinsic != L and i + 1 < n:
            nxt_syl = syllables[i + 1]
            if (
                len(nxt_syl.onset) == 1
                and nxt_syl.onset[0].word_initial
                and nxt_syl.onset[0].base in DRAWN_OUT_SONORANTS
            ):
                syl.quantity = X
                continue

        # digamma check
        if intrinsic != L and i + 1 < n:
            if _next_word_skeleton(syllables, i) in DIGAMMA_WORDS:
                syl.quantity = X
                continue

        syl.quantity = intrinsic

"""
Hexameter verse check
"""

def _word_break_positions(syllables: List["SyllableInfo"]) -> Set[int]:
    """

    """
    breaks: Set[int] = set()
    for i in range(len(syllables) - 1):
        nxt = syllables[i + 1]
        if nxt.nucleus.word_initial or any(u.word_initial for u in nxt.onset):
            breaks.add(i)
    return breaks


def _q_ok(actual: str, required: str) -> bool:
    """
    
    """
    if actual == X:
        return True
    return actual == required


def matches_hexameter(quantities: List[str],
                      word_breaks: Optional[Set[int]] = None) -> bool:
    """
    Test if syllable quantity sequence is valid hexametere

    Parameters:
        quantities ( List[str] ) -- sequence of scan markers (1/syllable)
        word_breaks (Set[int]) --

    """
    n = len(quantities)
    if n < 12 or n > 17:
        return False

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
                if (_q_ok(quantities[syl_idx], L) and
                        _q_ok(quantities[syl_idx + 1], S) and
                        _q_ok(quantities[syl_idx + 2], S)):
                    if _match(syl_idx + 3, foot + 1,
                              foot_starts + [syl_idx + 3]):
                        return True
            # try spondee
            if syl_idx + 2 <= n:
                if (_q_ok(quantities[syl_idx], L) and
                        _q_ok(quantities[syl_idx + 1], L)):
                    if _match(syl_idx + 2, foot + 1,
                              foot_starts + [syl_idx + 2]):
                        return True

        else:
            if syl_idx + 2 == n:
                if _q_ok(quantities[syl_idx], L):
                    return _caesura_ok(foot_starts)

        return False

    return _match(0, 0, [0])


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
    syllables: List["SyllableInfo"],
    word_breaks: Set[int],
    max_merges: int = 2,
):
    """

    """
    from itertools import combinations

    base = [s.quantity for s in syllables]
    yield base, word_breaks

    hiatus = [
        i for i in range(len(syllables) - 1)
        if not syllables[i].coda and not syllables[i + 1].onset
    ]
    for r in range(1, min(max_merges, len(hiatus)) + 1):
        for subset in combinations(hiatus, r):
            chosen = set(subset)
            merged: List[str] = []
            merged_breaks: Set[int] = set()
            i = 0
            while i < len(base):
                if i in chosen and i + 1 < len(base):
                    correptible = (
                        not syllables[i + 1].coda
                        and i + 2 < len(syllables)
                        and not syllables[i + 2].onset
                    )
                    merged.append(X if correptible else L)
                    if (i + 1) in word_breaks:
                        merged_breaks.add(len(merged) - 1)
                    i += 2
                else:
                    merged.append(base[i])
                    if i in word_breaks:
                        merged_breaks.add(len(merged) - 1)
                    i += 1
            yield merged, merged_breaks


def _scan_readings(line: str, use_macronizer: bool = True):
    """

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

    syllables = syllabify_line(scan_input)
    word_breaks = _word_break_positions(syllables)
    yield from _synizesis_candidates(syllables, word_breaks)


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
    for quantities, breaks in _scan_readings(line, use_macronizer):
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
        for quantities, breaks in _scan_readings(line, use_macronizer)
    )
