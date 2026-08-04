import logging
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

"""
Hierarchical positional concordance for Greek hexameter gap-filling
"""

# keep elision markers
_PUNCTUATION = ",.;·:!?()[]<>«»\u0387\u00b7\u2013\u2014\u2019\u201c\u201d\"—†"
_ELISION = "\u02bc\u1fbd'"
_EDITORIAL_MARKS = {0x0323, 0x0332}
_APOSTROPHE_FOLD = str.maketrans(
    {"\u2019": "\u02bc", "\u1fbd": "\u02bc", "'": "\u02bc"})
_COMBINING = "\u0300-\u0344\u0346-\u036f"
_ADSCRIPT_RE = re.compile(
    f"([\u03b7\u03c9][{_COMBINING}]*)\u03b9(?![{_COMBINING}]*\u0308)")


def normalize_word(word: str) -> str:
    w = "".join(
        c for c in unicodedata.normalize("NFD", word)
        if ord(c) not in _EDITORIAL_MARKS
    )
    w = _ADSCRIPT_RE.sub("\\1\u0345", w)
    w = unicodedata.normalize("NFC", w)
    w = w.translate(_APOSTROPHE_FOLD)
    w = w.strip(_PUNCTUATION)
    return w.lower()


def _split_line(line: str) -> Tuple[List[str], List[str]]:
    """
    Split line into index-aligned (surface_words, normalized_words)
    """
    transmitted_text, normalized_text = [], []
    for raw in line.split():
        n = normalize_word(raw)
        if not n:
            continue
        s = unicodedata.normalize("NFC", raw).strip(_PUNCTUATION)
        transmitted_text.append(s)
        normalized_text.append(sys.intern(n))
    return transmitted_text, normalized_text


def _sede_position(line_initial: bool, line_final: bool) -> str:
    """
    Line-position class of gap:
        I = initial
        F = final
        B = both
        M = medial
    """
    if line_initial and line_final:
        return "B"
    if line_initial:
        return "I"
    if line_final:
        return "F"
    return "M"


_CONTEXT_BACKOFF_SCHEDULE = [
    (2, 2), (2, 1), (1, 2), (1, 1), (2, 0), (0, 2), (1, 0), (0, 1),
]

# particles to downgrade
_PARTICLE_STOPWORDS = frozenset({
    "δέ", "δὲ", "δʼ", "τέ", "τὲ", "τʼ", "καί", "καὶ", "γάρ", "γὰρ",
    "ἄρ", "ἂρ", "ἄρα", "ῥα", "ῥά", "ῥʼ", "μέν", "μὲν", "δή", "δὴ",
    "ἄν", "ἂν", "κέ", "κὲ", "κέν", "κὲν", "περ", "πέρ", "γε", "γέ", "γʼ",
})

FORMAT_VERSION = 1

class FormularConcordance:
    """
    Per-stratum positional n-gram index for hexameter

    Attributes:
        strata ( Dict[str, Dict[key, Counter]] ) --
            corpus name -> {(left_tuple, right_tuple, sedes): Counter of
            normalized_text fills}
        transmitted_text ( Dict[str, str] ) --
            normalized_text fill -> transmitted_text form (first attestation)
        max_context_window (int) -- max context words kept on each side
        max_fill (int) -- max words per stored fill span
    """

    def __init__(self, max_context_window: int = 2, max_fill: int = 3):
        self.max_context_window = max_context_window
        self.max_fill = max_fill
        self.strata: Dict[str, Dict[Any, Counter]] = {}
        self.transmitted_text: Dict[str, str] = {}


    def add_corpus(self, name: str, lines: Iterable[str]) -> None:
        """
        Index every line of given corpus under 

        Parameters:
            name (str) -- stratum name (i.e. author)
            lines ( Iterable[str] ) -- one verse per string
        """
        stratum = self.strata.setdefault(name, defaultdict(Counter))
        num_lines = 0
        for line in lines:
            surface_words, words = _split_line(line)
            m = len(words)
            if m < 2:
                continue
            num_lines += 1
            for L in range(1, self.max_fill + 1):
                for i in range(0, m - L + 1):
                    fill = " ".join(words[i:i + L])
                    if fill not in self.transmitted_text:
                        self.transmitted_text[fill] = " ".join(
                            surface_words[i:i + L])
                    sedes = _sede_position(i == 0, i + L == m)
                    for j, k in _CONTEXT_BACKOFF_SCHEDULE:
                        if j > min(i, self.max_context_window):
                            continue
                        if k > min(m - (i + L), self.max_context_window):
                            continue
                        key = (
                            tuple(words[i - j:i]),
                            tuple(words[i + L:i + L + k]),
                            sedes,
                        )
                        stratum[key][fill] += 1
        logging.info(
            "FormularConcordance: stratum '%s' indexed %d lines (%d keys total)",
            name, num_lines, len(stratum),
        )

    def query(
        self,
        pre_lacuna: List[str],
        post_lacuna: List[str],
        line_initial: bool,
        line_final: bool,
        stratum_weights: Dict[str, float],
        num_attestations: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve attestations for gap
        """
        preceding_words = [normalize_word(w) for w in pre_lacuna]
        preceding_words = [w for w in preceding_words if w][-self.max_context_window:]
        post_words = [normalize_word(w) for w in post_lacuna]
        post_words = [w for w in post_words if w][:self.max_context_window]
        sedes = _sede_position(line_initial, line_final)

        active = []
        for name, weight in stratum_weights.items():
            stratum = self.strata.get(name)
            if stratum is None:
                logging.warning(
                    "FormularConcordance: unknown stratum '%s' ignored", name)
                continue
            if weight > 0:
                active.append((stratum, weight))
        if not active:
            return []

        def _consult(key):
            scores: Dict[str, float] = defaultdict(float)
            for stratum, weight in active:
                counts = stratum.get(key)
                if not counts:
                    continue
                total = sum(counts.values())
                for fill, c in counts.items():
                    scores[fill] += weight * (c / total)
            if not scores:
                return None
            ranked = sorted(
                scores.items(), key=lambda x: x[1], reverse=True,
            )[:num_attestations]
            return [
                (self.transmitted_text.get(f, f), s) for f, s in ranked
            ]

        deferred = []
        for j, k in _CONTEXT_BACKOFF_SCHEDULE:
            if j > len(preceding_words) or k > len(post_words):
                continue
            key = (
                tuple(preceding_words[len(preceding_words) - j:]) if j else (),
                tuple(post_words[:k]),
                sedes,
            )
            # particles downgraded
            if j + k == 1 and (key[0] + key[1])[0] in _PARTICLE_STOPWORDS:
                deferred.append(key)
                continue
            result = _consult(key)
            if result:
                return result
        for key in deferred:
            result = _consult(key)
            if result:
                return result
        return []

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "format_version": FORMAT_VERSION,
                    "max_context_window": self.max_context_window,
                    "max_fill": self.max_fill,
                    "strata": {n: dict(t) for n, t in self.strata.items()},
                    "transmitted_text": self.transmitted_text,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str) -> "FormularConcordance":
        with open(path, "rb") as f:
            data = pickle.load(f)
        version = data.get("format_version")
        if version is None:
            raise ValueError("Concordance has no format_version")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Concordance format v{version} later than app. Update app to use concordance."
            )
        idx = cls(max_context_window=data["max_context_window"], max_fill=data["max_fill"])
        idx.strata = {
            n: defaultdict(Counter, t) for n, t in data["strata"].items()
        }
        idx.transmitted_text = data["transmitted_text"]
        return idx


def stratum_weights(
    target_strata: List[str],
    cognate_strata: List[str],
    all_strata: List[str],
    lambdas: Tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> Dict[str, float]:
    """
    Three-level interpolation weights:
        -target author/work(s)
        -related tradition cluster
        -whole indexed tradition
    """
    weights: Dict[str, float] = defaultdict(float)
    for level, lam in zip(
            (target_strata, cognate_strata, all_strata), lambdas):
        if not level:
            continue
        share = lam / len(level)
        for name in level:
            weights[name] += share
    return dict(weights)
