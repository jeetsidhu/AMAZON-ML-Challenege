"""Script-independent phonetic skeleton of a Latin token.

Used as a fallback for Indic-script names whose words are missing from the learned lexicon:
unidecode("मार्केटिंग") = "maarkettinga" and "marketing" both reduce to "mrktng".
"""
import re

_RULES = [
    (re.compile(r"tion"), "sn"), (re.compile(r"ture"), "cr"), (re.compile(r"c(?=[eiy])"), "s"),
    (re.compile(r"x"), "ks"), (re.compile(r"ph"), "f"), (re.compile(r"sh"), "s"),
    (re.compile(r"([bcdgjkptr])h"), r"\1"),
]
_VOWEL_RE = re.compile(r"[aeiouy]")
_REPEAT_RE = re.compile(r"(.)\1+")
_SUBST = str.maketrans({"c": "k", "q": "k", "w": "b", "v": "b", "z": "j", "g": "j"})


def skeleton(tok):
    """First letter + consonants: aspirates merged, similar consonants unified, repeats collapsed."""
    t = tok.lower()
    if len(t) < 3 or not t.isalpha():
        return t
    for rx, rep in _RULES:
        t = rx.sub(rep, t)
    t = t.translate(_SUBST)
    s = t[0] + _VOWEL_RE.sub("", t[1:])
    return _REPEAT_RE.sub(r"\1", s)
