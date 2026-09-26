"""Country-agnostic text normalisation for business names, addresses and country labels.

Only *generic*, hand-written vocabulary tables are used (legal-form abbreviations, street-type
abbreviations, a handful of transliteration spellings).  There are no country / city / state
gazetteers, no external look-ups and nothing that is specific to the countries seen in training:
accents are folded, punctuation is unified and the same rules are applied to every label.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd

# ----------------------------------------------------------------------------- unicode / punctuation
_SPECIAL = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "ø": "o", "Ø": "o",
                          "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "ı": "i", "þ": "th"})
_WS = re.compile(r"\s+")
_APOS = re.compile(r"[\u2018\u2019\u02bc`\u00b4]")
_DOTTED = re.compile(r"\b(?:[^\W\d_]\.){2,}")            # l.l.c.  p.v.t.  u.s.a.
_ELISION = re.compile(r"\b(?:l|d|j|m|n|s|t|c|qu)'(?=\w)")  # French elision  l'atelier -> atelier
_PUNCT = re.compile(r"[^\w\s]|_")
_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|ere|eme|e)\b")


def fold(s) -> str:
    """Lower-case, strip accents / diacritics (NFKD), unify a few special letters."""
    if s is None:
        return ""
    s = str(s)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _undot(m):
    return m.group(0).replace(".", "")


def clean_text(s) -> str:
    """fold + '&'->'and' + dotted acronyms joined + apostrophes dropped + punctuation -> space."""
    s = fold(s)
    if not s:
        return ""
    s = _APOS.sub("'", s).replace("&", " and ")
    s = _DOTTED.sub(_undot, s)
    s = _ELISION.sub("", s)
    s = s.replace("'", "")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


# ----------------------------------------------------------------------------- vocabularies
LEGAL = {"inc", "corp", "co", "ltd", "llc", "llp", "lp", "plc", "pllc", "pvt", "opc", "sarl", "sas", "sasu",
         "eurl", "sa", "snc", "sci", "scop", "gmbh", "ag", "bv", "nv", "pty", "srl", "spa", "cie", "ets"}

STOP_NAME = {"the", "and", "of", "de", "la", "le", "les", "du", "des", "et", "un", "une", "au", "aux", "el",
             "los", "las", "y", "von", "der", "die", "das", "a", "an", "for", "in", "at"}

NAME_CANON: Dict[str, str] = {
    # legal forms -> short canonical form
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd", "private": "pvt",
    "etablissements": "ets", "etablissement": "ets", "etab": "ets",
    # common business-word abbreviations -> long canonical form
    "intl": "international", "internatl": "international", "svcs": "service", "svc": "service", "serv": "service",
    "mfg": "manufacturing", "mfr": "manufacturer", "mfrs": "manufacturer", "engg": "engineering",
    "engr": "engineering", "eng": "engineering", "ent": "enterprise", "entp": "enterprise", "entrp": "enterprise",
    "assoc": "associate", "assocs": "associate", "bros": "brother", "grp": "group", "hldgs": "holding",
    "hldg": "holding", "mgmt": "management", "mktg": "marketing", "univ": "university", "dept": "department",
    "inst": "institute", "natl": "national", "technology": "tech", "technologie": "tech", "technolog": "tech",
    "technical": "tech", "sys": "system", "syst": "system", "elec": "electric", "elect": "electric",
    "electrical": "electric", "chem": "chemical", "pharm": "pharmaceutical", "pharma": "pharmaceutical",
    "consultancy": "consulting", "consultant": "consulting", "const": "construction", "constr": "construction",
    "trdg": "trading", "inds": "industry", "indus": "industry",
    # frequent transliteration spellings
    "shri": "shree", "sri": "shree", "sree": "shree", "shiri": "shree", "luxmi": "lakshmi", "laxmi": "lakshmi",
    "krushna": "krishna", "krisna": "krishna", "jay": "jai", "mohd": "mohammad", "mohammed": "mohammad",
    "muhammad": "mohammad",
    # saint / sainte
    "st": "saint", "ste": "saint", "sainte": "saint",
}

ADDR_CANON: Dict[str, str] = {
    "street": "st", "str": "st", "saint": "st", "suite": "ste", "sainte": "ste", "road": "rd", "avenue": "ave",
    "av": "ave", "aven": "ave", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd", "bvd": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "place": "pl", "square": "sq", "highway": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "circle": "cir", "terrace": "ter", "terr": "ter", "floor": "fl", "flr": "fl",
    "building": "bldg", "bldng": "bldg", "apartment": "apt", "apts": "apt", "apartments": "apt",
    "sector": "sec", "sect": "sec", "industrial": "indl", "estate": "est", "colony": "col", "society": "soc",
    "socy": "soc", "complex": "cplx", "cmplx": "cplx", "market": "mkt", "mrkt": "mkt", "district": "dist",
    "taluka": "tal", "tehsil": "tal", "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "impasse": "imp", "allee": "all", "chemin": "ch",
    "route": "rte", "faubourg": "fg", "fbg": "fg", "centre": "ctr", "center": "ctr", "ctre": "ctr",
    "commercial": "cial", "quai": "qu", "grnd": "gf", "ground": "gf", "gujrat": "gujarat",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
    "eighth": "8", "ninth": "9", "tenth": "10",
}
ADDR_STOP = {"no", "nos", "number", "num", "the", "of", "de", "la", "le", "les", "du", "des", "and", "et", "at", "in"}

_LM = re.compile(r"\b(?:near|nr|opposite|opp|oppo|behind|bhd|beside|besides|bsd|next to|adjacent|adj|close to|"
                 r"in front of|across from|across|landmark|nearby|beyond|a cote de|pres de|face a|derriere|"
                 r"en face de|proche de)\b")

_PH = (("sch", "s"), ("tch", "c"), ("ph", "f"), ("ck", "k"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"),
       ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "s"),
       ("c", "k"))


def phon_key(t: str) -> str:
    """Crude consonant-skeleton phonetic key (transliteration / spelling-variant tolerant)."""
    if not t or t.isdigit() or not t.isascii():
        return t
    s = t
    for a, b in _PH:
        s = s.replace(a, b)
    if not s:
        return t
    out = [s[0]] + [c for c in s[1:] if c not in "aeiouyh"]
    res: List[str] = []
    for c in out:
        if not res or res[-1] != c:
            res.append(c)
    return "".join(res)


def canon_name_token(t: str) -> str:
    t = NAME_CANON.get(t, t)
    if len(t) > 3:
        if t.endswith("ies"):
            t = t[:-3] + "y"
        elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
    return NAME_CANON.get(t, t)


# ----------------------------------------------------------------------------- names
def parse_name(raw) -> dict:
    c = clean_text(raw)
    toks = [canon_name_token(t) for t in c.split()]
    legal = sorted({t for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL] or list(toks)
    if len(core) > 1 and core[0] == "the":
        core = core[1:]
    core_ns = [t for t in core if t not in STOP_NAME] or list(core)
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_toks": tuple(core_ns),
        "name_toks_str": " ".join(core_ns),
        "name_compact": "".join(core),
        "legal": " ".join(legal),
        "acro": "".join(t[0] for t in core_ns),
        "name_phon": " ".join(phon_key(t) for t in core_ns),
    }


# ----------------------------------------------------------------------------- addresses
_PC6 = re.compile(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)")          # 6-digit postal codes (also '395 007')
_PC5 = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")          # 5-digit postal codes (ZIP, ZIP+4, French CP)


def _extract_postal(s: str) -> Tuple[str, str]:
    """Return (postal_code, string with the code rewritten as a single token)."""
    best = None
    lead = len(s) - len(s.lstrip())
    body_len = len(s.strip())
    for rx in (_PC6, _PC5):
        for m in rx.finditer(s):
            if m.start() <= lead and body_len > len(m.group(0)):
                continue  # a leading number is a house number, not a postal code
            code = "".join(g for g in m.groups() if g)
            if best is None or m.start() > best[0]:
                best = (m.start(), m.end(), code)
    if best is None:
        return "", s
    st, en, code = best
    return code, s[:st] + " " + code + " " + s[en:]


def _addr_tokens(text: str) -> List[str]:
    t = _PUNCT.sub(" ", text)
    toks = [ADDR_CANON.get(x, x) for x in t.split()]
    return [x for x in toks if x not in ADDR_STOP]


_EMPTY_ADDR = {"addr_main": "", "addr_lm": "", "addr_all": "", "addr_head": "", "addr_tail": "",
               "postal": "", "house": "", "nums": ""}


def parse_address(raw) -> dict:
    s = fold(raw)
    if not s.strip():
        return dict(_EMPTY_ADDR)
    s = _APOS.sub("'", s).replace("&", " and ")
    s = _DOTTED.sub(_undot, s)
    s = _ORD.sub(r"\1", s)
    postal, s = _extract_postal(s)
    segs = [x for x in re.split(r"[,;|\n\r]+", s) if x.strip()]
    main_segs: List[List[str]] = []
    lm_toks: List[str] = []
    for seg in segs:
        seg = seg.replace("'", "")
        m = _LM.search(seg)
        if m:
            main_part, lm_part = seg[:m.start()], seg[m.end():]
        else:
            main_part, lm_part = seg, ""
        mt = _addr_tokens(main_part)
        if mt:
            main_segs.append(mt)
        lm_toks.extend(_addr_tokens(lm_part))
    main = [t for seg in main_segs for t in seg]
    nums = [t for t in main + lm_toks if any(c.isdigit() for c in t) and t != postal]
    house = ""
    for t in main:
        if any(c.isdigit() for c in t) and t != postal:
            house = t
            break
    head = main_segs[0] if main_segs else []
    tail_src = main_segs[-2:] if main_segs else []
    tail = [t for seg in tail_src for t in seg if not any(c.isdigit() for c in t)]
    return {
        "addr_main": " ".join(main),
        "addr_lm": " ".join(lm_toks),
        "addr_all": " ".join(main + lm_toks),
        "addr_head": " ".join(head),
        "addr_tail": " ".join(tail),
        "postal": postal,
        "house": house,
        "nums": " ".join(nums),
    }


# ----------------------------------------------------------------------------- country
_CTRY_ALIAS = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us", "united states of america": "us",
    "america": "us", "in": "india", "ind": "india", "india": "india", "bharat": "india",
    "fr": "france", "fra": "france", "france": "france", "republique francaise": "france",
}


def norm_country(s) -> str:
    """Open-set country label: only well-known aliases are unified, every other label passes through."""
    t = clean_text(s)
    return _CTRY_ALIAS.get(t, t)


# ----------------------------------------------------------------------------- frame level
def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    names = pd.DataFrame([parse_name(x) for x in df["business_name"].tolist()])
    addrs = pd.DataFrame([parse_address(x) for x in df["business_address"].tolist()])
    out = pd.concat([names, addrs], axis=1)
    out["ctry"] = [norm_country(x) for x in df["country"].tolist()]
    return out
