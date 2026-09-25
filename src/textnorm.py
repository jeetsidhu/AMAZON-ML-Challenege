"""Country-agnostic text normalisation for business names and addresses.

Everything here is rule based (hand-written abbreviation tables) plus an optional
Indic-script -> Latin lexicon that is *learned from the training ground truth*
(see build_lexicon.py). No external data or services are used.
"""
import re
import unicodedata

from rapidfuzz import fuzz

try:
    from unidecode import unidecode
except ImportError:  # offline Kaggle image without the package: accent folding only
    def unidecode(s):
        """Fallback: strip diacritics of Latin text (NFKD). Indic script cannot be transliterated this
        way; such tokens keep their own script unless the learned lexicon covers them."""
        return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

INDIC_RE = re.compile(r"[ऀ-෿]")
ZW_RE = re.compile(r"[​-‏⁠﻿­]")
ID_RE = re.compile(r"\(\s*id\s*:\s*\d+\s*\)", re.I)
ALIAS_RE = re.compile(
    r"\s*(?:\||\b(?:d\s*/\s*b\s*/\s*a|a\s*/\s*k\s*/\s*a|f\s*/\s*k\s*/\s*a|t\s*/\s*a)\b\.?"
    r"|\b(?:dba|formerly|aka|fka|also known as|trading as)\b\s*:?)\s*",
    re.I,
)
MS_RE = re.compile(r"\bm\s*/\s*s\b\.?", re.I)
NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|\bnone\b|\bc\s*/\s*o\b\.?", re.I)
# postal boxes / sorting-office codes (PO Box, French BP / CS / Cedex) and the French "N°" number sign
POBOX_RE = re.compile(r"\b(?:p\.?\s*o\.?\s*box|post\s*box|bp|cs)\s*[.:#]?\s*\d+\b|\bcedex(?:\s*\d+)?\b", re.I)
NUMSIGN_RE = re.compile(r"\bn\s*[°º]", re.I)
DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co\.in|in|co|biz|info|us|fr|io)$"
)
NONALNUM_RE = re.compile(r"[^a-z0-9]+")
DIGITS_RE = re.compile(r"\d+")

# ---------------------------------------------------------------- name tables
# Canonical forms collapse spelling variants onto ONE short form. The direction is always
# long -> short: "technologies" / "technology" -> "tech" (never "tech" -> "technologies",
# which would turn "Tech Mahindra" into "technologies mahindra"). A short token is therefore
# never expanded into something it may not mean; only unambiguous long forms are shortened.
NAME_CANON = {
    "incorporated": "inc", "corporation": "corp", "company": "co", "cos": "co",
    "limited": "ltd", "private": "pvt", "centre": "center", "brothers": "bros",
    "and": "&", "et": "&", "sri": "shri", "shree": "shri", "international": "intl",
    "manufacturing": "mfg", "services": "svcs", "service": "svc", "associates": "assoc",
    "technologies": "tech", "technology": "tech", "hospital": "hosp",
    "etablissements": "ets", "etablissement": "ets", "etabl": "ets", "compagnie": "cie",
    "societe": "ste", "pvtltd": "pvt ltd", "corpn": "corp",
}
# Legal-form tokens. STRICT ones are never anything but a legal form, so they are removed
# from the core name wherever they appear (corrupted records shuffle tokens: "Federal LLC
# Minerals Star"). WEAK ones are also ordinary words or initials ("PC World", "Ag Supply",
# "SA Toys", "Co-op", "PA Impex"), so they are only removed in a legal *position*: at the end
# of the name, right after "&" ("Tiffany & Co"), or next to another legal token ("Co Ltd").
STRICT_LEGAL = {
    "inc", "corp", "ltd", "pvt", "llc", "llp", "plc", "pllc", "ltda", "sarl", "sas", "sasu",
    "eurl", "gmbh", "opc", "selarl", "scop", "gie", "snc", "sca", "scm", "sci",
}
WEAK_LEGAL = {"co", "lp", "pc", "pa", "sa", "ag", "bv", "nv", "ei", "cie"}
LEGAL = STRICT_LEGAL | WEAK_LEGAL
# Articles / connectives / honorifics: dropped from the core name only when at least two
# other tokens remain, so "The One", "La Poste", "El Lincoln" keep their identity, while
# "The Dent Diner" -> "dent diner". They are always kept in the full name.
NAME_STOP = {
    "the", "of", "&", "mr", "mrs", "ms", "dr", "smt", "messrs", "de", "du", "des", "la", "le",
    "les", "el", "los", "las", "d", "l", "a", "an", "en", "au", "aux", "for",
}
MIN_CORE = 2
MIN_CORE_COUNTRY = 3  # tokens that must remain before a trailing country token is treated as a tag
PAREN_RE = re.compile(r"\(([^()]*)\)")

# ------------------------------------------------------------- address tables
ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "avenu": "ave",
    "boulevard": "blvd", "boul": "blvd", "bld": "blvd", "drive": "dr", "court": "ct",
    "lane": "ln", "place": "pl", "circle": "cir", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "trail": "trl", "square": "sq", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "suite": "ste", "apartment": "apt", "appartement": "apt", "appt": "apt", "building": "bldg",
    "floor": "fl", "flr": "fl", "number": "no", "nr": "near", "opposite": "opp", "saint": "st",
    "sainte": "ste", "mount": "mt", "fort": "ft", "point": "pt", "route": "rte", "expressway": "expy",
    "freeway": "fwy", "crossing": "xing", "impasse": "imp", "allee": "all",
    "allees": "all", "chemin": "che", "cours": "crs", "quai": "qu", "residence": "res",
    "faubourg": "fbg", "chaussee": "chau", "hno": "no", "house": "h", "sector": "sec",
    "sect": "sec", "nagar": "ngr", "marg": "mg", "colony": "col", "district": "dist",
    "township": "twp", "junction": "jn", "extension": "extn", "ext": "extn", "gali": "gali",
    "bengaluru": "bangalore", "gurugram": "gurgaon", "calcutta": "kolkata", "bombay": "mumbai",
    "madras": "chennai", "centre": "center", "first": "1st", "second": "2nd", "third": "3rd",
    "bvld": "blvd", "bvd": "blvd", "etage": "fl", "ndeg": "no",
}
# One- and two-letter French street abbreviations are only unambiguous in France: "R K Puram"
# (an Indian locality) must not become "rue k puram", and "Q" / "CH" are ordinary tokens elsewhere.
ADDR_CANON_FR = {"r": "rue", "q": "qu", "ch": "che", "bd": "blvd", "av": "ave"}
FR_COUNTRY_TOKENS = {"france", "fr"}
ADDR_STOP = {
    "de", "du", "des", "la", "le", "les", "d", "l", "of", "the", "and", "&", "au", "aux",
    "no", "eme",
}
# canonical street-type / direction / unit words: never used as "key" tokens in combined blocking keys
ADDR_GENERIC = set(ADDR_CANON.values()) | set(ADDR_CANON_FR.values()) | {
    "rue", "city", "town", "county", "village", "unit", "apt", "ste", "fl", "bldg", "floor", "po",
    "box", "near", "opp", "road", "main", "cross", "center", "plot", "flat", "shop", "office",
    "phase", "block", "complex", "tower", "society", "market", "chowk", "bazar", "industrial",
    "estate", "area", "layout", "stage", "behind", "post", "village",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
    "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu & kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "ladakh": "la", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "andaman and nicobar islands": "an", "lakshadweep": "ld",
}

FR_REGIONS = {
    "ara": ["auvergne rhone alpes", "ain", "allier", "ardeche", "cantal", "drome", "isere", "loire",
            "haute loire", "puy de dome", "rhone", "savoie", "haute savoie"],
    "bfc": ["bourgogne franche comte", "cote d or", "doubs", "jura", "nievre", "haute saone",
            "saone et loire", "yonne", "territoire de belfort"],
    "bre": ["bretagne", "cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["centre val de loire", "cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "cor": ["corse", "corse du sud", "haute corse"],
    "ges": ["grand est", "ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse",
            "moselle", "bas rhin", "haut rhin", "vosges"],
    "hdf": ["hauts de france", "aisne", "nord", "oise", "pas de calais", "somme"],
    "idf": ["ile de france", "seine et marne", "yvelines", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d oise"],
    "nor": ["normandie", "calvados", "eure", "manche", "orne", "seine maritime"],
    "naq": ["nouvelle aquitaine", "charente", "charente maritime", "correze", "creuse", "dordogne",
            "gironde", "landes", "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne",
            "haute vienne"],
    "occ": ["occitanie", "ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault",
            "lot", "lozere", "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "pac": ["provence alpes cote d azur", "paca", "alpes de haute provence", "hautes alpes",
            "alpes maritimes", "bouches du rhone", "var", "vaucluse"],
}
FR_ADMIN = {name: "fr" + code for code, names in FR_REGIONS.items() for name in names}

# Learned Indic lexicons (populated by set_lexicon in each worker process).
_NAME_LEX = {}
_ADDR_LEX = {}


def set_lexicon(lex):
    global _NAME_LEX, _ADDR_LEX
    if lex:
        _NAME_LEX = lex.get("name", {})
        _ADDR_LEX = lex.get("addr", {})


def indic_key(tok):
    """Canonical lookup key for an Indic-script token/component."""
    tok = ZW_RE.sub("", unicodedata.normalize("NFC", tok))
    return "".join(ch for ch in tok if ch.isalnum() or unicodedata.category(ch)[0] == "M").strip()


def to_ascii(s):
    return unidecode(ZW_RE.sub("", s)).lower()


def latin_tokens(s):
    return [t for t in NONALNUM_RE.split(s) if t]


def _translit_name(raw):
    """Replace Indic tokens with their learned Latin equivalents (fallback: unidecode)."""
    out = []
    for tok in raw.split():
        if INDIC_RE.search(tok):
            lat = _NAME_LEX.get(indic_key(tok))
            out.append(lat if lat is not None else to_ascii(tok))
        else:
            out.append(tok)
    return " ".join(out)


def _country_tag(text, country_toks):
    """True when a parenthesised chunk is the record's own country label, e.g. "(India)",
    including OCR-style corruptions such as "(lndia)" / "(1ndia)" (ratio >= 80)."""
    if not country_toks:
        return False
    toks = latin_tokens(text)
    if not toks:
        return False
    a, b = "".join(toks), "".join(country_toks)
    return a == b or (len(a) >= 3 and fuzz.ratio(a, b) >= 80)


def _merge_initials(toks):
    """Runs of single-letter tokens (optionally glued by "&") become one token:
    "j p morgan" -> "jp morgan", "d & l metro" -> "dl metro", "l a fitness" -> "la fitness".
    A lone single letter next to a longer token is left alone ("orelee s barbershop")."""
    out, run = [], []

    def flush():
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for t in toks:
        if len(t) == 1 and t != "&":
            run.append(t)
        elif t == "&" and run:
            continue  # "&" between initials is glue; it is dropped from the core anyway
        else:
            flush()
            out.append(t)
    flush()
    return out


def _strip_legal(toks):
    """Removes legal-form tokens from the core name (see STRICT_LEGAL / WEAK_LEGAL).
    Returns (remaining tokens, removed legal tokens)."""
    legal, keep = [], []
    n = len(toks)
    for i, t in enumerate(toks):
        if t in STRICT_LEGAL:
            legal.append(t)
            continue
        if t in WEAK_LEGAL:
            trailing = all(x in LEGAL or x in NAME_STOP for x in toks[i + 1:])
            after_amp = i > 0 and toks[i - 1] == "&"
            next_legal = (i + 1 < n and toks[i + 1] in LEGAL) or (i > 0 and toks[i - 1] in LEGAL)
            if trailing or after_amp or next_legal:
                legal.append(t)
                continue
        keep.append(t)
    return keep, legal


def _strip_trailing_country(toks, country_toks):
    """"Tata Consultancy Services India" -> drop the trailing country token, but only when
    at least MIN_CORE_COUNTRY other content tokens remain: "Air India", "Reliance India" and
    "Toys R Us" keep it. Being conservative is cheap: an extra trailing token on one side is
    handled by the token-set / extra-token features, whereas a deleted token is gone."""
    k = len(country_toks)
    if not k or len(toks) < k + MIN_CORE_COUNTRY:
        return toks
    if toks[-k:] == list(country_toks):
        rest = toks[:-k]
        if sum(1 for t in rest if t not in NAME_STOP) >= MIN_CORE_COUNTRY:
            return rest
    return toks


def _name_tokens(part, country_toks=()):
    """Normalise one alias part of a name -> (full tokens, core tokens, domain stems, legal tokens).

    `country_toks` are the tokens of the record's own country label. A parenthesised country tag
    "(India)" carries no identity and is removed everywhere; a bare country token is kept in the
    full name and only dropped from the core when it is a trailing location tag (see
    _strip_trailing_country) -- "Air India" is not "Air".
    """
    s = to_ascii(part)
    s = PAREN_RE.sub(lambda m: " " if _country_tag(m.group(1), country_toks) else " " + m.group(1) + " ", s)
    s = s.replace("&", " & ").replace("+", " & ")
    domains = []
    toks = []
    for raw in s.split():
        m = DOMAIN_RE.match(raw.strip(".,;:()[]{}<>*#-_\"'"))
        if m:
            domains.append(m.group(1).replace("-", ""))
            continue
        raw = raw.replace(".", "").replace("'", "").replace("\u2019", "")
        for t in latin_tokens(raw) if raw != "&" else ["&"]:
            toks.extend(NAME_CANON.get(t, t).split())
    toks = _merge_initials(toks)
    full = [t for t in toks if t != "&"]
    core, legal = _strip_legal(toks)
    core = _strip_trailing_country(core, tuple(country_toks))
    content = [t for t in core if t not in NAME_STOP]
    if len(content) >= MIN_CORE:
        core = content
    else:
        core = [t for t in core if t != "&"]
    core = list(dict.fromkeys(core))
    legal = [t for t in legal if t != "cie"]
    return full, core, domains, legal


def norm_name(raw, country=None):
    """Returns dict with normalised name fields."""
    if raw is None:
        raw = ""
    country_toks = tuple(latin_tokens(to_ascii(country))) if country else ()
    is_indic = bool(INDIC_RE.search(raw))
    s = ID_RE.sub(" ", raw)
    s = MS_RE.sub(" ", s)
    if is_indic:
        s = _translit_name(s)
    parts = [p for p in ALIAS_RE.split(s) if p and p.strip()]
    if not parts:
        parts = [""]
    fulls, cores, compacts, all_domains, legals = [], [], [], [], []
    for p in parts:
        full, core, domains, legal = _name_tokens(p, country_toks)
        if full or core:
            fulls.append(" ".join(full))
            cores.append(" ".join(core))
            if core:
                compacts.append("".join(core))
            if full:
                compacts.append("".join(full))
        all_domains.extend(domains)
        legals.extend(legal)
    compacts.extend(all_domains)
    # flatten: primary string uses every part (aliases are alternative names of the same entity)
    all_core, seen = [], set()
    for c in cores:
        for t in c.split():
            if t not in seen:
                seen.add(t)
                all_core.append(t)
    return {
        "n_full": " ".join(fulls),
        "n_core": " ".join(all_core),
        "n_parts": "|".join(c for c in cores if c),
        "n_compact": "|".join(dict.fromkeys(c for c in compacts if c)),
        "n_domain": "|".join(all_domains),
        "n_legal": " ".join(sorted(set(legals))),
        "f_indic": is_indic,
        "f_alias": len(parts) > 1,
    }


def _translit_addr_component(comp):
    if INDIC_RE.search(comp):
        lat = _ADDR_LEX.get(indic_key(comp))
        if lat is not None:
            return lat
        return to_ascii(comp)
    return comp


UNIT_WORDS = {"unit", "apt", "ste", "fl", "bldg", "room", "rm", "po", "box"}
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values()) | set(FR_ADMIN.values())


def norm_addr(raw, country=None):
    """Returns dict: a_norm (tokens), a_comp (components joined by ','), a_num (digit groups),
    a_hn (house number: first digit group of the street line), a_street (non-numeric tokens of
    the street line) and a_key (distinctive non-numeric tokens used in combined blocking keys).

    `country` gates the ambiguous French abbreviations (ADDR_CANON_FR)."""
    if raw is None:
        raw = ""
    canon = ADDR_CANON
    if country and to_ascii(country).strip() in FR_COUNTRY_TOKENS:
        canon = {**ADDR_CANON, **ADDR_CANON_FR}
    s = NULL_RE.sub(" ", raw)
    s = POBOX_RE.sub(" ", s)
    s = NUMSIGN_RE.sub(" ", s)
    comps = []
    for comp in s.split(","):
        comp = _translit_addr_component(comp.strip())
        c = to_ascii(comp).replace("&", " and ").replace("'", " ").replace("’", " ")
        toks = latin_tokens(c)
        if not toks:
            continue
        joined = " ".join(toks)
        st = US_STATES.get(joined) or IN_STATES.get(joined) or FR_ADMIN.get(joined)
        if st:
            comps.append([st])
            continue
        out = []
        for t in toks:
            if t.isdigit():
                t = t.lstrip("0") or "0"
            t = canon.get(t, t)
            if t in ADDR_STOP and len(toks) > 1:
                # articles are dropped inside a component, but a component that IS the token is
                # kept: "DE" (Delaware) and "LA" (Louisiana) are state codes, not French articles
                continue
            out.append(t)
        if out:
            comps.append(out)
    flat = [t for c in comps for t in c]
    nums = DIGITS_RE.findall(" ".join(flat))
    numbered = [c for c in comps if any(ch.isdigit() for t in c for ch in t)]
    street = next((c for c in numbered if c[0] not in UNIT_WORDS), numbered[0] if numbered else [])
    hn = next((m.group(0) for t in street for m in [DIGITS_RE.search(t)] if m), "")
    street_words = [t for t in street if not any(ch.isdigit() for ch in t)]
    key = []
    for t in street_words + [t for c in comps if c is not street for t in c]:
        if len(t) >= 3 and t not in ADDR_GENERIC and t not in STATE_CODES \
                and not any(ch.isdigit() for ch in t) and t not in key:
            key.append(t)
    return {
        "a_norm": " ".join(flat),
        "a_comp": ",".join(" ".join(c) for c in comps),
        "a_num": " ".join(dict.fromkeys(nums)),
        "a_hn": hn,
        "a_street": " ".join(street_words),
        "a_key": " ".join(key[:6]),
    }


def blocking_features(n, a):
    """Space separated blocking features for one record.

    name block   : n:<core token>, p:<order-free token pair>, k:<compact-name prefix>
    address block: a:<token>, b:<adjacent bigram>, h:<house number>_<key token>
    cross block  : c:<name token>_<key address token>  (separates namesakes in different places)
    """
    nf, af, cf = [], [], []
    for part in n["n_parts"].split("|"):
        toks = part.split()
        for t in toks:
            nf.append("n:" + t)
        head = toks[:6]
        for i in range(len(head)):
            for j in range(i + 1, len(head)):
                x, y = (head[i], head[j]) if head[i] < head[j] else (head[j], head[i])
                nf.append("p:" + x + "_" + y)
    for c in n["n_compact"].split("|"):
        if len(c) >= 5:
            nf.append("k:" + c[:8])
    for comp in a["a_comp"].split(","):
        toks = comp.split()
        for t in toks:
            af.append("a:" + t)
        for i in range(len(toks) - 1):
            af.append("b:" + toks[i] + "_" + toks[i + 1])
    key = a["a_key"].split()[:5]
    if a["a_hn"]:
        for k in key:
            af.append("h:" + a["a_hn"] + "_" + k)
    for t in n["n_core"].split()[:3]:
        for k in key:
            cf.append("c:" + t + "_" + k)
    return " ".join(dict.fromkeys(nf)), " ".join(dict.fromkeys(af)), " ".join(dict.fromkeys(cf))


def normalize_record(name, addr, country=None):
    n = norm_name(name, country)
    a = norm_addr(addr, country)
    fn, fa, fc = blocking_features(n, a)
    n.update(a)
    n["feat_name"] = fn
    n["feat_addr"] = fa
    n["feat_cross"] = fc
    return n
