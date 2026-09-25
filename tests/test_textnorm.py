"""Edge cases of the text normalisation (run: python -m pytest tests -q).

Every case here is a token that the previous, more aggressive cleaning destroyed or
mangled: country names that are part of the business name, articles that carry identity,
ambiguous two-letter legal forms, initials, abbreviations, and address state codes.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import textnorm  # noqa: E402
from textnorm import norm_addr, norm_name, normalize_record  # noqa: E402


def core(name, country=None):
    return norm_name(name, country)["n_core"]


def full(name, country=None):
    return norm_name(name, country)["n_full"]


# ------------------------------------------------------------------ country tokens
@pytest.mark.parametrize("name,country,expected_core", [
    ("Air India Ltd", "India", "air india"),
    ("Reliance India", "India", "reliance india"),
    ("India Cements Limited", "India", "india cements"),
    ("Air France SAS", "France", "air france"),
    ("Toys R Us", "US", "toys r us"),
    ("Us Engineering Private Limited", "US", "us engineering"),
])
def test_country_token_kept_when_part_of_name(name, country, expected_core):
    assert core(name, country) == expected_core


def test_parenthesised_country_tag_removed_everywhere():
    r = norm_name("Gulabchand Joy (India) Limited", "India")
    assert r["n_core"] == "gulabchand joy"
    assert "india" not in r["n_full"]


def test_ocr_corrupted_country_tag_removed():
    # "(lndia)" is the corrupted "(India)" seen in Source 2/3 records
    assert core("Limited Gulabchand Joy (lndia)", "India") == "gulabchand joy"


def test_parenthesised_non_country_content_kept():
    assert core("Elegance (Events) Private Limited", "India") == "elegance events"


def test_trailing_country_tag_dropped_only_from_long_names():
    assert core("Tata Consultancy Services India", "India") == "tata consultancy svcs"
    assert "india" in full("Tata Consultancy Services India", "India")
    # short names keep the trailing country token: dropping it would change the identity
    assert core("Air India", "India") == "air india"


# ------------------------------------------------------------------ articles / short tokens
@pytest.mark.parametrize("name,expected_core", [
    ("The Dent Diner", "dent diner"),
    ("The Gap Inc", "the gap"),
    ("The One", "the one"),
    ("La Poste", "la poste"),
    ("El Lincoln", "el lincoln"),
    ("Le Montessori School", "montessori school"),
])
def test_articles_kept_when_name_would_collapse(name, expected_core):
    assert core(name) == expected_core


def test_article_always_kept_in_full_name():
    assert full("The Dent Diner") == "the dent diner"


# ------------------------------------------------------------------ initials / abbreviations
@pytest.mark.parametrize("name,expected_core", [
    ("A & W Restaurants", "aw restaurants"),
    ("D & L Metro LLC", "dl metro"),
    ("J.P. Morgan Chase", "jp morgan chase"),
    ("L A Fitness", "la fitness"),
    ("U Q & Z Kimco", "uqz kimco"),
])
def test_initials_merged(name, expected_core):
    assert core(name) == expected_core


def test_initials_and_compact_form_agree():
    assert core("D&L Metro") == core("D & L Metro") == core("DL Metro")


def test_abbreviations_collapse_to_short_form_not_expanded():
    assert core("Tech Mahindra") == "tech mahindra"
    assert core("Bharat Jai Technologies") == "bharat jai tech"
    assert core("Bharat Jai Technology") == "bharat jai tech"
    assert core("Acme International") == core("Acme Intl")
    assert core("Acme Manufacturing Co") == core("Acme Mfg")


# ------------------------------------------------------------------ legal suffixes
@pytest.mark.parametrize("name,expected_core,expected_legal", [
    ("PC World Ltd", "pc world", "ltd"),
    ("Crystal Lending PC", "crystal lending", "pc"),
    ("Ag Supply Inc", "ag supply", "inc"),
    ("SA Toys", "sa toys", ""),
    ("Tiffany & Co", "tiffany", "co"),
    ("Namaskar & Co Company", "namaskar", "co"),
    ("PA Impex Private Limited", "pa impex", "ltd pvt"),
    ("Chaturthi Digital Pvt Ltd", "chaturthi digital", "ltd pvt"),
    ("Chaturthi Digital Private Limited", "chaturthi digital", "ltd pvt"),
])
def test_ambiguous_legal_tokens_are_contextual(name, expected_core, expected_legal):
    r = norm_name(name)
    assert r["n_core"] == expected_core
    assert r["n_legal"] == expected_legal


def test_strict_legal_token_removed_even_when_shuffled():
    # corrupted records shuffle tokens: the legal form can land in the middle
    assert core("Federal LLC Minerals Star") == "federal minerals star"
    assert core("Private Bhiwandi Refinery Limited") == "bhiwandi refinery"


def test_legal_conflict_information_preserved():
    assert norm_name("Acme LLC")["n_legal"] == "llc"
    assert norm_name("Acme Inc")["n_legal"] == "inc"


# ------------------------------------------------------------------ misc name handling
def test_id_tag_and_messrs_prefix_removed():
    assert core("M/s Ram Traders (ID: 123)") == "ram traders"


def test_square_brackets_are_not_removed_content():
    assert core("Total [Farms]") == "total farms"


def test_domain_name_extracted():
    r = norm_name("mérrittentertainment.com")
    assert r["n_domain"] == "merrittentertainment"


def test_alias_parts_split():
    r = norm_name("Acme Holdings DBA: Bob's Diner")
    assert r["f_alias"] is True
    assert r["n_parts"] == "acme holdings|bobs diner"


def test_possessive_apostrophe_removed():
    assert core("Orelee's Barbershop") == "orelees barbershop"


def test_empty_name():
    r = norm_name(None, "US")
    assert r["n_core"] == "" and r["n_full"] == ""


# ------------------------------------------------------------------ addresses
def test_state_code_components_kept():
    # "DE" and "LA" are Delaware / Louisiana here, not French articles
    assert "de" in norm_addr("123 Main St, Wilmington, DE", "US")["a_comp"].split(",")
    assert "la" in norm_addr("2571 Lark Street, Baton Rouge, LA", "US")["a_comp"].split(",")


def test_article_dropped_inside_component():
    assert norm_addr("12 rue de la Paix, Paris", "France")["a_comp"].startswith("12 rue paix")


def test_french_abbreviations_only_in_france():
    assert "rue" in norm_addr("12 r du Bac, Paris", "France")["a_norm"].split()
    assert "rue" not in norm_addr("R K Puram, New Delhi", "India")["a_norm"].split()
    assert "r" in norm_addr("R K Puram, New Delhi", "India")["a_norm"].split()


def test_full_state_name_canonicalised():
    a = norm_addr("713 Butternut Street, Syracuse, New York", "US")
    b = norm_addr("713 Butternut St, Syracuse, NY", "US")
    assert a["a_comp"] == b["a_comp"]


def test_house_number_leading_zero_stripped():
    assert norm_addr("011014 Island Dr, Town Of Gibraltar, Wisconsin", "US")["a_hn"] == "11014"


def test_normalize_record_contract():
    r = normalize_record("Acme Inc", "1 Main St, Springfield, IL", "US")
    for k in ("n_full", "n_core", "n_parts", "n_compact", "n_domain", "n_legal", "f_indic", "f_alias",
              "a_norm", "a_comp", "a_num", "a_hn", "a_street", "a_key", "feat_name", "feat_addr", "feat_cross"):
        assert k in r


def test_indic_lexicon_applied():
    textnorm.set_lexicon({"name": {textnorm.indic_key("राम"): "ram"}, "addr": {}})
    try:
        assert core("राम मार्केटिंग", "India").split()[0] == "ram"  # second token needs unidecode (optional)
    finally:
        textnorm.set_lexicon({"name": {}, "addr": {}})
