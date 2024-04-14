import re
from pathlib import Path

from pykakasi import Kakasi
from pyopenjtalk import OPEN_JTALK_DICT_DIR, OpenJTalk, _lazy_init

from util.expand.abbreviations import AbbrExpander
from util.text.common import normalize_with_dictionary
from util.text.common_dict import ROMAN_NUMERAL_DICT
from util.text.japanese import (
    handle_iteration_mark,
    run_kakasi,
    run_openjtalk,
)
from util.text.japanese_dict import (
    ASCII_DICT,
    DAKUTEN_DICT,
    FULL_WIDTH_DICT,
    HALF_WIDTH_DICT,
    MISC_JA_DICT,
    SHINJITAI_DICT,
    SMALL_KATAKANA_DICT,
    UNIT_ABBREV_DICT,
)
from util.text.korean import (
    normalize_english,
    normalize_number,
    normalize_quote,
    sanitize,
)
from util.text.korean_dict import ETC_DICT, SPECIAL_DICT, UPPER_DICT


class TextProcessor:
    def __init__(self, default_language: str):
        self.default_language = default_language
        self.ab_expander = AbbrExpander(
            Path(__file__).parent / "expand/abbreviations.csv"
        )
        _lazy_init()
        self.openjtalk = OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR)
        self.kakasi = Kakasi()

    def collapse_whitespace(self, text: str) -> str:
        return " ".join(text.split())

    def remove_aux_symbols(self, text: str) -> str:
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        return text

    def common_clean(self, text: str):
        text = sanitize(text)
        text = self.remove_aux_symbols(text)
        text = self.collapse_whitespace(text)
        text = normalize_with_dictionary(text, ROMAN_NUMERAL_DICT)
        return text

    def phoneme_cleaners(self, text: str, language: str | None = None):
        language = language or self.default_language
        text = self.common_clean(text)
        match language:
            case "en-us" | "en":
                text = self.clean_english(text)
            case "ko":
                text = self.clean_korean(text)
            case "ja":
                text = self.clean_japanese(text)
            case "es":
                text = self.clean_spanish(text)
            case _:
                print(f"Language {language} not supported")

        return text

    def clean_english(self, text: str):
        text = self.ab_expander.replace_abbr(text, language="en")
        return text

    def clean_spanish(self, text: str):
        text = self.ab_expander.replace_abbr(text, language="es")
        return text

    def clean_korean(self, text: str):
        text = normalize_with_dictionary(text, ETC_DICT)
        text = normalize_with_dictionary(text, SPECIAL_DICT)
        text = normalize_english(text)
        text = normalize_number(text)
        text = normalize_quote(text)
        text = normalize_with_dictionary(text.upper(), UPPER_DICT)
        return text

    def clean_japanese(self, text: str):
        text = handle_iteration_mark(text)
        text = normalize_with_dictionary(text, SHINJITAI_DICT)
        text = run_openjtalk(text, self.openjtalk)
        text = run_kakasi(text, self.kakasi)
        text = normalize_with_dictionary(text, FULL_WIDTH_DICT)
        text = normalize_with_dictionary(text, HALF_WIDTH_DICT)
        text = normalize_with_dictionary(text, ASCII_DICT)
        text = normalize_with_dictionary(text, DAKUTEN_DICT)
        text = normalize_with_dictionary(text, SMALL_KATAKANA_DICT)
        text = normalize_with_dictionary(text, UNIT_ABBREV_DICT)
        text = normalize_with_dictionary(text, MISC_JA_DICT)
        return text
