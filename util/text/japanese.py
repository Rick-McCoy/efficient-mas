import re

from pykakasi import Kakasi
from pyopenjtalk import OpenJTalk


def handle_iteration_mark(text: str) -> str:
    """
    Replace the IDEOGRAPHIC ITERATION MARK (U+3005) with its appropriate reading.
    The mark is used to represent the repetition of a kanji character.
    """
    return re.sub(r"(.)\u3005", r"\1\1", text)


def run_openjtalk(text: str, openjtalk: OpenJTalk):
    """
    Normalize years, numbers, and other numerical expressions in the given text.
    Uses OpenJTalk to process the text.

    OpenJTalk's function run_frontend() returns a list of dictionaries which contain
    various information about the text, such as the pronunciation, part of speech, and
    other linguistic information.
    This detects numerical expressions and converts them to their Kanji representation.

    However, some phrases have special readings that are not supported by PyKakasi.
    These are therefore handled separately.
    These include:
    - Units
        - ヶ月 (months)
        - ヶ所 (places)
        - ヶ国 (countries)
    - Roman numerals
    - Ordinal numbers
    - Numbers with counters
    - Numbers with units
    """
    openjtalk_processed = openjtalk.run_frontend(text)
    return "".join(
        item["string"] if item["pos_group2"] != "助数詞" else item["pron"]
        for item in openjtalk_processed
    )


def run_kakasi(text: str, kakasi: Kakasi):
    """
    Convert Kanji characters to Katakana pronunciation.
    Uses Kakasi to process the text.
    """
    return "".join(item["kana"] for item in kakasi.convert(text))
