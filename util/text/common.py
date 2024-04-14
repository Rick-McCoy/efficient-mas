from logging import Logger

from phonemizer.backend import EspeakBackend

from util.ipa import PUNCTUATION, VOCAB_TO_ID


def normalize_with_dictionary(text: str, dic: dict[str, str]):
    for key in dic:
        text = text.replace(key, dic[key])
    return text


def handle_unknown_char(char: str):
    return VOCAB_TO_ID.get(char, VOCAB_TO_ID["<unk>"])


def get_backend(lang: str, logger: Logger):
    return EspeakBackend(
        language=lang,
        preserve_punctuation=True,
        with_stress=True,
        tie=True,
        language_switch="remove-flags",
        logger=logger,
        punctuation_marks=PUNCTUATION,
    )
