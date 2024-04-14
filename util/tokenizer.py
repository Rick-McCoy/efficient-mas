import logging
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

from util.cleaner import TextProcessor
from util.ipa import BUGGY_CHARACTERS, PRECOMPOSED_CHARACTERS, VOCAB, VOCAB_TO_ID
from util.text.common import get_backend, handle_unknown_char


class Tokenizer:
    def __init__(
        self,
        default_language: str = "en-us",
        add_blank: bool = False,
        pad_sos_eos: bool = False,
    ):
        self.logger = logging.getLogger("phonemizer")
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except ValueError:
            output_dir = Path("outputs")

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(f"{output_dir}/dataset.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.backends = {default_language: get_backend(default_language, self.logger)}
        self.text_processor = TextProcessor(default_language=default_language)
        self.add_blank = add_blank
        self.pad_sos_eos = pad_sos_eos

        self.default_language = default_language
        self.not_found_characters: set[str] = set()

    def phonemize(self, texts: list[str], language: str | None = None):
        language = language or self.default_language
        if language not in self.backends:
            self.backends[language] = get_backend(language, self.logger)

        texts = [
            self.text_processor.phoneme_cleaners(text, language=language)
            for text in texts
        ]

        phonemized = self.backends[language].phonemize(texts)
        phonemized = [" ".join(phonemes.split()) for phonemes in phonemized]

        for key, value in BUGGY_CHARACTERS.items():
            phonemized = [text.replace(key, value) for text in phonemized]

        for key, value in PRECOMPOSED_CHARACTERS.items():
            phonemized = [text.replace(key, value) for text in phonemized]

        return phonemized

    def encode(self, text: str) -> list[int]:
        """Encodes a string of text as a sequence of IDs."""
        missing_chars = set(text) - set(VOCAB)
        if missing_chars - self.not_found_characters:
            self.not_found_characters |= missing_chars
            self.logger.debug(f" [!] Processing text: {text}")
            self.logger.debug(
                f" [!] Characters {missing_chars} not found in the vocabulary."
            )
            for char in missing_chars:
                self.logger.debug(f"     - {char} (U+{ord(char):05X})")

        token_ids = list(map(handle_unknown_char, text))

        return token_ids

    def phonemes_to_ids(self, phonemes: list[str]):
        ids = map(self.encode, phonemes)

        if self.add_blank:
            ids = map(self.intersperse_blank_char, ids)

        if self.pad_sos_eos:
            ids = map(self.pad_with_sos_eos, ids)

        return list(ids)

    def pad_with_sos_eos(self, char_sequence: list[int]):
        """Pads a sequence with the special start and end of sequence tokens."""
        return [VOCAB_TO_ID["<sos>"], *char_sequence, VOCAB_TO_ID["<eos>"]]

    def intersperse_blank_char(
        self, char_sequence: list[int], use_blank_char: bool = True
    ):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = VOCAB_TO_ID["<blank>" if use_blank_char else "<pad>"]
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result
