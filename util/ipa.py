_close_vowels = "iyɨʉɯu"
_near_close_vowels = "ɪʏʊ"
_close_mid_vowels = "eøɘɵɤo"
_mid_vowels = "ə"
_open_mid_vowels = "ɛœɜɞʌɔ"
_near_open_vowels = "æɐ"
_open_vowels = "aɶäɑɒ"
_rhotic_vowels = "ɚɝɫ"
_vowels = (
    _close_vowels
    + _near_close_vowels
    + _close_mid_vowels
    + _mid_vowels
    + _open_mid_vowels
    + _near_open_vowels
    + _open_vowels
    + _rhotic_vowels
)
_plosive_consonants = "pbtdʈɖcɟkɡqɢʔ"
_nasal_consonants = "mɱnɳɲŋɴ"
_trill_consonants = "ʙrʀ"
_tap_or_flap_consonants = "ⱱɾɽ"
_fricative_consonants = "ɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦ"
_lateral_fricative_consonants = "ɬɮ"
_approximant_consonants = "ʋɹɻjɰ"
_lateral_approximant_consonants = "lɭʎʟ"
_pulmonic_consonants = (
    _plosive_consonants
    + _nasal_consonants
    + _trill_consonants
    + _tap_or_flap_consonants
    + _fricative_consonants
    + _lateral_fricative_consonants
    + _approximant_consonants
    + _lateral_approximant_consonants
)
_non_pulmonic_consonants = "ʘɓʼǀɗǃʄǂɠǁʛ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧ"
_stress = "ˈˌ"
_length = "\u0306ˑː"
_tone = "\u02e5\u02e6\u02e7\u02e8\u02e9\ua71b\ua71c"
_intonation = "|‖↗↘"
_suprasegmentals = _stress + _length + _tone + _intonation
_articulation = "\u033c\u032a\u0346\u033a\u033b\u031f\u0320\u0308\u033d\u031d\u031e"
_air_flow = "↑↓"
_phonation = "\u0324\u0325\u032c\u0330"
_rounding_and_labialization = "\u02b7\u1da3\u1d5d\u0339\u031c"
_syllabicity = "\u0329\u032f"
_consonant_release = "\u02b0\u207f\u02e1\u031a"
_co_articulation = "\u02b2\u02e0\u0334\u02e4\u0303\u02de"
_tongue_root = "\u0318\u0319"
_fortis_and_lenis = "\u0348\u0349"
_tie = "\u0361"
_numbers = "0123456789"
_diacritics = (
    _articulation
    + _air_flow
    + _phonation
    + _rounding_and_labialization
    + _syllabicity
    + _consonant_release
    + _co_articulation
    + _tongue_root
    + _fortis_and_lenis
    + _tie
    + _numbers
)
_phonemes = (
    _vowels
    + _pulmonic_consonants
    + _non_pulmonic_consonants
    + _other_symbols
    + _suprasegmentals
    + _diacritics
)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>", "<blank>"]
PUNCTUATION = "-;:,.!?¡¿—…\"'«»“” 、。・「」『』【】〔〕〈〉《》〝〞〟"
NON_PHONEMES = SPECIAL_TOKENS + list(PUNCTUATION)
PHONEMES = list(_phonemes)
VOCAB = NON_PHONEMES + PHONEMES
VOCAB_SIZE = len(VOCAB)
assert VOCAB_SIZE == len(set(VOCAB))
VOCAB_TO_ID = {char: idx for idx, char in enumerate(VOCAB)}

BUGGY_CHARACTERS = {"ε": "ɛ", "#": "\u0325", "ᵻ": "ɨ"}
# ε, or U+03B5, is the greek small letter epsilon
# ɛ, or U+025B, is the latin letter epsilon representing the open-mid front unrounded vowel
# The icelandic rules mistakenly put # instead of \u0325, the combining ring below
# https://en.wikipedia.org/wiki/Voicelessness
# ᵻ, or U+1D7B, the latin small capital letter i with stroke is an alternate
# representation of ɨ, or U+0268, the latin small letter i with stroke
# which is used to represent the close central unrounded vowel
# https://en.wikipedia.org/wiki/Close_central_unrounded_vowel

PRECOMPOSED_CHARACTERS = {
    "\u00c3": "a\u0303",  # LATIN CAPITAL LETTER A WITH TILDE
    "\u00e3": "a\u0303",  # LATIN SMALL LETTER A WITH TILDE
    "\u1ebc": "e\u0303",  # LATIN CAPITAL LETTER E WITH TILDE
    "\u1ebd": "e\u0303",  # LATIN SMALL LETTER E WITH TILDE
    "\u0128": "i\u0303",  # LATIN CAPITAL LETTER I WITH TILDE
    "\u0129": "i\u0303",  # LATIN SMALL LETTER I WITH TILDE
    "\u00d5": "o\u0303",  # LATIN CAPITAL LETTER O WITH TILDE
    "\u00f5": "o\u0303",  # LATIN SMALL LETTER O WITH TILDE
    "\u0168": "u\u0303",  # LATIN CAPITAL LETTER U WITH TILDE
    "\u0169": "u\u0303",  # LATIN SMALL LETTER U WITH TILDE
}
