import re
from pathlib import Path

import polars as pl


class AbbrExpander:
    def __init__(self, abbr_file: Path):
        self.abbr: dict[str, dict[str, str]] = {}
        self.patterns: dict[str, re.Pattern] = {}
        self.load_abbr(abbr_file)

    def load_abbr(self, abbreviations_file: Path):
        df = pl.read_csv(abbreviations_file)
        for language in df["language"].unique():
            language = language.lower()
            filtered_df = df.filter(pl.col("language").str.to_lowercase() == language)
            self.abbr[language] = dict(
                zip(filtered_df["abbreviation"], filtered_df["expansion"])
            )
            join_abbr = "|".join(map(re.escape, filtered_df["abbreviation"]))
            self.patterns.setdefault(
                language, re.compile(rf"(?i)((?<=^)|(?<=\s))({join_abbr})(?=$|\s)")
            )

    def replace_abbr(self, text: str, language: str):
        language = language.lower()
        if language in self.patterns:
            return self.patterns[language].sub(
                lambda match: self.abbr[language][match.group(2).lower()], text
            )
        else:
            return text
