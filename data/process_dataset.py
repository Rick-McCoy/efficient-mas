import argparse
import json
from pathlib import Path

import polars as pl
from beartype.door import is_bearable
from tqdm import tqdm


def main(path: Path):
    source_path = path / "source"
    label_path = path / "label"

    assert source_path.exists(), f"Source path {source_path} does not exist"
    assert label_path.exists(), f"Label path {label_path} does not exist"

    langs = []
    speakers = []
    texts = []
    paths = []
    for json_path in tqdm(label_path.glob("*.json")):
        with open(json_path, "r") as f:
            label = json.load(f)
            lang = label["typeInfo"]["language"]
            speaker = label["dialogs"]["speakerId"]
            text = label["dialogs"]["text"]
            flac_path = source_path / f"{json_path.stem}.flac"

            if not flac_path.exists():
                print(f"Path {flac_path} does not exist")
                continue

            langs.append(lang)
            speakers.append(speaker)
            texts.append(text)
            paths.append(str(flac_path.relative_to(source_path)))

    print(f"Languages: {set(langs)}")
    print(f"Speaker num: {len(set(speakers))}")

    df = pl.DataFrame(
        {
            "lang": langs,
            "speaker": speakers,
            "text": texts,
            "path": paths,
        },
        {
            "lang": pl.Categorical,
            "speaker": pl.Categorical,
            "text": pl.Utf8,
            "path": pl.Utf8,
        },
    )

    unique_speakers = df.get_column("speaker").unique().shuffle()
    num_speakers = len(unique_speakers)
    num_train_speakers = int(num_speakers * 0.9)
    num_test_speakers = num_speakers - num_train_speakers
    train_speakers = unique_speakers.head(num_train_speakers)
    test_speakers = unique_speakers.tail(num_test_speakers)

    train_df = df.filter(pl.col("speaker").is_in(train_speakers))
    test_df = df.filter(pl.col("speaker").is_in(test_speakers))

    train_df.write_csv(path / "train.csv")
    test_df.write_csv(path / "test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()

    path: Path = args.path
    assert is_bearable(
        path, Path
    ), f"Expected path to be of type Path, but got {type(path)}"
    assert path.exists(), f"Path {path} does not exist"
    main(path)
