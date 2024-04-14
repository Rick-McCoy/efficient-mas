import logging
from collections.abc import Callable
from pathlib import Path

import librosa
import numpy as np
import polars as pl
import soundfile as sf
import torch
from hydra.core.hydra_config import HydraConfig
from soundfile import LibsndfileError
from torch.nn import functional as F
from torch.utils.data import Dataset, get_worker_info
from torchaudio import transforms as T

from util.ipa import VOCAB_TO_ID
from util.tokenizer import Tokenizer


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list[Path],
        sampling_rate: int,
        length: int,
        max_audio_len: int,
        max_phoneme_len: int,
        convert_audio_len: Callable[[int, bool], int],
    ):
        super().__init__()
        self.logger = logging.getLogger("dataset")
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        try:
            output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            output_dir = "outputs"
        handler = logging.FileHandler(f"{output_dir}/dataset.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.dataset_paths = dataset_paths
        self.tokenizer = Tokenizer()

        self.sampling_rate = sampling_rate
        self.resamplers: dict[int, T.Resample] = {}
        self.rng: np.random.Generator
        self.length = length
        self.max_audio_len = convert_audio_len(max_audio_len, True)
        self.max_phoneme_len = max_phoneme_len
        self.convert_audio_len = convert_audio_len

    def initialize(self, df: pl.DataFrame):
        self.paths = df.get_column("path")
        self.texts = df.get_column("text")
        self.indices = df.get_column("index")

        lang_count = df.get_column("lang").value_counts()
        total_count = lang_count.get_column("count").sum()
        lang_count = lang_count.with_columns(prob=pl.col("count").cast(pl.Float64))
        lang_count = lang_count.with_columns((pl.col("prob") / total_count).log() / 4)
        prob_sum = lang_count["prob"].exp().sum()
        lang_count = lang_count.with_columns(pl.col("prob").exp() / prob_sum)
        self.probs = lang_count.get_column("prob")
        self.langs = lang_count.get_column("lang")

        lang_idx: dict[str, list[int]] = {lang: [] for lang in self.langs}
        for i, lang in enumerate(df.get_column("lang")):
            lang_idx[lang].append(i)

        self.lang_idx: dict[str, np.ndarray] = {
            lang: np.array(lang_idx[lang]) for lang in lang_idx
        }

    def __getitem__(self, _index: int):
        while True:
            try:
                lang: str = self.rng.choice(self.langs, p=self.probs)
                index: int = self.rng.choice(self.lang_idx[lang])
                dataset_index: int = self.indices.item(index)
                dataset_path = self.dataset_paths[dataset_index]
                path: Path = dataset_path / "source" / self.paths.item(index)
                audio, sr = sf.read(path, dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                audio, _ = librosa.effects.trim(audio, top_db=30)
                audio = torch.from_numpy(audio)

                if sr != self.sampling_rate:
                    if sr not in self.resamplers:
                        self.resamplers[sr] = T.Resample(sr, self.sampling_rate)

                    audio = self.resamplers[sr](audio)

                if len(audio) > self.max_audio_len:
                    raise ValueError(
                        f"Audio too long: {len(audio) / self.sampling_rate}s"
                    )

                if len(audio) < 0.1 * self.sampling_rate:
                    raise ValueError(
                        f"Audio too short: {len(audio) / self.sampling_rate}s"
                    )

                audio_len = self.convert_audio_len(len(audio), False)
                audio = F.pad(audio, (0, self.max_audio_len - len(audio)))

                text = self.texts.item(index)
                phonemes = self.tokenizer.phonemize([text], language=lang)[0]

                if ("lˌɛɾɚ" in phonemes) or ("lˈe̞tə" in phonemes):
                    raise ValueError(f"Phonemization failed:\n{phonemes}\n{text}")

                ids = torch.tensor(self.tokenizer.phonemes_to_ids([phonemes])[0])
                id_len = len(ids)

                if id_len < 4:
                    raise ValueError(f"Phonemes too short: {phonemes}")

                if id_len > self.max_phoneme_len:
                    raise ValueError(f"Phonemes too long: {id_len}\n{phonemes}\n{text}")

                ids = F.pad(
                    ids,
                    (0, self.max_phoneme_len - id_len),
                    value=VOCAB_TO_ID["<pad>"],
                )

                if audio_len <= id_len:
                    raise ValueError(
                        f"Audio should be longer than text: {audio_len} <= {id_len}"
                    )

                break

            except (ValueError, LibsndfileError) as e:
                worker_info = get_worker_info()
                assert worker_info is not None
                worker_id = worker_info.id

                home_dir = Path.home()
                relative_path = path.relative_to(home_dir)
                self.logger.debug(e)
                self.logger.debug(f"ID: {worker_id}, Path: {relative_path}")
                continue

        return audio, ids, audio_len, id_len

    def __len__(self):
        return self.length
