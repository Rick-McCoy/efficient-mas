from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, get_worker_info

from data.dataset import AudioDataset


def worker_init_fn(worker_id: int, df_list: list[pl.DataFrame]):
    worker_info = get_worker_info()
    assert worker_info is not None
    dataset = worker_info.dataset
    assert isinstance(dataset, AudioDataset)
    dataset.rng = np.random.default_rng(worker_info.seed)
    df = df_list[worker_id]
    dataset.initialize(df)


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_paths: list[Path],
        batch_size: int,
        num_workers: int,
        sampling_rate: int,
        train_len: int,
        val_len: int,
        test_len: int,
        max_audio_len: int,
        max_phoneme_len: int,
        convert_audio_len: Callable[[int, bool], int],
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dataset_paths = [path.expanduser() for path in dataset_paths]
        self.sampling_rate = sampling_rate
        df = pl.concat(
            pl.read_csv(dataset_path / "train.csv", dtypes={"speaker": pl.Utf8}).select(
                speaker=pl.lit(f"{i}_") + pl.col("speaker")
            )
            for i, dataset_path in enumerate(self.dataset_paths)
        )
        speakers = (
            df.get_column("speaker")
            .unique()
            .sort()
            .sample(fraction=1, shuffle=True, seed=42)
        )
        train_speaker_size = int(len(speakers) * 0.9)
        val_speaker_size = len(speakers) - train_speaker_size
        self.train_speakers = speakers.head(train_speaker_size)
        self.val_speakers = speakers.tail(val_speaker_size)

        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len

        self.max_audio_len = max_audio_len
        self.max_phoneme_len = max_phoneme_len
        self.convert_audio_len = convert_audio_len

    def setup(self, stage: str | None = None):
        match stage:
            case "fit" | "validate":
                df_list = [
                    pl.read_csv(dataset_path / "train.csv", dtypes={"speaker": pl.Utf8})
                    for dataset_path in self.dataset_paths
                ]
                df_list = [
                    df.with_columns(speaker=f"{i}_" + pl.col("speaker"), index=i)
                    for i, df in enumerate(df_list)
                ]
                df = pl.concat(df_list, how="align")
                train_df = df.filter(pl.col("speaker").is_in(self.train_speakers))
                self.train_df_list = [
                    train_df.gather_every(self.num_workers, i)
                    for i in range(self.num_workers)
                ]
                val_df = df.filter(pl.col("speaker").is_in(self.val_speakers))
                self.val_df_list = [
                    val_df.gather_every(self.num_workers, i)
                    for i in range(self.num_workers)
                ]
                self.train_dataset = AudioDataset(
                    self.dataset_paths,
                    self.sampling_rate,
                    self.train_len,
                    self.max_audio_len,
                    self.max_phoneme_len,
                    self.convert_audio_len,
                )
                self.val_dataset = AudioDataset(
                    self.dataset_paths,
                    self.sampling_rate,
                    self.val_len,
                    self.max_audio_len,
                    self.max_phoneme_len,
                    self.convert_audio_len,
                )

                if self.num_workers == 0:
                    self.train_dataset.rng = np.random.default_rng(42)
                    self.train_dataset.initialize(train_df)
                    self.val_dataset.rng = np.random.default_rng(42)
                    self.val_dataset.initialize(val_df)

            case "test" | "predict":
                df_list = [
                    pl.read_csv(dataset_path / "test.csv", dtypes={"speaker": pl.Utf8})
                    for dataset_path in self.dataset_paths
                ]
                df_list = [
                    df.with_columns(speaker=f"{i}_" + pl.col("speaker"), index=i)
                    for i, df in enumerate(df_list)
                ]
                test_df = pl.concat(df_list, how="align")
                self.test_df_list = [
                    test_df.gather_every(self.num_workers, i)
                    for i in range(self.num_workers)
                ]
                self.test_dataset = AudioDataset(
                    self.dataset_paths,
                    self.sampling_rate,
                    self.test_len,
                    self.max_audio_len,
                    self.max_phoneme_len,
                    self.convert_audio_len,
                )

                if self.num_workers == 0:
                    self.test_dataset.rng = np.random.default_rng(42)
                    self.test_dataset.initialize(test_df)
            case _:
                raise ValueError(f"stage: {stage} is not supported")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
            worker_init_fn=partial(worker_init_fn, df_list=self.train_df_list),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
            worker_init_fn=partial(worker_init_fn, df_list=self.val_df_list),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
            worker_init_fn=partial(worker_init_fn, df_list=self.test_df_list),
            persistent_workers=True,
        )
