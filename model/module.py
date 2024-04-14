from collections.abc import Callable
from typing import Any

import torch
import wandb
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import Tensor
from vocos.feature_extractors import FeatureExtractor

from model.aligner import Aligner, mas_approx
from model.loss import AttentionCTCLoss, attention_bin_loss
from model.voco import MelVoco
from util.ipa import VOCAB_SIZE
from util.typing import (
    AlignMaskTensor,
    AlignProbTensor,
    AudioEncTensor,
    AudioTensor,
    LengthTensor,
    LossTensor,
    PhonemeTensor,
)
from util.utils import normalize_audio, plot_with_cmap


class AlignerModule(LightningModule):
    def __init__(
        self,
        dim_query: int,
        dim_key: int,
        feature_extractor: FeatureExtractor,
        convert_length: Callable[[int, bool], int],
        voco_type: str,
        optimizer: str = "Adam",
        lr: float = 1e-4,
        scheduler: str = "linear_warmup_decay",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "convert_length"])
        self.voco_type = voco_type
        self.feature_extractor = feature_extractor
        self.convert_length = convert_length

        self.aligner = Aligner(dim_query=dim_query, dim_key=dim_key, dim_hidden=dim_key)

        self.align_loss = AttentionCTCLoss()

        self.optim = optimizer
        self.lr = lr
        self.scheduler = scheduler

        self.example_input_array = (
            torch.randn(16, self.convert_length(2048, True)),
            torch.randint(1, VOCAB_SIZE, (16, 512)),
            torch.randint(1024, 2048, (16,)),
            torch.randint(256, 512, (16,)),
        )

        self.bin_count = 0
        self.anomaly_count = 0
        self.bin_sum = 0.0
        self.bin_sum_sq = 0.0
        self.bin_mean = 0.0
        self.bin_std = 0.0

    def encode_audio(self, audio: AudioTensor, audio_lens: LengthTensor):
        with torch.no_grad():
            if self.voco_type == "mel":
                kwargs = {}
            else:
                kwargs = {"bandwidth_id": 2}

            audio_enc = self.feature_extractor(audio, **kwargs)
            audio_enc = rearrange(audio_enc, "b d n -> b n d")

            if self.voco_type == "mel":
                audio_enc, *_ = normalize_audio(audio_enc, audio_lens)

        return audio_enc

    def get_alignment(
        self,
        phonemes: PhonemeTensor,
        phoneme_lens: LengthTensor,
        audio_enc: AudioEncTensor,
        audio_lens: LengthTensor,
    ) -> tuple[AlignProbTensor, AlignMaskTensor]:
        alignment_logprob = self.aligner(
            query=audio_enc,
            query_lens=audio_lens,
            key=phonemes,
            key_lens=phoneme_lens,
        )

        with torch.no_grad():
            alignment_mas = mas_approx(alignment_logprob, audio_lens, phoneme_lens)

        return alignment_logprob, alignment_mas

    def forward(
        self,
        audio: AudioTensor,
        phonemes: PhonemeTensor,
        audio_lens: LengthTensor,
        phoneme_lens: LengthTensor,
    ) -> tuple[LossTensor, LossTensor]:
        audio_enc = self.encode_audio(audio, audio_lens)

        alignment_logprob, alignment_mas = self.get_alignment(
            phonemes, phoneme_lens, audio_enc, audio_lens
        )

        align_loss = self.align_loss(alignment_logprob, phoneme_lens, audio_lens)
        bin_loss = attention_bin_loss(alignment_mas, alignment_logprob)

        if self.training:
            with torch.no_grad():
                self.bin_sum += bin_loss.sum().detach()
                self.bin_sum_sq += (bin_loss**2).sum().detach()
                self.bin_count += bin_loss.numel()

            if self.bin_std > 0.0:
                anomaly_mask = bin_loss <= self.bin_mean + 2 * self.bin_std
                self.anomaly_count += (~anomaly_mask).sum().detach()
                align_loss = align_loss[anomaly_mask]
                bin_loss = bin_loss[anomaly_mask]

        else:
            nan_mask = align_loss.isnan() | bin_loss.isnan()
            align_loss = align_loss[~nan_mask]
            bin_loss = bin_loss[~nan_mask]
            if nan_mask.any():
                print(audio_lens[nan_mask])
                print(phoneme_lens[nan_mask])

        if align_loss.nelement() == 0:
            align_loss = torch.tensor(0.0).to(self.device)
            bin_loss = torch.tensor(0.0).to(self.device)

        align_loss = align_loss.mean()
        bin_loss = bin_loss.mean()

        return align_loss, bin_loss

    @torch.compiler.disable()
    def log_metric(self, id: str, loss: Tensor | float, train: bool):
        if train:
            self.log(id, loss, on_step=True, on_epoch=False)
        else:
            self.log(id, loss, on_step=False, on_epoch=True, sync_dist=True)

    def single_step(self, batch: list[Tensor], prefix: str) -> LossTensor:
        audio, phonemes, audio_len, phoneme_len = batch
        align_loss, bin_loss = self.forward(audio, phonemes, audio_len, phoneme_len)
        train = prefix == "train"
        self.log_metric(f"{prefix}/align_loss", align_loss, train)
        self.log_metric(f"{prefix}/bin_loss", bin_loss, train)

        aligner_loss = align_loss + bin_loss
        self.log_metric(f"{prefix}/aligner_loss", aligner_loss, train)

        return aligner_loss

    def training_step(self, batch: list[Tensor], batch_idx: int):
        return self.single_step(batch, "train")

    def validation_step(self, batch: list[Tensor], batch_idx: int):
        self.single_step(batch, "val")
        if batch_idx == 0 and self.global_rank == 0:
            self.log_table(batch, "val")

    def test_step(self, batch: list[Tensor], batch_idx: int):
        self.single_step(batch, "test")
        if batch_idx == 0 and self.global_rank == 0:
            self.log_table(batch, "test")

    @torch.compiler.disable()
    def log_table(self, batch: list[Tensor], prefix: str):
        audio, phonemes, audio_lens, phoneme_lens = batch
        random_audio_index = torch.randint(0, audio.shape[0], (1,))
        random_audio_len = audio_lens[[random_audio_index]]
        max_audio_len = int(random_audio_len.max().item())
        conv_max_audio_len = self.convert_length(max_audio_len, True)
        random_phoneme_len = phoneme_lens[[random_audio_index]]
        random_audio = audio[[random_audio_index], :conv_max_audio_len]
        random_phonemes = phonemes[[random_audio_index], :random_phoneme_len]

        with torch.no_grad():
            random_audio_enc = self.encode_audio(random_audio, random_audio_len)

            alignment_logprob, alignment_mas = self.get_alignment(
                random_phonemes, random_phoneme_len, random_audio_enc, random_audio_len
            )
            alignment = alignment_logprob.exp().detach().cpu().numpy()
            alignment_path = alignment_mas.detach().cpu().numpy()
            if self.voco_type == "mel":
                mel = random_audio_enc.detach().cpu().numpy()
            else:
                mel_voco = MelVoco().to(self.device)
                mel = mel_voco.encode(random_audio).detach().cpu().numpy()

            alignment_plot = plot_with_cmap(
                [alignment[0].T, alignment_path[0].T, mel[0].T]
            )

        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                f"{prefix}/alignment", [wandb.Image(alignment_plot, mode="RGBA")]
            )
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f"{prefix}/alignment",
                alignment_plot,
                self.global_step,
                dataformats="HWC",
            )
        elif isinstance(self.logger, MLFlowLogger):
            epoch = self.current_epoch
            self.logger.experiment.log_image(
                self.logger.run_id,
                alignment_plot,
                f"{prefix}/{epoch:03d}_alignment.png",
            )

    def configure_optimizers(self):
        match self.optim:
            case "Adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            case "AdamW":
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=self.lr, weight_decay=1e-2
                )
            case _:
                raise ValueError(f"Unknown optimizer: {self.optim}")

        match self.scheduler:
            case "None":
                return optimizer
            case "linear_warmup_decay":
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1 / 5000, end_factor=1.0, total_iters=5000
                )
                decay_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=0.0, total_iters=1000000
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, decay_scheduler],
                    milestones=[5000],
                )
            case _:
                raise ValueError(f"Unknown scheduler: {self.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        self.bin_mean = self.bin_sum / self.bin_count
        self.bin_std = (self.bin_sum_sq / self.bin_count - self.bin_mean**2) ** 0.5
        anomaly_ratio = self.anomaly_count / self.bin_count
        self.log_metric("train/anomaly_ratio", anomaly_ratio, False)
        self.bin_sum = 0.0
        self.bin_sum_sq = 0.0
        self.bin_count = 0
        self.anomaly_count = 0

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict: dict[str, Any] = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if key.startswith("feature_extractor"):
                del state_dict[key]
