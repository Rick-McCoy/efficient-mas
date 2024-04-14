import time
from pathlib import Path
from typing import cast

import hydra
import mlflow
import torch
import torch.distributed
from beartype.door import is_bearable
from hydra.core.config_store import ConfigStore
from jaxtyping import install_import_hook
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from config.config import Config, dict_to_config
from config.data.config import DataConfig
from config.model.config import (
    AlignerLargeConfig,
    AlignerMediumConfig,
    AlignerSmallConfig,
)
from config.train.config import TrainConfig
from data.datamodule import AudioDataModule
from model.module import AlignerModule

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base", node=TrainConfig)
cs.store(group="data", name="base", node=DataConfig)
cs.store(group="aligner", name="small", node=AlignerSmallConfig)
cs.store(group="aligner", name="medium", node=AlignerMediumConfig)
cs.store(group="aligner", name="large", node=AlignerLargeConfig)


@hydra.main(config_name="config", version_base=None)
def main(_cfg: DictConfig):
    cfg = dict_to_config(_cfg)

    if cfg.train.compile:
        from model.voco import EncodecVoco, MelVoco
    else:
        with install_import_hook(
            ["data", "model", "utils", "module", "encodec"],
            typechecker="beartype.beartype",
        ):
            from model.voco import EncodecVoco, MelVoco

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    match cfg.voco:
        case "mel":
            voco = MelVoco()
        case "encodec":
            voco = EncodecVoco()
        case _:
            raise ValueError(f"Unsupported voco: {cfg.voco}")

    voco.eval()
    for param in voco.parameters():
        param.requires_grad_(False)

    sampling_rate = voco.sampling_rate
    convert_length = voco.convert_length

    paths = OmegaConf.to_object(cfg.data.paths)
    assert isinstance(paths, list)
    assert is_bearable(paths, list[Path])

    total_batch_size = cfg.train.batch_size * torch.cuda.device_count() * cfg.train.acc

    datamodule = AudioDataModule(
        dataset_paths=paths,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sampling_rate=sampling_rate,
        train_len=cfg.train.train_len * total_batch_size,
        val_len=cfg.train.val_len * total_batch_size,
        test_len=cfg.train.test_len * total_batch_size,
        max_audio_len=cfg.aligner.max_audio_len,
        max_phoneme_len=cfg.aligner.max_phoneme_len,
        convert_audio_len=convert_length,
    )

    model = AlignerModule(
        dim_query=voco.latent_dim,
        dim_key=cfg.aligner.dim,
        feature_extractor=voco.vocos.feature_extractor,
        convert_length=convert_length,
        voco_type=cfg.voco,
        optimizer=cfg.train.optimizer,
        lr=cfg.train.lr,
        scheduler=cfg.train.scheduler,
    )

    Path("logs").mkdir(exist_ok=True)
    if cfg.train.fast_dev_run:
        logger = None
        run_name = None
    elif cfg.train.mlflow:
        logger = MLFlowLogger(
            experiment_name=cfg.train.project,
            save_dir="./logs",
            tracking_uri="file:./logs",
        )
        run_id = logger.run_id
        if run_id is None:
            run_name = None
        else:
            run = mlflow.get_run(run_id)
            run_name = run.info.run_name
            run = mlflow.start_run(run_id=run_id, log_system_metrics=True)
    else:
        logger = TensorBoardLogger(save_dir="logs", name="voicebox")
        run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    if run_name is not None:
        checkpoint_dir = Path("checkpoints") / str(run_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = None

    # seed_everything(42, workers=True)

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "aligner.txt", "w") as f:
            f.write(str(model))

    if cfg.train.compile:
        model = cast(LightningModule, torch.compile(model))

    monitor_loss = "val/aligner_loss"
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{{epoch:03d}}-aligner-loss={{{monitor_loss}:.4f}}",
        monitor=monitor_loss,
        save_top_k=3,
        mode="min",
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [model_checkpoint]

    if cfg.train.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_loss,
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
                strict=False,
                check_finite=False,
            )
        )

    if cfg.train.scheduler != "None":
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if cfg.train.weight_average:

        def avg_fn(
            averaged_model_parameter: Tensor,
            model_parameter: Tensor,
            _num_averaged: Tensor,
        ) -> Tensor:
            return averaged_model_parameter * 0.99 + model_parameter * 0.01

        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=cfg.train.lr / 10,
                swa_epoch_start=0.5,
                annealing_epochs=10,
                annealing_strategy="cos",
                avg_fn=avg_fn,
            )
        )

    precision = cfg.train.precision
    assert precision == "16-mixed" or precision == "32" or precision == "bf16-mixed"

    trainer = Trainer(
        strategy="ddp",
        accumulate_grad_batches=cfg.train.acc,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks,
        detect_anomaly=cfg.train.fast_dev_run,
        fast_dev_run=cfg.train.fast_dev_run,
        logger=logger,
        log_every_n_steps=10,
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=10,
        precision=precision,
        enable_model_summary=False,
    )

    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.train.resume_aligner_path
    )

    if cfg.train.fast_dev_run:
        best_model = model
    else:
        best_path = model_checkpoint.best_model_path

        best_model = AlignerModule.load_from_checkpoint(
            best_path,
            feature_extractor=voco.vocos.feature_extractor,
            convert_length=convert_length,
            strict=False,
        )

    trainer.test(model=best_model, datamodule=datamodule)

    if checkpoint_dir is None:
        save_path = "aligner.ckpt"
    else:
        save_path = checkpoint_dir / "aligner.ckpt"

    if not cfg.train.fast_dev_run:
        trainer.save_checkpoint(save_path, weights_only=True)

    if cfg.train.mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
