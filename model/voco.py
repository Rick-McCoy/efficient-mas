from pathlib import Path

import torch
from einops import rearrange
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import hf_hub_download, repo_folder_name
from torch import Tensor, nn
from vocos import Vocos

from util.typing import AudioEncTensor, AudioTensor


def load_voco(repo_id: str):
    cache_dir = Path(HF_HUB_CACHE) / repo_folder_name(
        repo_id=repo_id, repo_type="model"
    )
    local_files_only = cache_dir.exists()
    config_path = hf_hub_download(
        repo_id=repo_id, filename="config.yaml", local_files_only=local_files_only
    )
    model_path = hf_hub_download(
        repo_id=repo_id, filename="pytorch_model.bin", local_files_only=local_files_only
    )
    model = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class Voco(nn.Module):
    vocos: Vocos

    @property
    def latent_dim(self) -> int:
        raise NotImplementedError()

    @property
    def compression_factor(self) -> int:
        raise NotImplementedError()

    @property
    def sampling_rate(self) -> int:
        raise NotImplementedError()

    def encode(self, audio: AudioTensor, **kwargs) -> AudioEncTensor:
        mel = self.vocos.feature_extractor(audio, **kwargs)
        mel = rearrange(mel, "b d n -> b n d")
        return mel

    def decode(self, latents: AudioEncTensor) -> AudioTensor:
        raise NotImplementedError()

    def convert_length(self, length: int, reverse: bool = False) -> int:
        raise NotImplementedError()


class MelVoco(Voco):
    def __init__(self, repo_id: str = "charactr/vocos-mel-24khz"):
        super().__init__()
        self.vocos = load_voco(repo_id)

    @property
    def latent_dim(self) -> int:
        return self.vocos.feature_extractor.mel_spec.n_mels

    @property
    def compression_factor(self) -> int:
        return self.vocos.feature_extractor.mel_spec.hop_length

    @property
    def sampling_rate(self) -> int:
        return self.vocos.feature_extractor.mel_spec.sample_rate

    def decode(self, mel: AudioEncTensor) -> AudioTensor:
        mel = rearrange(mel.float(), "b n d -> b d n")

        return self.vocos.decode(mel)

    def convert_length(self, lengths: int, reverse: bool = False) -> int:
        if reverse:
            return lengths * self.compression_factor - 1

        return lengths // self.compression_factor + 1


class EncodecVoco(Voco):
    def __init__(self, repo_id="charactr/vocos-encodec-24khz", bandwidth_id=2):
        super().__init__()
        self.vocos = load_voco(repo_id)
        self.register_buffer("bandwidth_id", torch.tensor([bandwidth_id]))
        self.bandwidth_id: Tensor
        self.vocos.feature_extractor.encodec.set_target_bandwidth(
            self.vocos.feature_extractor.bandwidths[bandwidth_id]
        )

    @property
    def latent_dim(self) -> int:
        return self.vocos.feature_extractor.encodec.encoder.dimension

    @property
    def compression_factor(self) -> int:
        bandwidth = int(self.vocos.feature_extractor.encodec.bandwidth * 1000)
        num_quantizers = self.vocos.feature_extractor.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.vocos.feature_extractor.encodec.frame_rate,
            self.vocos.feature_extractor.encodec.bandwidth,
        )
        bits_per_codebook = self.vocos.feature_extractor.encodec.bits_per_codebook
        codec_rate = bandwidth // num_quantizers // bits_per_codebook
        return self.sampling_rate // codec_rate

    @property
    def sampling_rate(self) -> int:
        return self.vocos.feature_extractor.encodec.sample_rate

    def encode(self, audio: AudioTensor) -> AudioEncTensor:
        return super().encode(audio, bandwidth_id=self.bandwidth_id)

    def decode(self, latents: AudioEncTensor) -> AudioTensor:
        latents = rearrange(latents, "b n d -> b d n")
        codes = self.vocos.feature_extractor.encodec.quantizer.encode(
            latents,
            self.vocos.feature_extractor.encodec.frame_rate,
            self.vocos.feature_extractor.encodec.bandwidth,
        )
        codes = rearrange(codes, "q b n -> b q n")

        all_audios = []
        for code in codes:
            features = self.vocos.codes_to_features(code)
            audio = self.vocos.decode(features, bandwidth_id=self.bandwidth_id)
            audio = rearrange(
                audio, f"{self.vocos.feature_extractor.encodec.channels} t -> t"
            )
            all_audios.append(audio)

        return torch.stack(all_audios)

    def convert_length(self, lengths: int, reverse: bool = False) -> int:
        if reverse:
            return lengths * self.compression_factor

        return (lengths - 1) // self.compression_factor + 1
