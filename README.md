# voice-box

## Installation

Python 3.11 required.
```bash
pip install -r requirements.txt
sudo apt install espeak-ng
```

## Structure

Open collapsed lists for information regarding each folder.
- <details>
    <summary>config</summary>

    Contains dataclasses for [hydra](https://hydra.cc) configuration.
    - <details>
        <summary>data</summary>

        - config.py
        </details>
    - <details>
        <summary>model</summary>

        - config.py
        </details>
    - <details>
        <summary>train</summary>

        - config.py
        </details>
    - config.py
</details>

- <details>
    <summary>data</summary>

    Contains data handling code.
    See [Data](#data) section for data & metadata formats.

    - datamodule.py
        - Contains the code for custom [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). Supports combining multiple data sources.
    - dataset.py
        - Reads audio file & resamples.

</details>

- <details>
    <summary>encodec</summary>

    Original code significantly modified to work with type checking & `torch.compile`.

    Inference only; no fine-tuning.

    Please don't touch this. Report all problems to [this guy](mailto:juneyoung.yi@ngine.co.kr).
</details>

- <details>
    <summary>model</summary>

    The real meat goes here.

    - aligner.py
        - MAS aligner. Component not present in original paper.
    - duration_predictor.py
        - Duration predictor using flow matching.
    - lightning_module.py
        - Wraps all components into a LightningModule. Contains training, validation, sampling implementations.
    - loss.py
        - Masked MSE loss for flow matching.
        - Binarization loss and CTC loss for MAS alignment.
    - modules.py
        - Wrappers around `nn.Module` classes for [jaxtyping](https://github.com/google/jaxtyping) type checking.
        - GGELU & Positional Encoding. Both components not present in oroginal paper.
    - norm.py
        - RMSNorm & Adaptive RMSNorm. Component not present in original paper.
    - rotary_embedding.py
        - Rotary Embedding & helper functions. Component not present in original paper.
    - transformer.py
        - Custom Transformer for Rotary Embedding integration.
    - voco.py
        - Wrapper around [Vocos](https://github.com/gemelo-ai/vocos) vocoder. Component not present in original paper.
    - voicebox.py
        - Voicebox using flow matching.
</details>

- <details>
    <summary>torchode</summary>

    Original code somewhat modified to work with type checking & `torch.compile`.
</details>

- <details>
    <summary>utils</summary>

    Helper functions and text processing.

    - <details>
        <summary>expand</summary>

        English, Spanish, and French text sanitization.
        Currently unused; refactoring necessary.

        Don't even try to run these, these are untested and their required libraries aren't in `requirements.txt`.
        </details>

    - <details>
        <summary>text</summary>
        
        Korean text sanitization.
        Converts non-Korean segments into their Korean counterparts.

        - korean_dict.py
            - Mappings from simple acronyms, special characters, units, and numbers into Korean pronunciations.
        - korean.py
            - Functions for Korean sanitization.
        </details>

    - cleaner.py
        - Text processor for sanitizing text. Designed to support multiple languages, currently only supports Korean.
    - ipa.py
        - A list of all IPA characters used in the output of [phonemizer](https://github.com/bootphon/phonemizer). Plus some fixes for bugs within the library.
        - Future-proofed for who knows when. When will we ever support Icelandic?
    - mask.py
        - Helper functions for masking.
    - prior.py
        - Log Beta-binomial distribution implementation in PyTorch. Praise be the PyTorch team for the `torch.lgamma` function.
        - Used in MAS alignment.
    - tokenizer.py
        - Converts sanitized text into integer tokens.
        - Uses [phonemizer](https://github.com/bootphon/phonemizer) for conversion into IPA symbols.
        - See [Data](#data) for details.
    - typing.py
        - All possible types & shapes of Tensors for [jaxtyping](https://github.com/google/jaxtyping) type checking.
    - utils.py
        - Miscellaneous functions, such as turning mel spectrograms into images.

</details>

- synthesize.py
    - See [Synthesizing](#synthesizing) for details.
- train.py
    - See [Training](#training) for details.


## Training

All configurations are handled by [Hydra](https://hydra.cc).

### Configuration

There are 4 configurable components; train, data, voicebox, and duration.

- <details>
    <summary>train</summary>

    - acc
        - Default `1`
        - [Gradient accumulation](https://lightning.ai/blog/gradient-accumulation/): Increase your batch size with minimal overhead!
    - batch_size
        - Default `4`
    - ckpt_path
        - Default `None`
        - Resume training from this checkpoint path. Set to `None` to train from scratch.
    - early_stop
        - Default `True`
        - Monitor validation loss and stop when it starts to overfit.
    - fast_dev_run
        - Default `False`
        - Run 1 training, validation, and test step for testing purposes.
        - Also enables anomaly detection for NaN checking.
    - lr
        - Default `1e-4`
        - Learning rate
    - num_workers
        - Default `8`
        - Number of Dataloader workers per GPU.
    - optimizer
        - Default `Adam`
        - Supporting `Adam` and `AdamW`. Weight decay fixed at `1e-2` for `AdamW`.
    - scheduler
        - Default `linear_warmup_decay`
        - LR scheduler. Supporting `linear_warmup_decay` for the original schedule specified in the paper.
        - Set to `None` for constant LR.
    - gradient_clip_val
        - Default `1.0`
        - Clip gradient by norm. Set to `0` to disable clipping.
    - precision
        - Default `16-mixed`
        - Gradient precision via autocast; Supporting `32`, `16-mixed`, and `bf16-mixed`.
        - `32` is the default 32-bit precision
        - `16-mixed` is 16-bit mixed precision, where 32-bit is used in specific operations such as batch normalization.
        - `bf16-mixed` is [bfloat](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) 16-bit mixed precision. Untested.
    - project
        - Default `voicebox`
        - Used for W&B logging project. Ignore.
    - wandb
        - Default `True`
        - Use W&B logging. Set to `False` to use local Tensorboard logging for testing purposes.
    - weight_average
        - Defualt `False`
        - Keep a moving exponential average of weights per epoch.
        - Originally from VALL-E; Untested.
</details>

- <details>
    <summary>data</summary>

    - paths
        - Default `[]`
        - A list of dataset paths. See [Data](#data) for dataset structure.
    - sampling_rate
        - Default `24000`
        - Audio sampling rate.
    - sample_sentence
        - Default `나는 고양이로소이다. 이름은 아직 없다.`
        - Sample sentence for validation. A sentence not present within training & validation data.

</details>

- <details>
    <summary>voicebox</summary>
    
    All default values from original paper.

    - dim
        - Default `1024`
        - Size of hidden dimension.
    - depth
        - Default `24`
        - Number of transformer layers.
    - heads
        - Default `16`
        - Number of heads in multi-head attention.
        - As of 2024-01-18, `dim // heads` must be at most 64 (or 96 depending on GPU architecture) for fast attention.
    - attn_dropout
        - Default `0.0`
        - Dropout value for multi-head attention.
    - ff_dropout
        - Default `0.1`
        - Dropout value for linear layer.
    - kernel_size
        - Default `31`
        - Kernel size for 1D convolution in convolutional position embedding.
    - voco
        - Default `mel`
        - Vocoder type for audio. Supports `mel` and `encodec`.
        - Note that selecting either of them will not change the visualization of mel spectrograms in validation & testing.
    - max_audio_len
        - Defualt `1024`
        - Maximum audio frame length; randomly slices longer audio. Slicing only occurs when training.
        - Using default settings, 1024 mel frames corresponds to ~`10.9` seconds.
</details>


- <details>
    <summary>duration</summary>
    
    All default values from original paper.
    Since we use flow matching for duration, it's mostly similar to voicebox.

    - dim
        - Default `512`
        - Size of hidden dimension.
    - depth
        - Default `10`
        - Number of transformer layers.
    - heads
        - Default `8`
        - Number of heads in multi-head attention.
        - As of 2024-01-18, `dim // heads` must be at most 64 (or 96 depending on GPU architecture) for fast attention.
    - attn_dropout
        - Default `0.1`
        - Dropout value for multi-head attention. The default value being different to `voicebox` is intentional.
    - ff_dropout
        - Default `0.1`
        - Dropout value for linear layer.
    - kernel_size
        - Default `15`
        - Kernel size for 1D convolution in convolutional position embedding.
    - max_phoneme_len
        - Defualt `256`
        - Maximum phoneme length; randomly slices longer text. Slicing only occurs when training.
        - Using default settings, 256 phonemes corresponds to roughly ~`43` English words and ~`25` Korean words.
</details>

All configurations can be altered via arguments as such:

```bash
python train.py data.paths=["~/dataset/librispeech","~/dataset/libritts"] train.fast_dev_run=True train.acc=4 voicebox.voco=encodec duration.attn_dropout=0
```

#### Presets

Since the size of the voicebox & duration model varies between testing and training purposes, certain presets have been defined.

- Voicebox
    - large: `dim=1024, depth=24, heads=16`
    - medium: `dim=512, depth=12, heads=8`
    - small: `dim=256, depth=6, heads=4`
- Duration
    - large: `dim=512, depth=10, heads=8`
    - medium: `dim=256, depth=4, heads=4`
    - small: `dim=128, depth=2, heads=2`

These presets can be set as such:

```bash
python train.py ... voicebox=medium duration=small
```

## Synthesizing

Synthesis is implemented at `synthesize.py` in the root directory.

Unlike `train.py` with its `hydra` arguments, `synthesize.py` uses `argparse` for simplicity's sake.

Currently, only single sentences are supported.

Specify `ckpt_path`, `original_audio`, `original_script`, and `target_script`.

To change output directory (default `./outputs/`) and resulting file name (default `./outputs/output.wav`), specify `output_dir` and `name` (without the `.wav` suffix) respectively.

```bash
python synthesize.py --ckpt_path ./checkpoints/voicebox.ckpt --original_audio original.wav --original_script "캐릭터 성장을 빠르게 할 수 있도록 난이도를 하향하겠습니다." --target_script "나는 고양이로소이다. 이름은 아직 없다."
```

## Data

Training requires audio files and their respective transcriptions & speaker names.

The dataset structure is simple: audio files, a `train.csv` file, and a `val.csv` file.

Each CSV file needs to contain 3 columns: `wav_path`, `script`, and `speaker`. All other columns are ignored.

`wav_path` specifies the path of the WAV file relative to the top directory. For example, if the audio file is located at `~/dataset/librispeech/train/1.wav`, and the dataset top directory is `~/dataset/librispeech`, then `wav_path` must be `train/1.wav`.

`script` is the text transcription. `speaker` must be the unique speaker ID.

Note that while `speaker` isn't strictly necessary, testing for unseen speaker generalization required speaker separation between training, validation, and testing.

Unfortunately, due to legacy issues, the CSV files are confusingly named. All files listed in `train.csv` will be used in training and validation, while all files listed in `val.csv` will be used in testing. Training and validation data is split automatically with a roughly 9-1 ratio.

### Text processing

As detailed in the [Structure](#structure) section, we sanitize and phonemize the given text using custom cleaners and a phonemizer. Both processes are language-specific.

The cleaner converts foreign loanwords, expand units, and numerical values into their pronunciations. These do not have a unified library, and need to be added on a per language basis.

We convert the sanitized text into phonemes using the [phonemizer](https://github.com/bootphon/phonemizer) library with the [espeak-ng](https://github.com/espeak-ng/espeak-ng) backend. All possible IPA characters are part of the token set in order to maximize compatibility with unforeseen inputs.

## Model

The model follows the basic structures of the model detailed in the original paper. There are additional modifications made by lucidrain and myself. We removed many modifications that lucidrain made that were not suitable for our purposes.

lucidrain added Rotary Embedding, RMS Norm, Encodec, and Vocos vocoder.

We added MAS alignment for a more complete end-to-end model.

## Contributing

Up to this point, contribution has been a one-man effort, and so the main branch was used without impunity.

The following process will be implemented shortly, but as of 2024-01-18 is not realized.

A `live` and a `dev` branch will be separated. The `live` branch will only accept merge commits from the `dev` branch.
Each new feature should be implemented via a merge request branched off `dev`, then merged when a code review is done.

## Deployment

Currently not implemented. Future implementation of a [TorchServe](https://pytorch.org/serve) handler will enable deployment.

## Documentation

You're looking at it!

## Troubleshooting

- 

## Contact

The poor dude: Juneyoung.yi@ngine.co.kr

## Acknowledgements

- [lucidrains](https://github.com/lucidrains), for their [original implementation](https://github.com/lucidrains/voicebox-pytorch) of VoiceBox.

## TODO
- [ ] Finish this README
- [ ] Implement separate aligner training
- [ ] Implement TorchServe handler
