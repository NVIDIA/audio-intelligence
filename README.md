# ETTA: Elucidating the Design Space of Text-to-Audio Models

#### **Sang-gil Lee***, **Zhifeng Kong***, Arushi Goel, Sungwon Kim, Rafael Valle, Bryan Catanzaro (*Equal contribution.)

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2412.19351"><img src="https://img.shields.io/badge/arXiv-2412.19351-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/ETTA/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/elucidated-text-to-audio/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/elucidated-text-to-audio.svg?style=social"></a>
</div>
<!-- [[Paper]](https://arxiv.org/abs/2412.19351) - [[Code]](https://github.com/NVIDIA/elucidated-text-to-audio) - [[Project Page]](https://research.nvidia.com/labs/adlr/ETTA/) -->


## Overview

This repository contains model, training, and inference code implementation of [ETTA: Elucidating the Design Space of Text-to-Audio Models](https://arxiv.org/abs/2412.19351) (ICML 2025):

* Synthetic audio caption generation pipeline is built on top of [Audio Flamingo](https://github.com/NVIDIA/audio-flamingo) from NVIDIA. See ```AFSynthetic/README.md``` for more details.

* Text-to-audio model is built on top of [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) from Stability AI.

## Installation

The codebase has been tested on Python `3.11` and PyTorch `>=2.6.0`. Below is an example command to create a conda environment:

```shell
# create and activate a conda environment
conda create -n etta python=3.11
conda activate etta
# install pytorch
pip install torch torchvision torchaudio
# install flash-attn
pip install flash-attn --no-build-isolation
# install this repository in editable mode
pip install -e .
```

Alternatively, we also provide an example `Dockerfile` based on an official PyTorch image to run the model on a Docker container.
```shell
docker build --progress plain -t etta:latest .
docker run --gpus all etta:latest
```

## Inference Examples

Below is an example ETTA inference script on a single GPU:
```
CUDA_VISIBLE_DEVICES=0 python inference_tta.py \
--text_prompt "A hip-hop track using sounds from a construction site—hammering nails as the beat, drilling sounds as scratches, and metal clanks as rhythm accents." "A saxophone that sounds like meowing of cat." \
--output_dir ./tmp \
--model_ckpt_path /path/to/your/etta/model_unwrap.ckpt \
--target_sample_rate 44100 \
--sampler_type euler \
--steps 100 \
--cfg_scale 3.5 \
--seconds_start 0 \
--seconds_total 10 \
--batch_size 4
```


## Training Examples

Below is an example command to train ETTA-VAE on 8 GPUs:
```
NUM_GPUS=8 && \
torchrun --nproc_per_node=$NUM_GPUS train.py \
--name DEBUG_etta_vae \
--dataset_config stable_audio_tools/configs/dataset_configs/etta_vae_training_example.json \
--model_config stable_audio_tools/configs/model_configs/autoencoders/etta_vae.json \
--save_dir tmp --ckpt_path last \
--enable_progress_bar true \
--seed 2025 \
--num_gpus $NUM_GPUS \
--batch_size 8 \
--params \
training.max_steps=2800000 \
training.loss_configs.bottleneck.weights.kl=0.0001
```

Below is an example command to train ETTA-DiT on 8 GPUs:
```
NUM_GPUS=8 && \
torchrun --nproc_per_node=$NUM_GPUS train.py \
--name DEBUG_etta_dit \
--dataset_config stable_audio_tools/configs/dataset_configs/etta_dit_training_example.json \
--model_config stable_audio_tools/configs/model_configs/txt2audio/etta_dit.json \
--save_dir tmp --ckpt_path last \
--enable_progress_bar true \
--seed 2025 \
--num_gpus $NUM_GPUS \
--batch_size 8 \
--params \
pretransform_ckpt_path=/path/to/etta_vae/model_unwrap_step_2800000.ckpt \
model.diffusion.config.depth=24 \
training.max_steps=1000000
```

Below is an example command to unwrap a trained model into `model_unwrap.ckpt`:
```
CKPT_DIR=/path/to/your/etta &&
python unwrap_model.py \
--model-config $CKPT_DIR/config.json \
--ckpt-path $CKPT_DIR/epoch=x-step=xxxxxx.ckpt \
--name $CKPT_DIR/model_unwrap
```


## Citation
```bibtex
@article{lee2024etta,
  title={ETTA: Elucidating the Design Space of Text-to-Audio Models},
  author={Lee, Sang-gil and Kong, Zhifeng and Goel, Arushi and Kim, Sungwon and Valle, Rafael and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2412.19351},
  year={2024}
}
```

## Reference
For more detail in `stable-audio-tools` from which we build ETTA upon, see [original README.md of stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools/blob/main/README.md).
