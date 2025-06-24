# ETTA: Elucidating the Design Space of Text-to-Audio Models

#### Sang-gil Lee*, Zhifeng Kong*, Arushi Goel, Sungwon Kim, Rafael Valle, Bryan Catanzaro

(*Equal contribution.)

A customized stable-audio-tools.

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

## Quickstart 

ETTA inference on a single GPU
```
CUDA_VISIBLE_DEVICES=0 python inference_tta.py \
--text_dir /path/to/debug_stable_audio_captions \
--output_dir /path/to/debug_stable_audio_samples \
--model_ckpt_path /path/to/etta_dit/model_unwrap_step_1000000.ckpt \
--sampler_type euler \
--steps 100 \
--cfg_scale 3.5 \
--seconds_start 0 \
--seconds_total 10
```

train VAE on a single GPU
```
CUDA_VISIBLE_DEVICES=0 python train.py \
--num_gpus 1 \
--name DEBUG_etta_vae \
--dataset_config stable_audio_tools/configs/dataset_configs/etta_vae_training_example.json \
--model_config stable_audio_tools/configs/model_configs/autoencoders/etta_vae.json \
--save_dir tmp --ckpt_path last \
--enable_progress_bar true \
--params \
batch_size=8 \
training.loss_configs.bottleneck.weights.kl=0.0001
```

train ETTA on a single GPU
```
CUDA_VISIBLE_DEVICES=0 python train.py \
--num_gpus 1 \
--name DEBUG_etta_dit \
--dataset_config stable_audio_tools/configs/dataset_configs/etta_dit_training_example.json \
--model_config stable_audio_tools/configs/model_configs/txt2audio/etta_dit.json \
--save_dir tmp --ckpt_path last \
--enable_progress_bar true \
--params \
pretransform_ckpt_path=/path/to/etta_vae/model_unwrap_step_2800000.ckpt \
batch_size=16 \
model.diffusion.config.depth=24
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
