#############################
# start of Dockerfile
#############################
# start with latest official pytorch docker
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# to minimize node stress
ENV MAX_JOBS=8
ENV NINJA_JOBS=8

# install essential apt packages first
RUN apt-get update && apt-get install -y \
    git build-essential ninja-build libsndfile1 ffmpeg espeak-ng sox libsox-fmt-all apt-transport-https wget curl gnupg libffi-dev

# install pip and setuptools to latest
RUN pip install -U pip setuptools

# Set the working directory
WORKDIR /app

#############################
# install stable-audio-tools relatad pip packages with loose version checks
#############################
RUN pip install flash-attn --no-build-isolation
RUN pip install numpy soundfile pedalboard jupyter notebook packaging alias-free-torch auraloss descript-audio-codec einops einops-exts ema-pytorch encodec gradio huggingface_hub importlib-resources k-diffusion laion-clap local-attention pandas prefigure pytorch_lightning lightning pywavelets pypesq safetensors sentencepiece torchmetrics tqdm transformers v-diffusion-pytorch vector-quantize-pytorch wandb webdataset x-transformers diffusers["torch"] deepspeed

#############################
# end of Dockerfile
#############################