FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# install essential apt packages first
RUN apt-get update && apt-get install -y \
    git build-essential ninja-build libsndfile1 ffmpeg espeak-ng sox libsox-fmt-all apt-transport-https wget curl gnupg libffi-dev tmux htop vim

#############################
# Build NATTEN from source for CUDA enablement
#############################
# Clone the NATTEN repository with recursive submodules.
RUN git clone --recursive https://github.com/SHI-Labs/NATTEN.git /app/NATTEN
WORKDIR /app/NATTEN
# Install dependencies from requirements.txt
RUN pip install -r requirements.txt
# Uncomment one of the following if specific CUDA architecture targeting is required
RUN make WITH_CUDA=1 CUDA_ARCH="8.0;8.6;8.7"
# Run unit tests to verify the build
RUN pip install pytest && make test

#############################
# install basic pip packages first to build others (pypesq, etc)
#############################
RUN pip install numpy soundfile pedalboard jupyter notebook packaging ninja
RUN pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
RUN pip install flash-attn --no-build-isolation

#############################
# install related pip packages with loose version checks
#############################
RUN pip install alias-free-torch auraloss descript-audio-codec einops einops-exts ema-pytorch encodec gradio huggingface_hub importlib-resources k-diffusion laion-clap local-attention pandas prefigure pytorch_lightning lightning pywavelets pypesq safetensors sentencepiece torchmetrics tqdm transformers datasets v-diffusion-pytorch vector-quantize-pytorch wandb webdataset x-transformers phaseaug perceiver-pytorch diffusers["torch"] nnAudio audiobox_aesthetics deepspeed openai-whisper accelerate boto3 sphn phonemizer psutil deepspeed lhotse pyyaml joblib typeguard humanfriendly lmdb

####################################################################################################################
# end of Dockerfile
####################################################################################################################