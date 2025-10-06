# Audio Intelligence

# Overview

This repository contains implementation of several state-of-the-art audio intelligence research projects from NVIDIA. 

# Projects

## Audio Understanding, Generation, and Reasoning

### Audio Flamingo 3

**Advancing Audio Intelligence with Fully Open Large Audio Language Models**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2507.08128"><img src="https://img.shields.io/badge/arXiv-2507.08128-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/AF3/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_3"><img src='https://img.shields.io/badge/Github-Audio Flamingo 3-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-3">
    <img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/nvidia/audio-flamingo-3-chat">
    <img src="https://img.shields.io/badge/🤗-Checkpoints (Chat)-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/spaces/nvidia/audio_flamingo_3">
    <img src="https://img.shields.io/badge/🤗-Gradio Demo (7B)-5F9EA0.svg" style="margin-right: 5px;">
  </a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://huggingface.co/datasets/nvidia/AudioSkills">
    <img src="https://img.shields.io/badge/🤗-Dataset: AudioSkills--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/LongAudio">
    <img src="https://img.shields.io/badge/🤗-Dataset: LongAudio--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Chat">
    <img src="https://img.shields.io/badge/🤗-Dataset: AF--Chat-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Think">
    <img src="https://img.shields.io/badge/🤗-Dataset: AF--Think-ED5A22.svg">
  </a>
</div>

<br>

Audio Flamingo 3 is a 7B audio language model using the [LLaVA](https://arxiv.org/abs/2304.08485) architecture for audio understanding. We trained our unified AF-Whisper audio encoder based on [Whisper](https://arxiv.org/abs/2212.04356) to handle understanding beyond speech recognition. We included speech-related tasks in Audio Flamingo 3 and scaled up the training dataset to about 50M audio-text pairs. Therefore, Audio Flamingo 3 is able to handle all three modalities in audio: **sound**, **music**, and **speech**. It outperforms prior SOTA models on a number of understanding and reasoning benchmarks. Audio Flamingo 3 can take up to 10 minutes of audio inputs, and has a streaming TTS module (AF3-Chat) to output voice. 

---

### ETTA

**Elucidating the Design Space of Text-to-Audio Models**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2412.19351"><img src="https://img.shields.io/badge/arXiv-2412.19351-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/ETTA/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/tree/main/ETTA"><img src='https://img.shields.io/badge/Github-ETTA-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-intelligence.svg?style=social"></a>
</div>

ETTA is a 1.4B latent diffusion model for text-to-audio generation. We trained ETTA on over 1M synthetic captions annotated by Audio Flamingo, and proved that this approach can lead to high quality audio generation as well as emergent abilities with scale. 

---

### UALM
<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org"><img src="https://img.shields.io/badge/arXiv-coming_soon-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/UALM/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/tree/main/UALM"><img src='https://img.shields.io/badge/Github-UALM-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-intelligence.svg?style=social"></a>
</div>

<!-- ## Audio Enhancement
### CleanUNet
### A2SB

## Speech Models
### WaveGlow -->

# License
Please refer to each project folder for the detailed licenses. 