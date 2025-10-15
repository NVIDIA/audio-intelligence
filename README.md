# Audio Intelligence

# Overview

This repository contains implementation of several state-of-the-art audio intelligence research projects from NVIDIA. 

# Projects

# Audio Understanding, Generation, and Reasoning

### Audio Flamingo 3 (Audio Understanding)

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

### UALM (Audio Understanding and Generation)
<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org"><img src="https://img.shields.io/badge/arXiv-coming_soon-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/UALM/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/tree/main/UALM"><img src='https://img.shields.io/badge/Github-UALM-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-intelligence.svg?style=social"></a>
</div>

---

### ETTA (Audio Generation)

**Elucidating the Design Space of Text-to-Audio Models**

**Improving Text-To-Audio Models with Synthetic Captions**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2412.19351"><img src="https://img.shields.io/badge/arXiv-2412.19351-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://arxiv.org/abs/2406.15487"><img src="https://img.shields.io/badge/arXiv-2406.15487-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/ETTA/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/tree/main/ETTA"><img src='https://img.shields.io/badge/Github-ETTA-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-intelligence.svg?style=social"></a>
</div>

ETTA is a 1.4B latent diffusion model for text-to-audio generation. We trained ETTA on over 1M synthetic captions annotated by Audio Flamingo, and proved that this approach can lead to high quality audio generation as well as emergent abilities with scale. 

---

### Fugatto 1 (Audio Editing and Generation)
**Foundational Generative Audio Transformer Opus 1**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://openreview.net/pdf?id=B2Fqu7Y2cd"><img src="https://img.shields.io/badge/paper-Fugatto 1-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://blogs.nvidia.com/blog/fugatto-gen-ai-sound-model/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
</div>

<br>

Fugatto is a versatile audio synthesis and transformation model capable of following free-form text instructions with optional audio inputs.

---

### DCASE 2025 Challenge Task 5 (Audio Understanding Challenge)
**Multi-Domain Audio Question Answering Toward Acoustic Content Reasoning in the DCASE 2025 Challenge**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2505.07365"><img src="https://img.shields.io/badge/arXiv-2505.07365-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://dcase.community/challenge2025/task-audio-question-answering"><img src="https://img.shields.io/badge/DCASE-Task Description-228B22" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/datasets/PeacefulData/2025_DCASE_AudioQA_Official"><img src='https://img.shields.io/badge/🤗-PeacefulData-ED5A22' style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/PeacefulData/2025_DCASE_AudioQA_Baselines"><img src='https://img.shields.io/badge/🤗-Baselines-ED5A22' style="margin-right: 5px;"></a>
</div>

---

### TangoFlux (Audio Generation)
**Tangoflux: Super fast and faithful text to audio generation with flow matching and clap-ranked preference optimization**


<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2412.21037"><img src="https://img.shields.io/badge/arXiv-2412.21037-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://tangoflux.github.io/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/declare-lab/TangoFlux"><img src='https://img.shields.io/badge/Github-TangoFlux-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/declare-lab/TangoFlux/stargazers"><img src="https://img.shields.io/github/stars/declare-lab/TangoFlux.svg?style=social"></a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/declare-lab/TangoFlux">
    <img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/spaces/declare-lab/TangoFlux">
    <img src="https://img.shields.io/badge/🤗-Gradio-5F9EA0.svg" style="margin-right: 5px;">
  </a>
</div>

<br>

TangoFlux is an efficient and high-quality text-to-audio model with FluxTransformer and CLAP-ranked preference optimization. This project was in collaboration with SUTD and Lambda Labs.

---

### OMCAT (Audio-Visual Understanding)
**Omni Context Aware Transformer**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2410.12109"><img src="https://img.shields.io/badge/arXiv-2410.12109-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://om-cat.github.io/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
</div>

OMCAT is an audio-visual understanding model with ROTE (Rotary Time Embeddings).

<br>
<br>

---

<br>
<br>

# Representation Learning

### UniWav (Speech Codec)
**Towards Unified Pre-training for Speech Representation Learning and Generation**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2503.00733"><img src="https://img.shields.io/badge/arXiv-2503.00733-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://alexander-h-liu.github.io/uniwav-demo.github.io/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
</div>


<br>
<br>

---

<br>
<br>

# Audio Enhancement
### A2SB (Bandwidth Extension and Inpainting)
**Audio-to-Audio Schrodinger Bridges**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2501.11311"><img src="https://img.shields.io/badge/arXiv-2501.11311-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/A2SB/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/diffusion-audio-restoration"><img src='https://img.shields.io/badge/Github-Diffusion_Audio_Restoration-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/diffusion-audio-restoration/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/diffusion-audio-restoration.svg?style=social"></a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge"><img src="https://img.shields.io/badge/🤗-Checkpoints_(1_split)-ED5A22.svg" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge"><img src="https://img.shields.io/badge/🤗-Checkpoints_(2_split)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

<br> 

A2SB is an audio restoration model tailored for high-res music at 44.1kHz. It is capable of both bandwidth extension (predicting high-frequency components) and inpainting (re-generating missing segments). Critically, A2SB is end-to-end without need of a vocoder to predict waveform outputs, and able to restore hour-long audio inputs. A2SB is capable of achieving state-of-the-art bandwidth extension and inpainting quality on several out-of-distribution music test sets.

--- 

### CleanUNet (Speech Denoising)

**CleanUNet: Speech Denoising in the Waveform Domain with Self-Attention**

**Cleanunet 2: A hybrid speech denoising model on waveform and spectrogram**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2202.07790"><img src="https://img.shields.io/badge/arXiv (CleanUNet)-2202.07790-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://arxiv.org/abs/2309.05975"><img src="https://img.shields.io/badge/arXiv (CleanUNet2)-2309.05975-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/projects/cleanunet/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/CleanUNet"><img src='https://img.shields.io/badge/Github-CleanUNet-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/CleanUNet/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/CleanUNet.svg?style=social"></a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://github.com/NVIDIA/CleanUNet/tree/main/exp/DNS-large-full/checkpoint"><img src="https://img.shields.io/badge/Checkpoints_(DNS_large_full)-ED5A22.svg" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/CleanUNet/tree/main/exp/DNS-large-high/checkpoint"><img src="https://img.shields.io/badge/Checkpoints_(DNS_large_high)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

<br> 

CleanUNet is a causal speech denoising model on the raw waveform. CleanUNet 2 is a speech denoising model that combines the advantages of waveform denoiser and spectrogram denoiser and achieves the best of both worlds.

<br>
<br>

---

<br>
<br>

# Text-to-Speech Models
### BigVGAN-v2
**A Universal Neural Vocoder with Large-Scale Training**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2206.04658"><img src="https://img.shields.io/badge/arXiv-2206.04658-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/projects/bigvgan/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://developer.nvidia.com/blog/achieving-state-of-the-art-zero-shot-waveform-audio-generation-across-audio-types/"><img src="https://img.shields.io/badge/Blogpost-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/BigVGAN"><img src='https://img.shields.io/badge/Github-BigVGAN-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/BigVGAN/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/BigVGAN.svg?style=social"></a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/collections/nvidia/bigvgan-66959df3d97fd7d98d97dc9a">
    <img src="https://img.shields.io/badge/🤗-All_Checkpoints-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/spaces/nvidia/BigVGAN">
    <img src="https://img.shields.io/badge/🤗-Gradio Demo-5F9EA0.svg" style="margin-right: 5px;">
  </a>
</div>

<br>

BigVGAN-v2 is a widely-used universal vocoder that generalizes well for various out-of-distribution scenarios without fine-tuning. We release our checkpoints with various configurations such as sampling rates. 

---

### P-Flow and A2-Flow

**P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting**

**A2-Flow: Alignment-Aware Pre-training for Speech Synthesis with Flow Matching**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Paper-Conference.pdf"><img src="https://img.shields.io/badge/paper-P_Flow-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://openreview.net/pdf?id=e2p1BWR3vq"><img src="https://img.shields.io/badge/paper-A2_Flow-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/projects/pflow/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
</div>

<br>

Please refer to the [Magpie-TTS API](https://build.nvidia.com/nvidia/magpie-tts-flow) for interactive demo of NVIDIA's TTS models that leveraged techniques of these papers. 

---

### DiffWave
**A Versatile Diffusion Model for Audio Synthesis**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2009.09761"><img src="https://img.shields.io/badge/arXiv-2009.09761-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://diffwave-demo.github.io/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://github.com/philsyn/DiffWave-Vocoder"><img src='https://img.shields.io/badge/Github-PyTorch reimplementation with checkpoints-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/lmnt-com/diffwave"><img src='https://img.shields.io/badge/Github-Another PyTorch reimplementation-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/albertfgu/diffwave-sashimi"><img src='https://img.shields.io/badge/Github-DiffWave Sashimi-9C276A' style="margin-right: 5px;"></a>
</div>
<br> 

DiffWave is the first diffusion model for raw waveform synthesis. It is a versatile waveform synthesis model for speech and non-speech generation. 
<br> 

---

### WaveGlow 
**A Flow-based Generative Network for Speech Synthesis**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683143"><img src="https://img.shields.io/badge/IEEE-paper-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/WaveGlow/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/waveglow"><img src='https://img.shields.io/badge/Github-WaveGlow-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/waveglow/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/waveglow.svg?style=social"></a>
</div>

<br> 

---

### Flowtron
**Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2005.05957"><img src="https://img.shields.io/badge/arxiv-2005.05957-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/Flowtron/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/flowtron"><img src='https://img.shields.io/badge/Github-Flowtron-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/flowtron/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/flowtron.svg?style=social"></a>
</div>

<br> 

---


### RAD

**RAD-TTS: Parallel Flow-Based TTS with Robust Alignment Learning and Diverse Synthesis**

**RAD-MMM: Multilingual Multiaccented Multispeaker TTS with RADTTS**

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://openreview.net/forum?id=0NQwnnwAORi"><img src="https://img.shields.io/badge/paper-RAD TTS-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/RADTTS/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/radtts"><img src='https://img.shields.io/badge/Github-RAD TTS-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/radtts/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/radtts.svg?style=social"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://www.isca-archive.org/interspeech_2023/badlani23_interspeech.pdf"><img src="https://img.shields.io/badge/paper-RAD MMM-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/projects/radmmm/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/RAD-MMM"><img src='https://img.shields.io/badge/Github-RAD MMM-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/RAD-MMM/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/RAD-MMM.svg?style=social"></a>
</div>

# License
The codes for different projects may be released under different licenses, including MIT, NVIDIA OneWay Noncommercial License, NVIDIA Sourcecode License, and so on. Please refer to each project folder or their original GitHub links for the detailed licenses. 