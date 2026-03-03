# PyTorch Implementation of UALM

This repo contains PyTorch implementation of the following **ICLR 2026 Oral** paper: 

**UALM: Unified Audio Language Model for Understanding, Generation, and Reasoning**

**Authors**: Jinchuan Tian*, Sang-gil Lee*, Zhifeng Kong*, Sreyan Ghosh, Arushi Goel, Chao-Han Huck Yang, Wenliang Dai, Zihan Liu, Hanrong Ye, Shinji Watanabe, Mohammad Shoeybi, Bryan Catanzaro, Rafael Valle, Wei Ping

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2510.12000"><img src="https://img.shields.io/badge/arXiv-2510.12000-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/UALM/"><img src="https://img.shields.io/badge/Demo_page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/tree/main/UALM"><img src='https://img.shields.io/badge/Github-UALM-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-intelligence/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-intelligence.svg?style=social"></a>
</div>

<br>

UALM is an advanced Audio-Language Model that unifies text and audio tasks including: text problem solving, audio understanding, text-to-audio generation, and multimodal reasoning across modalities. UALM matches the quality of state-of-the-art specialized models in each task, and is the first demonstration of cross-modal generative reasoning in the audio research domain.

<br>
<br>

# Data Preparation
Data preparation is perhaps the most complicated part of launching this model. Since there are too many files in audio-text modeling, we implemented a custom tarball-based storage (similar to WebDatasets) that is suitable for efficient audio storage and loading. 

We have three types of data: text-only, text-to-audio, and audio-understanding. 

(1) Get raw data (most difficult)
- Text-only: we collect data from NVIDIA's text datasets: [dataset 1](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) and [dataset 2](https://huggingface.co/datasets/nvidia/AceMath-Instruct-Training-Data). Each row of the ```jsonl``` is ```{"input": [{"role": "user", "content": "Question"}], "output": "Output"}```.
- Audio understanding: follow Audio Flamingo 3 datasets (see [here](https://huggingface.co/datasets/nvidia/AudioSkills)). We need a list of ```json``` files in the format of 
```json
[
  {
    "id": "audio_id",
    "sound": "/abs/path/to/wav",
    "duration": 10.0,
    "conversations": [{"from": "human", "value": "<sound>\nQuestion"}, {"from": "gpt", "value": "Answer"}]
  },
  ...
```
- Text-to-audio: follow ETTA datasets (see [here](https://arxiv.org/pdf/2412.19351)). We need a list of ```jsonl``` files in the format of 
```json
{"location": "/abs/path/to/wav", "start_time":0.0, "end_time":10.0, "duration":10.0, "caption": "Caption", "sample_rate": 22050}
```

(2) Process raw data into shards and tar files. No need to process text data in this step.
- Audio understanding: fill in ```ualm/tools/object_storage_manifest/manifest_config_examples/config_AF.yaml```
- Text-to-audio generation: fill in ```ualm/tools/object_storage_manifest/manifest_config_examples/config_ETTA.yaml```

Remember to fill in all paths properly whenever there is ```/path/to/...``` in the yaml. Then run 
```bash
python batch_create_manifests.py --config manifest_config_examples/config_AF.yaml
python batch_create_manifests.py --config manifest_config_examples/config_ETTA.yaml
```

(3) Prepare UALM manifest for each experiment
- Make a symlink ```recipes/ualm_all_task/ualm/.tmp``` to map to ```.tmp/```. 
- Fill in yaml files in ```ualm/tools/tar_to_ualm_manifest_converter/manifest_config_examples```. You could (and should) put every thing into a single yaml (so ```manifests.train``` is a list of all datasets you want to train on). 
- Then run
```
python tools/tar_to_ualm_manifest_converter/convert_tar_to_ualm_manifest.py \
    --config tools/tar_to_ualm_manifest_converter/manifest_config_examples/config_NAME.yaml \
    --output-dir .tmp/manifest_NAME
```

<br>
<br>

# Launch the training and inference:

(1) Go to the directory.
```bash
cd recipes/ualm_all_task/ualm
```
Note that ```ualm_all_task``` is the experiment name that you can change but do not change the ```ualm``` name after it. This experiment name is ideal for managing experiments with major differences (e.g. a TTA-only model vs a multi-task model).

(2) Train the model
```bash
bash launch.sh
```
Note that ```exp_dir``` in ```launch.sh``` is a separate experiment name that the previous name. It is ideal to distinguish different training parameters such as number of nodes. 

(3) Inference
```bash
bash inference.sh
```

<br>
<br>

# Environment

See Dockerfile for exact docker image creation. Alternatively, below is the environment based on Conda. 

## Local Installation (miniconda)
(1) Ensure you have a valid Python environment. 

(2) Install Pytorch. A newer version is appreciated.
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

(3) Install dependencies
```bash
pip install -r requirement.txt
```

(4) Install Flash attention. Recommend to build from source
```bash
# from pre-built wheel
pip install flash-attn --no-build-isolation 
# or from source
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

(5) Install TorchCodec.
```bash
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu129
```

<br>
<br>

# Additional Notes
- Current training config points to the Qwen2.5-Omni-7B tokenizer for audio inputs. You could switch to AF-Whisper (Audio Flamingo 3 audio encoder) similar to [this thread](https://huggingface.co/nvidia/audio-flamingo-3/discussions/2), store it to a HF checkpoint, and change ```encoder_hf_model_tag``` value to ```/path/to/huggingface_cache/AF-Whisper```. 

<br>
<br>

# Citation
```
@inproceedings{
    tian2026ualm,
    title={{UALM}: Unified Audio Language Model for Understanding, Generation and Reasoning},
    author={Jinchuan Tian and Sang-gil Lee and Zhifeng Kong and Sreyan Ghosh and Arushi Goel and Chao-Han Huck Yang and Wenliang Dai and Zihan Liu and Hanrong Ye and Shinji Watanabe and Mohammad Shoeybi and Bryan Catanzaro and Rafael Valle and Wei Ping},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=TsdlOjcQNu}
}
```

<br>
<br>

# Code Reference

The code structure is based on [ESPNet](https://github.com/espnet/espnet).