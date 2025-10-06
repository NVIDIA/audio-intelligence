# Copyright (c) 2025 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import numpy as np

import re
import json
import torch
import torchaudio
import librosa

from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import (
    load_ckpt_state_dict,
    remove_weight_norm_from_model,
)
from stable_audio_tools.utils.addict import Dict as AttrDict
from tqdm import tqdm
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_local_pretrained_model(model_ckpt_path: str, strict: bool=True):
    """
    Load a model from a local checkpoint file.
    
    Args:
        model_ckpt_path: Path to the model checkpoint file (.ckpt or .safetensors)
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        
    Returns:
        model: The loaded model
        model_config: The model configuration
    """
    config_path = os.path.join(os.path.dirname(model_ckpt_path), "config.json")
    
    with open(config_path) as f:
        config = json.load(f)
        
    try: # models has model_config within "model_config key"
        model_config = config["model_config"]
    except KeyError: # public model from stability json config itself is model_config
        model_config = config
        
    model = create_model_from_config(model_config)
    
    if model_config.get("remove_pretransform_weight_norm", ''):
        remove_weight_norm_from_model(model.pretransform)
        
    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path), strict=strict)
    
    return model, model_config


def generate_audio_from_caption(
    model,
    steps,
    cfg_scale,
    conditioning,
    sample_size,
    device,
    sigma_min,
    sigma_max,
    sampler_type,
    bad_model,
    autoguidance_scale,
    batch_size=1
):
    """
    Generate audio from a text caption using the diffusion model.
    
    Args:
        model: The diffusion model
        steps: Number of diffusion steps
        cfg_scale: Classifier-free guidance scale
        conditioning: Text conditioning information
        sample_size: Size of the audio sample to generate
        device: Device to run generation on (cuda or cpu)
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sampler_type: Type of sampler to use (e.g., 'euler', 'dpmpp-3m-sde')
        bad_model: Optional bad model for autoguidance
        autoguidance_scale: Scale for autoguidance
        
    Returns:
        output: Generated audio tensor
    """
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        batch_size=batch_size,
        sample_size=sample_size,
        device=device,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        bad_model=bad_model,
        autoguidance_scale=autoguidance_scale,
    )
    
    output = (
        output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).cpu()
    )  # [-1, 1] float
    return output

def resample(audio, orig_sr, target_sr):
    """
    Resample audio to the target sample rate.
    
    Args:
        audio: Audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio

def save_audio_and_caption(output_dir, base_name, index, batch_idx, audio, caption, target_sample_rate):
    """
    Save generated audio and corresponding caption to files.
    
    Args:
        output_dir: Directory to save files
        base_name: Base name for the output files
        index: Index for multiple captions with the same base name
        batch_idx: Index for multiple variations of the same caption
        audio: Audio data to save
        caption: Caption text to save
        target_sample_rate: Sample rate for the output audio file
    """
    
    # For batch variations, add _varN suffix after the caption index
    if batch_idx is not None:
        audio_file_name = f"{base_name}_{index}_var{batch_idx}.wav"
        caption_file_name = f"{base_name}_{index}_var{batch_idx}.txt"
    else:
        audio_file_name = f"{base_name}_{index}.wav"
        caption_file_name = f"{base_name}_{index}.txt"

    audio_file_path = os.path.join(output_dir, audio_file_name)
    caption_file_path = os.path.join(output_dir, caption_file_name)

    sf.write(audio_file_path, audio, samplerate=target_sample_rate)
    with open(caption_file_path, "w") as caption_file:
        caption_file.write(caption)


def main(
    json_file_path: str,
    text_dir: str,
    output_dir: str,
    model_ckpt_path: str,
    sampler_type: str,
    steps: int = 100,
    cfg_scale: float = 3.5,
    seconds_start: int = 0,
    seconds_total: int = 10,
    bad_model_ckpt_path = None,
    autoguidance_scale = None,
    text_prompt: str = None,
    target_sample_rate: int = 48000,
    batch_size: int = 1,
):
    """
    Main function to generate audio from text prompts using a diffusion model.
    
    Args:
        json_file_path: Path to JSON file containing captions
        text_dir: Directory containing text files with captions
        output_dir: Directory to save generated audio
        model_ckpt_path: Path to model checkpoint
        sampler_type: Type of sampler to use (default: 'euler')
        steps: Number of diffusion steps (default: 100)
        cfg_scale: Classifier-free guidance scale (default: 3.5)
        seconds_start: Start time of audio in seconds (default: 0)
        seconds_total: Total duration of audio in seconds (default: 10)
        bad_model_ckpt_path: Path to bad model for autoguidance
        autoguidance_scale: Scale for autoguidance
        text_prompt: Text prompt from command line
        target_sample_rate: Sample rate for output audio (default: 48000),
        batch_size: Batch size for multiple audio variant generation (default: 1)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO(main)] Loading model from {model_ckpt_path}")
    if model_ckpt_path.endswith(".ckpt") or model_ckpt_path.endswith(".safetensors"):
        try:
            model, model_config = get_local_pretrained_model(
                model_ckpt_path, strict=True
            )
        except Exception as e:
            print(
                f"[WARNING(main)] failed to load model {model_ckpt_path} . trying again with strict=False! {e}"
            )
            model, model_config = get_local_pretrained_model(
                model_ckpt_path, strict=False
            )
    else:
        model, model_config = get_pretrained_model(model_ckpt_path)

    # convert it to AttrDict (dot-accessible dictionary)
    model_config = AttrDict(model_config)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model = model.to(device).eval().requires_grad_(False)

    if bad_model_ckpt_path is not None:
        assert (
            autoguidance_scale is not None
        ), "autoguidance_scale is not provided for bad_model"
        bad_model, bad_model_config = get_local_pretrained_model(
            bad_model_ckpt_path, strict=True
        )
        bad_model = bad_model.to(device).eval().requires_grad_(False)
        bad_model_config = AttrDict(bad_model_config)
    else:
        bad_model, bad_model_config = None, None

    diffusion_objective = model_config["model"]["diffusion"].get(
        "diffusion_objective", "v"
    )

    # Set sigma_min, sigma_max, scale_phi based on the diffusion objective
    # sampler_type is defined externally
    if diffusion_objective == "v": # following stable-audio-tools
        sigma_min = 0.3
        sigma_max = 500
    elif diffusion_objective == "rectified_flow": # OT-CFM defaults
        sigma_min = None
        sigma_max = 1.0
    else:
        raise ValueError(f"Unknown diffusion objective: {diffusion_objective}")

    print(f"######################################################################")
    if json_file_path:
        print(f"json_file: {json_file_path}")
    elif text_dir:
        print(f"text_dir: {text_dir}")
    elif text_prompt:
        print(f"text_prompt: {text_prompt}")
    print(f"output_dir: {output_dir}")
    print(f"model_ckpt_path: {model_ckpt_path}")
    print(f"diffusion_objective: {diffusion_objective}")
    print(f"steps: {steps}")
    print(f"cfg_scale: {cfg_scale}")
    print(f"seconds_start: {seconds_start}")
    print(f"seconds_total: {seconds_total}")
    print(f"sigma_min: {sigma_min}")
    print(f"sigma_max: {sigma_max}")
    print(f"sampler_type: {sampler_type}")
    print(f"bad_model_ckpt_path: {bad_model_ckpt_path}")
    print(f"autoguidance_scale: {autoguidance_scale}")
    print(f"target_sample_rate: {target_sample_rate}")
    print(f"batch_size: {batch_size}")
    print(f"######################################################################")

    # Read data from json_file, text_dir, or text_prompt
    if json_file_path:
        with open(json_file_path, "r") as f:
            data = [json.loads(line) for line in f]
        items = data
    elif text_dir:
        text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
        items = []
        for text_file in text_files:
            base_name = os.path.splitext(text_file)[0]
            caption_file_path = os.path.join(text_dir, text_file)
            with open(caption_file_path, "r") as f:
                caption = f.read().strip()
            item = {
                "base_name": base_name,
                "captions": caption,
            }
            items.append(item)
    elif text_prompt:
        # Handle multiple text prompts provided as separate arguments
        items = []
        for i, prompt in enumerate(text_prompt):
            items.append({
                "base_name": f"etta_{i}" if len(text_prompt) > 1 else "etta",
                "captions": prompt,
            })
    else:
        raise ValueError("Either json_file_path, text_dir, or text_prompt must be provided.")

    basename_count = {}
    for item in tqdm(items, desc="Generating"):
        if json_file_path:
            location = item["location"]
            caption = item["captions"]
            base_name = os.path.splitext(os.path.basename(location))[0]
        else:  # text_dir or text_prompt
            base_name = item["base_name"]
            caption = item["captions"]

        # Ensure unique filenames for audiocaps with multiple captions
        if base_name not in basename_count:
            basename_count[base_name] = 0
        index = basename_count[base_name]
        basename_count[base_name] += 1

        # prepare dict format for the model
        conditioning = [
            {
                "prompt": caption,
                "prompt_global": caption,
                "seconds_start": seconds_start,
                "seconds_total": seconds_total,
            }
        ] * batch_size

        # Generate audio using the appropriate settings for diffusion objective
        with torch.inference_mode():
            batch_output = generate_audio_from_caption(
                model,
                steps,
                cfg_scale,
                conditioning,
                sample_size,
                device,
                sigma_min,
                sigma_max,
                sampler_type,
                bad_model,
                autoguidance_scale,
                batch_size=batch_size,
            )
            
        # Process each item in the batch
        for batch_idx in range(batch_size):
            single_output = batch_output[batch_idx]
            
            if single_output.shape[-1] > seconds_total * sample_rate:
                single_output = single_output[..., : seconds_total * sample_rate]
            
            single_output = single_output.numpy()  # Convert to numpy array
            
            if sample_rate != target_sample_rate:
                # Resample to target sample rate
                single_output= resample(single_output, sample_rate, target_sample_rate)

            # Change output shape to [T, channel] for sf.write
            single_output = np.transpose(single_output)
            
            save_audio_and_caption(
                output_dir=output_dir, 
                base_name=base_name,
                index=index, 
                batch_idx=batch_idx if batch_size > 1 else None,  # Only add variation suffix if batch_size > 1
                audio=single_output, 
                caption=caption, 
                target_sample_rate=target_sample_rate
            )

    print(f"----------------------------------------------------------------------")
    if json_file_path:
        print(f"Inference complete for {json_file_path}")
    elif text_dir:
        print(f"Inference complete for {text_dir}")
    elif text_prompt:
        print(f"Inference complete for text prompt: {text_prompt}")
    print(f"output_dir: {output_dir}")
    print(f"----------------------------------------------------------------------")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate audio from captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage for text prompt:
    python inference_tta.py \\
    --text_prompt "A hip-hop track using sounds from a construction site—hammering nails as the beat, drilling sounds as scratches, and metal clanks as rhythm accents." "A saxophone that sounds like meowing of cat." \\
    --output_dir ./tmp \\
    --model_ckpt_path /path/to/model.ckpt
    
Example usage with text directory:
    python inference_tta.py \\
    --text_dir /path/to/captions \\
    --output_dir ./tmp \\
    --model_ckpt_path /path/to/model.ckpt
    
Example usage with json manifest:
    python inference_tta.py \\
    --json_file /path/to/manifest.json \\
    --output_dir ./tmp \\
    --model_ckpt_path /path/to/model.ckpt
"""
    )
    
    # Path to the model checkpoint
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint or pretrained model",
    )
    
    # The script supports three modes:
    # 1. Text prompt
    # 2. Text directory
    # 3. JSON file
    parser.add_argument(
        "--text_prompt",
        type=str,
        nargs='*',
        required=False,
        help="Text prompt(s) for audio generation. Provide multiple prompts as separate arguments.",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        required=False,
        help="Directory containing caption text files. Each text file should contain a single line of the caption text prompt",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=False,
        help="Path to the test data NDJSON manifest file that contains 'location' key as GT audio path (used as basename for generated audio) and 'captions' key as text prompt",
    )
    
    # Output directory for generated audio files
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="Directory to save the generated audio files (default: ./tmp)",
    )

    # Audio output settings
    parser.add_argument(
        "--target_sample_rate", type=int, default=48000, 
        help="Sample rate for output audio (default: 48000). The script resamples the generated audio to the specified target sample rate."
    )
    
    # Sampling hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for multiple audio variant generation for each caption (default: 1)",
    )

    parser.add_argument(
        "--sampler_type", type=str, default="dpmpp", help="Sampler type for diffusion (default: dpmpp)"
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of diffusion steps (default: 100)"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=3.5, help="CFG scale for guidance (default: 3.5)"
    )
    parser.add_argument(
        "--seconds_start", type=int, default=0, help="Starting seconds of the generated sample (default: 0)"
    )
    parser.add_argument(
        "--seconds_total", type=int, default=10, help="Total duration in seconds of the generated sample (default: 10)"
    )

    # Optional parameters for autoguidance
    parser.add_argument(
        "--bad_model_ckpt_path",
        type=str,
        default=None,
        help="Path to the bad model checkpoint for autoguidance (optional)",
    )
    parser.add_argument(
        "--autoguidance_scale",
        type=float,
        default=None,
        help="Scale for autoguidance using bad model (required if bad_model_ckpt_path is provided)",
    )
    
    args = parser.parse_args()

    # Ensure that only one of --json_file, --text_dir, or --text_prompt is provided
    input_options = [args.json_file, args.text_dir, args.text_prompt]
    if sum(x is not None for x in input_options) != 1:
        parser.error("Please provide exactly one of --json_file, --text_dir, or --text_prompt.")

    main(
        args.json_file,
        args.text_dir,
        args.output_dir,
        args.model_ckpt_path,
        args.sampler_type,
        args.steps,
        args.cfg_scale,
        args.seconds_start,
        args.seconds_total,
        args.bad_model_ckpt_path,
        args.autoguidance_scale,
        args.text_prompt,
        args.target_sample_rate,
        args.batch_size,
    )
