import gc
import platform
import os

import numpy as np
import gradio as gr
import json 
import torch
import torchaudio

from stable_audio_tools.interface.aeiou import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict

# global sample_rate and sample_size vars. This is useful for overwriting model_config
# ex: extrapolate 10-sec only trained model to 20-sec. Since ours used RoPE it supports arbitrary length
model = None
sample_rate = 44100
sample_size = 441000

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
        
    try: # internal models has model_config within "model_config key"
        model_config = config["model_config"]
    except KeyError: # public model from stability json config itself is model_config
        model_config = config
        
    model = create_model_from_config(model_config)
    
    if model_config.get("remove_pretransform_weight_norm", ''):
        remove_weight_norm_from_model(model.pretransform)
        
    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path), strict=strict)
    
    return model, model_config


def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=10,
        cfg_scale=3.5,
        steps=100,
        preview_every=None,
        seed=-1,
        sampler_type="euler",
        sigma_min=None,
        sigma_max=1.0,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1    
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Return fake stereo audio
    conditioning = [{
        "prompt": prompt, 
        "prompt_global": prompt, 
        "seconds_start": seconds_start, 
        "seconds_total": seconds_total
        }] * batch_size

    if negative_prompt:
        negative_conditioning = [{
            "prompt": negative_prompt, 
            "prompt_global": negative_prompt,
            "seconds_start": seconds_start, 
            "seconds_total": seconds_total
        }] * batch_size
    else:
        negative_conditioning = None
        
    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # If inpainting, send mask args
    # This will definitely change in the future
    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 

    # Do the audio generation
    with torch.inference_mode():
        audio = generate_diffusion_cond(
            model, 
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            sample_size=input_sample_size,
            sample_rate=sample_rate,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            mask_args = mask_args,
            callback = progress_callback if preview_every is not None else None,
            scale_phi = cfg_rescale
        )

    # Convert to WAV file
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save("tmp/output.wav", audio, sample_rate)

    # Let's look at a nice spectrogram too
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("tmp/output.wav", [audio_spectrogram, *preview_images])

def generate_uncond(
        steps=250,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        batch_size=1,
        preview_every=None
        ):

    global preview_images

    preview_images = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:

            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)

            denoised = rearrange(denoised, "b d n -> d (b n)")

            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)

            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    audio = generate_diffusion_uncond(
        model, 
        steps=steps,
        batch_size=batch_size,
        sample_size=input_sample_size,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        callback = progress_callback if preview_every is not None else None
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram, *preview_images])

def generate_lm(
        temperature=1.0,
        top_p=0.95,
        top_k=0,    
        batch_size=1,
        ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    audio = model.generate_audio(
        batch_size=batch_size,
        max_gen_len = sample_size//model.pretransform.downsampling_ratio,
        conditioning=None,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram])


def create_uncond_sampling_ui(model_config):   
    generate_button = gr.Button("Generate", variant='primary', scale=1)
    
    with gr.Row(equal_height=False):
        with gr.Column():            
            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=50, label="Steps")

            with gr.Accordion("Sampler params", open=False):
            
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

            # Sampler params
                with gr.Row():
                    # sampler_type_dropdown = gr.Dropdown(["euler", "heun", "dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="heun")
                    sampler_type_dropdown = gr.Dropdown(["euler", "heun"], label="Sampler type", value="euler")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Sigma max")

            with gr.Accordion("Init audio", open=False):
                init_audio_checkbox = gr.Checkbox(label="Use init audio")
                init_audio_input = gr.Audio(label="Init audio")
                init_noise_level_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.75, label="Init noise level")

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])
    
    generate_button.click(fn=generate_uncond, 
        inputs=[
            steps_slider, 
            seed_textbox, 
            sampler_type_dropdown, 
            sigma_min_slider, 
            sigma_max_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
        ], 
        outputs=[
            audio_output, 
            audio_spectrogram_output
        ], 
        api_name="generate")

def create_sampling_ui(model_config, inpainting=False):
    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(show_label=False, placeholder="Prompt")
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
        generate_button = gr.Button("Generate", variant='primary', scale=1)
    
    model_conditioning_config = model_config["model"].get("conditioning", None)

    has_seconds_start = False
    has_seconds_total = False

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row(visible = has_seconds_start or has_seconds_total):
                # Timing controls
                seconds_start_slider = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="Seconds start", visible=has_seconds_start)
                seconds_total_slider = gr.Slider(minimum=0, maximum=sample_size // sample_rate, step=1, value=10, label="Seconds total", visible=has_seconds_total)
            
            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=10, maximum=500, step=1, value=100, label="Steps")

                # Preview Every slider
                preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every", visible=False)

                # CFG scale 
                cfg_scale_slider = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.5, label="CFG scale")

            with gr.Accordion("Sampler params", open=False):
            
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                # Sampler params
                with gr.Row():
                    # sampler_type_dropdown = gr.Dropdown(["euler", "heun", "dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="heun")
                    sampler_type_dropdown = gr.Dropdown(["euler", "heun"], label="Sampler type", value="euler")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Sigma max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="CFG rescale amount")

            if inpainting: 
                # Inpainting Tab
                with gr.Accordion("Inpainting", open=False):
                    sigma_max_slider.maximum=1.0
                    
                    init_audio_checkbox = gr.Checkbox(label="Do inpainting")
                    init_audio_input = gr.Audio(label="Init audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.75, label="Init audio noise level")

                    mask_cropfrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Crop From %")
                    mask_pastefrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Paste From %")
                    mask_pasteto_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Paste To %")

                    mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=50, label="Mask Start %")
                    mask_maskend_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Mask End %")
                    mask_softnessL_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Left Crossfade Length %")
                    mask_softnessR_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Right Crossfade Length %")
                    mask_marination_slider = gr.Slider(minimum=0.0, maximum=1, step=0.0001, value=0, label="Marination level") # still working on the usefulness of this 

                    inputs = [prompt, 
                        negative_prompt,
                        seconds_start_slider, 
                        seconds_total_slider, 
                        cfg_scale_slider, 
                        steps_slider, 
                        preview_every_slider, 
                        seed_textbox, 
                        sampler_type_dropdown, 
                        sigma_min_slider, 
                        sigma_max_slider,
                        cfg_rescale_slider,
                        init_audio_checkbox,
                        init_audio_input,
                        init_noise_level_slider,
                        mask_cropfrom_slider,
                        mask_pastefrom_slider,
                        mask_pasteto_slider,
                        mask_maskstart_slider,
                        mask_maskend_slider,
                        mask_softnessL_slider,
                        mask_softnessR_slider,
                        mask_marination_slider
                    ]
            else:
                # Default generation tab
                with gr.Accordion("Init audio", open=False):
                    init_audio_checkbox = gr.Checkbox(label="Use init audio")
                    init_audio_input = gr.Audio(label="Init audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.75, label="Init noise level")

                    inputs = [prompt, 
                        negative_prompt,
                        seconds_start_slider, 
                        seconds_total_slider, 
                        cfg_scale_slider, 
                        steps_slider, 
                        preview_every_slider, 
                        seed_textbox, 
                        sampler_type_dropdown, 
                        sigma_min_slider, 
                        sigma_max_slider,
                        cfg_rescale_slider,
                        init_audio_checkbox,
                        init_audio_input,
                        init_noise_level_slider
                    ]

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])
    
    generate_button.click(fn=generate_cond, 
        inputs=inputs,
        outputs=[
            audio_output, 
            audio_spectrogram_output
        ], 
        api_name="generate")

    with gr.Row():
        with gr.Column(scale=6):
            gr.Examples(
                    [
                        [
                            "A hip-hop track using sounds from a construction site—hammering nails as the beat, drilling sounds as scratches, and metal clanks as rhythm accents.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A saxophone that sounds like meowing of cat.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A techno song where all the electronic sounds are generated from kitchen noises—blender whirs, toaster pops, and the sizzle of cooking.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "Dogs barking, birds chirping, and electronic dance music.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "Dog barks a beautiful and fast-paced folk melody while several cats sing chords while meowing.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A time-lapse of a city evolving over a thousand years, represented through shifting musical genres blending seamlessly from ancient to futuristic sounds.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "An underwater city where buildings hum melodies as currents pass through them, accompanied by the distant drumming of bioluminescent sea creatures.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A factory machinery that screams in metallic agony.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A lullaby sung by robotic voices, accompanied by the gentle hum of electric currents and the soft beeping of machines.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A soundscape with a choir of alarm siren from an ambulance car but to produce a lush and calm choir composition with sustained chords.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "The sound of ocean waves where each crash is infused with a musical chord, and the calls of seagulls are transformed into flute melodies.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "Mechanical flowers blooming at dawn, each petal unfolding with a soft chime, orchestrated with the gentle ticking of gears.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "The sound of a meteor shower where each falling star emits a unique musical note, creating a celestial symphony in the night sky.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A clock shop where the ticking and chiming of various timepieces synchronize into a complex polyrhythmic composition.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "An enchanted library where each book opened releases sounds of its story—adventure tales bring drum beats, romances evoke violin strains.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A rainstorm where each raindrop hitting different surfaces produces unique musical pitches, forming an unpredictable symphony.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A carnival where the laughter of children and carousel music intertwine, and the sound of games and rides blend into a festive overture.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A futuristic rainforest where holographic animals emit digital soundscapes, and virtual raindrops produce glitchy electronic rhythms.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "An echo inside a cave where droplets of water create a cascading xylophone melody, and bats' echolocation forms ambient harmonies.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                        [
                            "A steampunk cityscape where steam engines puff in rhythm, and metallic gears turning produce mechanical melodies.", 
                            None, 0, 10, 3.5, 100, 0, -1, "euler", 0, 1, 0, False, None, None
                        ],
                    ],
                    fn=generate_cond,
                    inputs=inputs,
                    outputs=[
                        audio_output, 
                        audio_spectrogram_output
                    ]
                )

def create_txt2audio_ui(model_config):
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_config) 
        # with gr.Tab("Inpainting"):
        #     create_sampling_ui(model_config, inpainting=True)    
    return ui

def create_diffusion_uncond_ui(model_config):
    with gr.Blocks() as ui:
        create_uncond_sampling_ui(model_config)
    
    return ui

def autoencoder_process(audio, latent_noise, n_quantizers):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM, 
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model. 
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    if n_quantizers > 0:
        latents = model.encode_audio(audio, chunked=False, n_quantizers=n_quantizers)
    else:
        latents = model.encode_audio(audio, chunked=False)

    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_autoencoder_ui(model_config):

    is_dac_rvq = "model" in model_config and "bottleneck" in model_config["model"] and model_config["model"]["bottleneck"]["type"] in ["dac_rvq","dac_rvq_vae"]

    if is_dac_rvq:
        n_quantizers = model_config["model"]["bottleneck"]["config"]["n_codebooks"]
    else:
        n_quantizers = 0

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        n_quantizers_slider = gr.Slider(minimum=1, maximum=n_quantizers, step=1, value=n_quantizers, label="# quantizers", visible=is_dac_rvq)
        latent_noise_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.001, value=0.0, label="Add latent noise")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=autoencoder_process, inputs=[input_audio, latent_noise_slider, n_quantizers_slider], outputs=output_audio, api_name="process")

    return ui

def diffusion_prior_process(audio, steps, sampler_type, sigma_min, sigma_max):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    audio = audio.unsqueeze(0)

    audio = model.stereoize(audio, in_sr, steps, sampler_kwargs={"sampler_type": sampler_type, "sigma_min": sigma_min, "sigma_max": sigma_max})

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_diffusion_prior_ui(model_config):
    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        # Sampler params
        with gr.Row():
            steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
            sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
            sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
            sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=diffusion_prior_process, inputs=[input_audio, steps_slider, sampler_type_dropdown, sigma_min_slider, sigma_max_slider], outputs=output_audio, api_name="process")    

    return ui

def create_lm_ui(model_config):
    with gr.Blocks() as ui:
        output_audio = gr.Audio(label="Output audio", interactive=False)
        audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)

        # Sampling params
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0, maximum=5, step=0.01, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95, label="Top p")
            top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top k")

        generate_button = gr.Button("Generate", variant='primary', scale=1)
        generate_button.click(
            fn=generate_lm, 
            inputs=[
                temperature_slider, 
                top_p_slider, 
                top_k_slider
            ], 
            outputs=[output_audio, audio_spectrogram_output],
            api_name="generate"
        )

    return ui

def create_ui(ckpt_path=None, model_half=False):
    global model  # access the global model variable
    
    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print(f"Loading model from {ckpt_path}")
    model, model_config = get_local_pretrained_model(ckpt_path)
    
    model = model.to(device).eval().requires_grad_(False)
    
    if model_half:
        print("Converting model to half precision")
        model = model.half()
    
    model_type = model_config["model_type"]

    if model_type == "diffusion_cond":
        ui = create_txt2audio_ui(model_config)
    elif model_type == "diffusion_uncond":
        ui = create_diffusion_uncond_ui(model_config)
    elif model_type == "autoencoder" or model_type == "diffusion_autoencoder":
        ui = create_autoencoder_ui(model_config)
    elif model_type == "diffusion_prior":
        ui = create_diffusion_prior_ui(model_config)
    elif model_type == "lm":
        ui = create_lm_ui(model_config)
        
    return ui
