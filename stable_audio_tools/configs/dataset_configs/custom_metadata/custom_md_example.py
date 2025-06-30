# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.
# modified from stable-audio-tools under the MIT license

def get_custom_metadata(dataset, audio, info):
    # Use relative path as the prompt.
    # NOTE: this is mostly for debugging purposes unless relpath contains valid metadata to be used as text prompt.
    info.update({
        "prompt": info["relpath"],
        })
    
    return audio, info