# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.
# modified from stable-audio-tools under the MIT license

def get_custom_metadata(dataset, audio, info):
    # Use 'captions' metadata to prompt without modification
    info.update({
        "prompt": info['metadata']['captions'],
        "prompt_global": info['metadata']['captions']
        })
    
    return audio, info