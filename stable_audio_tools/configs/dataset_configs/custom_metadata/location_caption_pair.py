def get_custom_metadata(dataset, audio, info):
    # Use 'captions' metadata to prompt without modification
    info.update({
        "prompt": info['metadata']['captions'],
        "prompt_global": info['metadata']['captions']
        })
    
    return audio, info