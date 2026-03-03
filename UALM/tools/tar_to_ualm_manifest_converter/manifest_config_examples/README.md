# YAML Explained


## Text-only

The text-only config is simple: see ```config_text_only_example.yaml``` as a quick example. You just need to provide a list of text subset names and their ```jsonl``` paths. Each row of the ```jsonl``` file looks like this:

```{"input": [{"role": "user", "content": "Question"}], "output": "Output"}```

## Text-to-Audio and Audio Understanding

Manifests for audio generation and understanding are a little bit tricky. Once you finish data prep step 2 (all tar and manifest parts are created), paste the paths (```base_manifest_dir``` for the manifest path and ```root_audio_dir``` for the tar path, which could be identical if everything is put in the same path) to 
```config_xxx.yaml```. Then, run the step 3 command

```
python tools/tar_to_ualm_manifest_converter/convert_tar_to_ualm_manifest.py \
    --config tools/tar_to_ualm_manifest_converter/manifest_config_examples/config_NAME.yaml \
    --output-dir .tmp/manifest_NAME
```

This will create the ```.tmp/manifest_NAME``` folder that you need to refer to in your training script. 

Note that you should put all tasks into the same ```config_NAME.yaml``` if you want to train a unified model; the dataloader will pick the correct processing method automatically.

Note that all datasets are default to ```weight=1.0``` as in ```scripts/train.py (L274)``` but you are welcome to pass flexible weights to ```specifiers``` yourself.


## Visualization
You could also run the ```try_dataloader.py``` script to print a few batches of the data. 