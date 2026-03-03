# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.


"""
python tools/tar_to_ualm_manifest_converter/convert_tar_to_ualm_manifest.py \
    --config tools/tar_to_ualm_manifest_converter/manifest_config_examples/config_NAME.yaml \
    --output-dir .tmp/manifest_NAME
"""


import os
import sys
import json
import yaml
import random
import numpy as np
import soundfile as sf
import IPython.display as ipd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

# Add project root to path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)


# Configuration
OUTPUT_DIR = Path(".tmp/manifest_NAME")

# Check output
print(f"Checking files in {OUTPUT_DIR}...")

manifests = list(OUTPUT_DIR.glob("*_manifest.json"))
stats_files = list((OUTPUT_DIR / "stats").glob("*.jsonl"))
metadata_lmdb = OUTPUT_DIR / "_metadata.lmdb"

print(f"Found {len(manifests)} manifests:")
for m in manifests: print(f"  - {m.name}")

print(f"\nFound {len(stats_files)} stats files:")
for s in stats_files: print(f"  - {s.name}")

if metadata_lmdb.exists():
    print(f"\n✅ Unified Metadata LMDB found: {metadata_lmdb}")
else:
    print(f"\n❌ Unified Metadata LMDB MISSING!")

from dataloader.multimodal_loader.tarball_reader import TarballAudioReader, TarballDialogueReader

# Initialize Readers (pointing to the shared metadata)
audio_reader = TarballAudioReader(str(metadata_lmdb))
text_reader = TarballDialogueReader(str(metadata_lmdb))

for manifest_path in manifests:
    with open(manifest_path, 'r') as f:
        m_data = json.load(f)
        
    dataset_id = manifest_path.stem.replace("_manifest", "")
    sample_id = random.choice(m_data["samples"]) # Pick random sample
    
    print(f"\n🔎 Inspecting Dataset: {dataset_id}")
    print(f"   Sample ID: {sample_id}")
    
    # 1. Check Dialogue Structure (Task Template Verification)
    dialogue = text_reader[sample_id]
    print("   💬 Dialogue Structure:")
    for role, modality, content in dialogue:
        preview = content[:60] + "..." if len(content) > 60 else content
        print(f"      [{role.upper()}] ({modality}): {preview}")

    # 2. Check Audio Loading
    try:
        audio, sr = audio_reader[sample_id]
        duration = audio.shape[1] / sr
        print(f"   🎵 Audio: {audio.shape} @ {sr}Hz ({duration:.2f}s)")
        # display(ipd.Audio(audio[0], rate=sr))
        # sf.write(f".tmp/audio_{sample_id}.wav", audio[0], sr)

        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, audio.shape[1]/sr, audio.shape[1]), audio[0])
        plt.title(f"Waveform: {sample_id}")
        # plt.show()
        # plt.savefig(f".tmp/waveform_{sample_id}.png")
        
    except Exception as e:
        print(f"   ❌ Audio Load Error: {e}")


from dataloader.iterator import DataIteratorFactory

# Setup Registry for CombinedDataset
registry_path = OUTPUT_DIR / "dataset_registry.yaml"
registry_content = {}

for m in manifests:
    ds_id = m.stem.replace("_manifest", "")
    registry_content[ds_id] = {"path": str(m.resolve())}

with open(registry_path, "w") as f:
    yaml.dump(registry_content, f)

os.environ["ESPNET_DATASET_REGISTRY"] = str(registry_path)
print(f"\nCreated registry at {registry_path}")

# Construct Specifier String for ALL datasets
specifiers = []

for s_file in stats_files:
    # Filename: stats_{task}_{id}.jsonl
    fname = s_file.stem # stats_...
    
    # Find which dataset ID is at the end of this string
    matched_ds_id = None
    for m in manifests:
        ds_id = m.stem.replace("_manifest", "")
        if fname.endswith(ds_id):
            matched_ds_id = ds_id
            break
            
    if matched_ds_id:
        # Extract task: stats_{task}_{id} -> remove prefix and suffix
        task_part = fname[6 : -len(matched_ds_id)-1] # stats_..._ID
        specifiers.append(f"{task_part}:{matched_ds_id}:1.0")

full_specifier = " ".join(specifiers)
# print(f"\nInitializing Factory with specifier:\n  {full_specifier}")

try:
    factory = DataIteratorFactory(
        registered_specifier=full_specifier,
        stats_dir=OUTPUT_DIR / "stats",
        batch_size=5000,
        batchfy_method="bucket",
        shuffle=True,
        num_workers=6,
        seed=42,
        collate_fn=lambda x: x # Bypass tensor stacking
    )
    
    loader = factory.build_iter(global_step=0, length=20)
    print("\n✅ Factory Initialized. Iterating batches...")
    
    for i, batch in enumerate(loader):
        print(f"  Batch {i+1}: {len(batch)} samples")
        # Verify content of several samples in the batch
        batch_length = len(batch)
        # for sample_id in [0, 10, int(batch_length//2), batch_length-10, batch_length-1]:
        for sample_id in range(batch_length):
            sample = batch[sample_id]
            if isinstance(sample, (list, tuple)):
                sample = sample[1] # Unpack data dict
            
            print(batch[sample_id])
            print(f"    Key: {batch[sample_id][0]}")
            print(f"    Audio: {sample['audio'][0].shape} (loaded)")

            
            
        
except Exception as e:
    print(f"\n❌ Factory Error: {e}")
    import traceback
    traceback.print_exc()