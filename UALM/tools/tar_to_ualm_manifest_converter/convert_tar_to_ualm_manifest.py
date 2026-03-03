# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
"""
Production-grade conversion tool for UALM Tarball Manifests.

This script converts sharded NDJSON manifests (from object_storage_manifest) into the 
UALM training format. It performs the following key functions:

1.  **Length Estimation**: Estimates total token count (Audio + Text) for bucket sorting.
2.  **Metadata Unification**: Consolidates all datasets into a single random-access index 
    (LMDB or JSONL) to minimize file handle overhead during training.
3.  **Task Injection**: Injects task-specific metadata (e.g., 'text_to_audio') to guide 
    the runtime data loader's chat template construction.
4.  **Audio Segmentation**: For audio-only datasets, splits long audio into fixed-duration
    non-overlapping segments for deterministic LM training.
5.  **Weight Handling**: Calculates and outputs suggested training arguments for 
    weighted sampling via `DataIteratorFactory`.

Architecture:
    - `Config`: Handles configuration parsing and validation.
    - `ManifestProcessor`: Manages parallel processing of source files.
    - `MetadataBuilder`: Standardizes and cleans raw entries into UALM metadata.
    - `OutputWriter`: Handles writing artifacts (Manifests, Stats, LMDB/JSONL).

Usage:
    python convert_tar_to_ualm_manifest.py --config config.yaml --output-dir exp/manifest
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
import lmdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Structures & Configuration
# ==============================================================================

@dataclass
class ProcessingOptions:
    root_audio_dir: Optional[str] = None
    location_keyname: str = "audio"
    caption_keyname: str = "text"
    conversation_keyname: str = "conversations"
    ualm_task: str = "unknown_task"
    data_weight: float = 1.0
    # Segmentation options for audio_only task
    segment_duration_max: Optional[float] = None
    segment_duration_min: float = 1.0
    
    def __init__(self, **kwargs):
        self.root_audio_dir = kwargs.get("root_audio_dir")
        self.location_keyname = kwargs.get("location_keyname", "audio")
        self.caption_keyname = kwargs.get("caption_keyname", "text")
        self.conversation_keyname = kwargs.get("conversation_keyname", "conversations")
        self.ualm_task = kwargs.get("ualm_task", "unknown_task")
        self.data_weight = kwargs.get("data_weight", 1.0)
        # Segmentation
        self.segment_duration_max = kwargs.get("segment_duration_max")
        self.segment_duration_min = kwargs.get("segment_duration_min", 3.0)

@dataclass
class DatasetConfig:
    """Configuration for a dataset source."""
    id: str
    base_manifest_dir: Path
    splits: List[str]
    options: ProcessingOptions

@dataclass
class GlobalConfig:
    """Global script configuration."""
    text_tokenizer: str
    audio_frame_rate: float
    seed: int
    num_workers: int
    disable_lmdb: bool
    output_dir: Path
    datasets: List[DatasetConfig]

    @classmethod
    def from_args(cls, args):
        with open(args.config, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        datasets = []
        if "manifests" in raw_config and "train" in raw_config["manifests"]:
            for item in raw_config["manifests"]["train"]:
                opts = ProcessingOptions(**item.get("processing_options", {}))
                
                # Validate: segment_duration_max is only allowed for audio_only task
                if opts.segment_duration_max is not None and opts.ualm_task != "audio_only":
                    raise ValueError(
                        f"Dataset '{item['id']}': Non-overlapping audio segmentation "
                        f"(segment_duration_max={opts.segment_duration_max}) is only supported "
                        f"for 'audio_only' task, but got ualm_task='{opts.ualm_task}'. "
                        f"Remove 'segment_duration_max' from processing_options or set ualm_task to 'audio_only'."
                    )
                
                src = item.get("manifest_source", {})
                ds = DatasetConfig(
                    id=item["id"],
                    base_manifest_dir=Path(src["base_manifest_dir"]),
                    splits=src["splits"],
                    options=opts
                )
                datasets.append(ds)
        
        return cls(
            text_tokenizer=args.text_tokenizer,
            audio_frame_rate=args.audio_frame_rate,
            seed=args.seed,
            num_workers=args.num_workers,
            disable_lmdb=args.disable_lmdb,
            output_dir=args.output_dir,
            datasets=datasets
        )


# ==============================================================================
# Metadata Processing Logic (Worker)
# ==============================================================================

class TokenizerWrapper:
    """Thread-safe wrapper for tokenizer."""
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class MetadataBuilder:
    """Encapsulates logic for cleaning and normalizing metadata entries."""
    
    @staticmethod
    def calculate_length(entry: Dict, tokenizer: TokenizerWrapper, 
                        options: ProcessingOptions, audio_fps: float,
                        override_duration: Optional[float] = None) -> int:
        """
        Estimates total sequence length (tokens).
        
        Args:
            override_duration: If provided, use this duration instead of the one in entry.
                               Used for segmented audio-only samples.
        """
        total_tokens = 1  # <bos>
        
        # Extract conversations
        conversations = entry.get(options.conversation_keyname)
        if not conversations:
            caption = entry.get(options.caption_keyname)
            if caption:
                conversations = [
                    {"from": "human", "value": "<sound>"},
                    {"from": "gpt", "value": caption}
                ]
            else:
                conversations = [{"from": "human", "value": "<sound>"}]

        # Calculate length
        for msg in conversations:
            content = msg.get("value", "")
            total_tokens += 3  # Overhead (<role> <modality> <eos>)
            
            parts = content.split("<sound>")
            for i, part in enumerate(parts):
                if part.strip():
                    total_tokens += tokenizer.count_tokens(part)
                
                # Audio part (if split indicates <sound> was present)
                if i < len(parts) - 1:
                    if override_duration is not None:
                        total_tokens += int(override_duration * audio_fps)
                    else:
                        audio_meta = entry.get(options.location_keyname)
                        if audio_meta and "duration" in audio_meta:
                            total_tokens += int(audio_meta["duration"] * audio_fps)
        
        return total_tokens

    @staticmethod
    def build_unified_entry(sid: str, length: int, raw_entry: Dict, 
                          options: ProcessingOptions,
                          segment_offset: Optional[float] = None,
                          segment_duration: Optional[float] = None) -> Tuple[str, int, Dict]:
        """
        Constructs the standardized metadata entry.
        
        Args:
            segment_offset: If provided, overrides the audio offset (for segmented samples).
            segment_duration: If provided, overrides the audio duration (for segmented samples).
        
        Returns: (sample_id, length, metadata_dict)
        """
        # 1. Audio Metadata
        audio_info = raw_entry.get(options.location_keyname, {})
        
        # Resolve absolute path
        tar_path = audio_info.get("tar_path")
        if options.root_audio_dir and tar_path and not os.path.isabs(tar_path):
            tar_path = os.path.join(options.root_audio_dir, tar_path)

        # Handle segment offset: combine with original file offset
        base_offset = audio_info.get("offset", 0.0) or 0.0
        final_offset = base_offset + (segment_offset if segment_offset is not None else 0.0)
        final_duration = segment_duration if segment_duration is not None else audio_info.get("duration")

        audio_meta = {
            "tar_path": tar_path,
            "tar_offset": audio_info.get("tar_offset"),
            "tar_size": audio_info.get("tar_size"),
            "offset": final_offset,
            "duration": final_duration,
            "sampling_rate": audio_info.get("sampling_rate"),
            "channels": audio_info.get("channels"),
            "encoding": audio_info.get("encoding"),
            "bytes_per_sample": audio_info.get("bytes_per_sample"),
            "data_offset": audio_info.get("data_offset"),
        }

        # 2. Text Content (Sanitized)
        text_content = {
            "conversations": raw_entry.get(options.conversation_keyname),
            "messages": raw_entry.get("messages"),
            "text": raw_entry.get(options.caption_keyname),
            "caption": raw_entry.get("caption"),
        }
        # Remove empty fields
        text_content = {k: v for k, v in text_content.items() if v is not None}

        # 3. Unified Entry
        metadata = {
            "id": sid,
            "ualm_task": options.ualm_task,
            "audio": audio_meta,
            "text": text_content
        }
        
        return sid, length, metadata


def generate_segments(entry: Dict, options: ProcessingOptions) -> List[Tuple[float, float, int]]:
    """
    Generate non-overlapping segment windows for an audio entry.
    
    Returns:
        List of (segment_offset, segment_duration, segment_index) tuples.
        Returns empty list if segmentation is disabled or not applicable.
    """
    # Check if segmentation should be applied
    if options.segment_duration_max is None or options.segment_duration_max <= 0:
        return []
    
    if options.ualm_task != "audio_only":
        return []
    
    audio_info = entry.get(options.location_keyname, {})
    total_duration = audio_info.get("duration", 0.0)
    
    if total_duration <= 0:
        return []
    
    segments = []
    seg_max = options.segment_duration_max
    seg_min = options.segment_duration_min
    
    num_full_segments = int(total_duration // seg_max)
    
    # Add full segments
    for i in range(num_full_segments):
        offset = i * seg_max
        segments.append((offset, seg_max, i))
    
    # Handle remainder (tail segment)
    remainder = total_duration - (num_full_segments * seg_max)
    if remainder >= seg_min:
        offset = num_full_segments * seg_max
        segments.append((offset, remainder, num_full_segments))
    
    return segments


_worker_tokenizer = None

def _init_worker(tokenizer_name):
    """Initializer function for each worker process - called once per worker."""
    global _worker_tokenizer
    _worker_tokenizer = TokenizerWrapper(tokenizer_name)


def process_file_worker(args):
    """Multiprocessing worker function."""
    global _worker_tokenizer
    file_path, options, tokenizer_name, audio_fps, dataset_id = args
    
    # tokenizer = TokenizerWrapper(tokenizer_name)
    tokenizer = _worker_tokenizer
    results = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    sid = entry.get("audio_id")
                    if not sid: continue
                    
                    # Check if segmentation applies
                    segments = generate_segments(entry, options)
                    
                    if segments:
                        # Segmented audio-only: create one sample per segment
                        for seg_offset, seg_duration, seg_idx in segments:
                            # Segment ID: dataset_id::original_id::seg_N
                            seg_id = f"{dataset_id}::{sid}::seg_{seg_idx}"
                            
                            # Calculate length for this segment
                            length = MetadataBuilder.calculate_length(
                                entry, tokenizer, options, audio_fps,
                                override_duration=seg_duration
                            )
                            
                            # Build metadata with segment-specific offset/duration
                            result = MetadataBuilder.build_unified_entry(
                                seg_id, length, entry, options,
                                segment_offset=seg_offset,
                                segment_duration=seg_duration
                            )
                            results.append(result)
                    else:
                        # Non-segmented: standard processing
                        global_id = f"{dataset_id}::{sid}"
                        
                        length = MetadataBuilder.calculate_length(
                            entry, tokenizer, options, audio_fps
                        )
                        
                        result = MetadataBuilder.build_unified_entry(
                            global_id, length, entry, options
                        )
                        results.append(result)
                    
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []
        
    return results


# ==============================================================================
# Output Management
# ==============================================================================

class OutputWriter:
    """Handles writing of all output artifacts."""
    
    def __init__(self, output_dir: Path, use_lmdb: bool):
        self.output_dir = output_dir
        self.stats_dir = output_dir / "stats"
        self.use_lmdb = use_lmdb
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)
        
        self.meta_filename = "_metadata.lmdb" if use_lmdb else "_metadata.jsonl"

    def save_manifest(self, dataset_id: str, sample_ids: List[str]):
        """Saves the dataset-specific manifest JSON."""
        manifest_path = self.output_dir / f"{dataset_id}_manifest.json"
        
        data = {
            "data_entry": [
                {
                    "name": "audio",
                    "path": str(self.output_dir / self.meta_filename),
                    "reader": "tarball_audio_byteseek"
                },
                {
                    "name": "text",
                    "path": str(self.output_dir / self.meta_filename),
                    "reader": "tarball_dialogue"
                }
            ],
            "samples": sample_ids
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(data, f, indent=2)
        return manifest_path

    def save_stats(self, task: str, dataset_id: str, stats: Dict[str, int]):
        """Saves length statistics for bucketing."""
        path = self.stats_dir / f"stats_{task}_{dataset_id}.jsonl"
        with open(path, 'w') as f:
            for sid, length in stats.items():
                f.write(json.dumps({sid: length}) + "\n")

    def save_unified_metadata(self, all_metadata: Dict[str, Dict]):
        """Saves the unified random-access index."""
        # 1. JSONL (Always save for inspection)
        jsonl_path = self.output_dir / "_metadata.jsonl"
        logger.info(f"Saving unified metadata JSONL to {jsonl_path}...")
        with open(jsonl_path, 'w') as f:
            for entry in all_metadata.values():
                f.write(json.dumps(entry) + "\n")
        
        # 2. LMDB (Optional but recommended)
        if self.use_lmdb:
            lmdb_path = self.output_dir / "_metadata.lmdb"
            logger.info(f"Saving unified metadata LMDB to {lmdb_path}...")
            
            # Estimate size: 1TB max (sparse)
            map_size = 1024 * 1024 * 1024 * 1024
            env = lmdb.open(str(lmdb_path), map_size=map_size)
            
            with env.begin(write=True) as txn:
                for sid, entry in tqdm(all_metadata.items(), desc="Writing LMDB"):
                    txn.put(sid.encode('utf-8'), pickle.dumps(entry))
            logger.info("LMDB created successfully.")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

class ManifestProcessor:
    """Main processor class."""
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.writer = OutputWriter(config.output_dir, not config.disable_lmdb)

    def _print_banner(self):
        """Print startup banner explaining the tool."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    UALM MANIFEST CONVERSION TOOL                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This tool converts sharded NDJSON manifests into the UALM training format.  ║
║                                                                              ║
║  What it does:                                                               ║
║    1. Reads raw audio metadata from source NDJSON manifest files             ║
║    2. Estimates sequence lengths (audio + text tokens) for bucket sorting    ║
║    3. Unifies all datasets into a single LMDB/JSONL index for fast lookup    ║
║    4. Injects task labels (e.g., 'audio_only', 'caption_to_audio')           ║
║    5. Optionally segments long audio into fixed-duration chunks              ║
║    6. Generates per-dataset manifest JSONs for training                      ║
║                                                                              ║
║  Output files:                                                               ║
║    - <dataset_id>_manifest.json : Per-dataset sample index for DataLoader    ║
║    - _metadata.jsonl / .lmdb    : Unified metadata store (all datasets)      ║
║    - stats/<task>_<id>.jsonl    : Length stats for bucket-based batching     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)

    def _print_config_plan(self, work_items_by_dataset: Dict[str, List]):
        """Print configuration and processing plan."""
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        print(f"  Output Directory   : {self.config.output_dir}")
        print(f"  Text Tokenizer     : {self.config.text_tokenizer}")
        print(f"  Audio Frame Rate   : {self.config.audio_frame_rate} tokens/sec")
        print(f"  Parallel Workers   : {self.config.num_workers}")
        print(f"  LMDB Output        : {'Enabled' if not self.config.disable_lmdb else 'Disabled (JSONL only)'}")
        
        print("\n" + "-"*80)
        print("DATASETS TO PROCESS")
        print("-"*80)
        
        for ds_config in self.config.datasets:
            ds_id = ds_config.id
            num_files = len(work_items_by_dataset.get(ds_id, []))
            
            print(f"\n  [{ds_id}]")
            print(f"    • Task Type      : {ds_config.options.ualm_task}")
            print(f"    • Source Dir     : {ds_config.base_manifest_dir}")
            print(f"    • Splits         : {', '.join(ds_config.splits)}")
            print(f"    • Source Files   : {num_files} NDJSON file(s) found")
            print(f"    • Data Weight    : {ds_config.options.data_weight}")
            
            if ds_config.options.segment_duration_max:
                print(f"    • Segmentation   : ENABLED (non-overlapping)")
                print(f"        - Max segment duration : {ds_config.options.segment_duration_max}s")
                print(f"        - Min segment duration : {ds_config.options.segment_duration_min}s")
                print(f"        - Note: Long audio will be split into fixed-duration chunks.")
                print(f"                Trailing segments shorter than min will be discarded.")
            else:
                print(f"    • Segmentation   : Disabled (whole audio per sample)")
        
        print("\n" + "="*80)

    def _print_phase(self, phase_num: int, title: str, description: str):
        """Print phase header with description."""
        print(f"\n{'─'*80}")
        print(f"PHASE {phase_num}: {title}")
        print(f"{'─'*80}")
        print(f"  {description}")
        print()

    def _print_dataset_summary(self, ds_id: str, ds_config: DatasetConfig, 
                               num_samples: int, manifest_path: Path):
        """Print per-dataset processing summary."""
        if ds_config.options.segment_duration_max:
            sample_type = "segments"
            extra_info = f" (segmented from original audio, max={ds_config.options.segment_duration_max}s)"
        else:
            sample_type = "samples"
            extra_info = ""
        
        print(f"    ✓ {ds_id}")
        print(f"        Task     : {ds_config.options.ualm_task}")
        print(f"        Samples  : {num_samples:,} {sample_type}{extra_info}")
        print(f"        Manifest : {manifest_path}")

    def _print_final_summary(self, total_samples: int, num_datasets: int):
        """Print final summary of outputs."""
        print("\n" + "="*80)
        print("CONVERSION COMPLETE")
        print("="*80)
        print(f"  Total Samples Indexed : {total_samples:,}")
        print(f"  Datasets Processed    : {num_datasets}")
        print(f"\n  Output Files Created:")
        print(f"    • Manifests      : {self.config.output_dir}/<dataset_id>_manifest.json")
        print(f"    • Unified Index  : {self.config.output_dir}/_metadata.{'lmdb' if not self.config.disable_lmdb else 'jsonl'}")
        print(f"    • Length Stats   : {self.writer.stats_dir}/stats_<task>_<dataset_id>.jsonl")
        print("="*80)

    def run(self):
        # Print startup banner
        self._print_banner()
        
        # Phase 1: Collect work items
        self._print_phase(1, "DISCOVERY", 
            "Scanning source directories for NDJSON manifest files...")
        
        work_items = self._collect_work_items()
        
        # Group work items by dataset for summary
        work_items_by_dataset = {ds.id: [] for ds in self.config.datasets}
        for item in work_items:
            ds_id = item[4]
            work_items_by_dataset[ds_id].append(item)
        
        # Print configuration plan
        self._print_config_plan(work_items_by_dataset)
        
        logger.info(f"Discovery complete: {len(work_items)} source files across {len(self.config.datasets)} datasets.")
        
        # Phase 2: Parallel processing
        self._print_phase(2, "PROCESSING", 
            "Reading source manifests, estimating token lengths, and building unified entries.\n"
            "  For each sample:\n"
            "    - Extract audio metadata (path, duration, sample rate, etc.)\n"
            "    - Estimate sequence length = audio_tokens + text_tokens\n"
            "    - Apply segmentation if enabled (audio_only task with segment_duration_max)\n"
            "    - Assign global sample ID: <dataset_id>::<original_id>[::seg_N]")
        
        dataset_results = self._process_parallel(work_items)
        
        # Phase 3: Aggregate and write
        self._print_phase(3, "OUTPUT GENERATION", 
            "Writing per-dataset manifests, length statistics, and unified metadata index.")
        
        all_metadata = {}
        suggested_args = []
        processed_datasets = 0
        
        print("  Processing datasets:")
        for ds_config in self.config.datasets:
            ds_id = ds_config.id
            samples = dataset_results.get(ds_id, [])
            
            if not samples:
                logger.warning(f"No samples found for dataset {ds_id}")
                continue
            
            processed_datasets += 1
            
            # Split results
            sample_ids = []
            stats = {}
            
            for sid, length, meta in samples:
                sample_ids.append(sid)
                stats[sid] = length
                if sid not in all_metadata:
                    all_metadata[sid] = meta
                else:
                    logger.warning(f"Duplicate sample ID found: {sid}")
            
            # Write Manifest and Stats
            manifest_path = self.writer.save_manifest(ds_id, sample_ids)
            self.writer.save_stats(ds_config.options.ualm_task, ds_id, stats)
            
            # Print dataset summary
            self._print_dataset_summary(ds_id, ds_config, len(samples), manifest_path)
            
            # Generate CLI suggestion
            arg = f"{ds_config.options.ualm_task}:{ds_id}:{manifest_path}:{ds_config.options.data_weight}"
            suggested_args.append(arg)
        
        # Phase 4: Write unified metadata
        print(f"\n  Writing unified metadata index ({len(all_metadata):,} total entries)...")
        self.writer.save_unified_metadata(all_metadata)
        
        # Print final summary
        self._print_final_summary(len(all_metadata), processed_datasets)
        
        # Print training suggestions
        self._print_suggestions(suggested_args)

    def _collect_work_items(self):
        """Scans directories for input files."""
        items = []
        for ds in self.config.datasets:
            files = []
            for split in ds.splits:
                # Pattern matching for sharded files
                split_files = list(ds.base_manifest_dir.glob(f"{split}*.ndjson"))
                if not split_files:
                    split_files = list(ds.base_manifest_dir.glob(f"**/{split}*.ndjson"))
                files.extend(split_files)
            
            for f in files:
                items.append((
                    str(f), 
                    ds.options, 
                    self.config.text_tokenizer,
                    self.config.audio_frame_rate,
                    ds.id
                ))
        return items

    def _process_parallel(self, work_items):
        """Runs workers and groups results by dataset ID."""
        results = {ds.id: [] for ds in self.config.datasets}
        
        # with Pool(self.config.num_workers) as pool:
        with Pool(
            self.config.num_workers,
            initializer=_init_worker,
            initargs=(self.config.text_tokenizer,)
        ) as pool:
            for ds_id, batch_results in tqdm(
                pool.imap_unordered(self._worker_wrapper, work_items), 
                total=len(work_items),
                desc="Processing Files"
            ):
                results[ds_id].extend(batch_results)
        return results

    @staticmethod
    def _worker_wrapper(args):
        """Static wrapper to be picklable."""
        ds_id = args[4]
        res = process_file_worker(args)
        return ds_id, res

    def _print_suggestions(self, args_list):
        print("\n" + "="*80)
        print("SUGGESTED TRAINING ARGUMENTS:")
        print("="*80)
        print(f"--train-unregistered-specifier \"{' '.join(args_list)}\"")
        print(f"--stats-dir {self.writer.stats_dir}")
        print("="*80 + "\n")


def main():
    parser = get_parser()
    args = parser.parse_args()

    config = GlobalConfig.from_args(args)
    processor = ManifestProcessor(config)
    processor.run()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert tarball manifests to UALM format with optional segmentation."
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for manifests and metadata")
    parser.add_argument("--text-tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace tokenizer for text length estimation")
    parser.add_argument("--audio-frame-rate", type=float, default=50.0,
                        help="Audio codec frame rate (tokens/sec) for length estimation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--disable-lmdb", action="store_true",
                        help="Disable LMDB output (JSONL only)")
    return parser


if __name__ == "__main__":
    main()
