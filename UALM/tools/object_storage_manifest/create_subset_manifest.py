# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
"""
Production-grade script to create a subset manifest from existing sharded manifests.
Preserves original shard filenames and tarball references while allowing metadata updates.

Example Usage:
    python scripts/object_storage_manifest/create_subset_manifest.py \
        --input_subset_manifest /path/to/filter_manifest.jsonl \
        --source_manifest_dir /path/to/existing_sharded_manifests_dir/ \
        --output_dir /path/to/output/subset_manifests/ \
        --prefix subset_train \
        --dataset_name my_subset_v1 \
        --update_metadata

Features:
- Filters existing sharded manifests based on a provided subset (raw manifest).
- Preserves tarball byte-seek metadata (tar_path, offset, size) crucial for object storage loaders.
- Optionally updates captions and metadata from the input filter.
- Generates a new index file reflecting the subset statistics.
- Allows explicit dataset name override to track filtering strategies.
"""

import argparse
import json
import logging
import os
import sys
import glob
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from collections import Counter, defaultdict
from dataclasses import dataclass

# Dynamic import to handle running from different contexts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from create_manifest import ManifestEntry, ManifestWriter
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'scripts', 'object_storage_manifest'))
    from create_manifest import ManifestEntry, ManifestWriter

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FilterLoader:
    """Handles loading and indexing of the subset/filter manifest."""

    @staticmethod
    def load(input_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load filter entries from a file or directory.
        
        Args:
            input_path: Path to .json, .jsonl file or directory containing them.
            
        Returns:
            Dictionary mapping audio filename (basename) to a list of raw manifest entries.
        """
        logger.info(f"Loading filter from {input_path}")
        target_entries = defaultdict(list)
        
        files = FilterLoader._resolve_files(input_path)
        if not files:
            logger.error(f"No input files found in {input_path}")
            return {}
            
        for file_path in tqdm(files, desc="Loading filter files"):
            try:
                entries = FilterLoader._read_file(file_path)
                for entry in entries:
                    key = FilterLoader._extract_key(entry)
                    if key:
                        target_entries[key].append(entry)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                
        logger.info(f"Loaded filter: {sum(len(v) for v in target_entries.values())} entries covering {len(target_entries)} unique files")
        return target_entries

    @staticmethod
    def _resolve_files(input_path: str) -> List[str]:
        if os.path.isfile(input_path):
            return [input_path]
        elif os.path.isdir(input_path):
            files = []
            files.extend(glob.glob(os.path.join(input_path, "**/*.jsonl"), recursive=True))
            files.extend(glob.glob(os.path.join(input_path, "**/*.json"), recursive=True))
            return files
        return []

    @staticmethod
    def _read_file(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                return json.load(f)
            else:
                return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _extract_key(entry: Dict[str, Any]) -> Optional[str]:
        """Extract the filename key for matching.
        
        For multi-audio entries, uses the first audio path as the key.
        """
        # Priority: location -> audio.path (handles both single and multi-audio)
        location = entry.get('location')
        
        # If location is a list (multi-audio input), take first element
        if isinstance(location, list):
            location = location[0] if location else None
        
        if not location and 'audio' in entry:
            audio = entry['audio']
            # Handle multi-audio: audio is list
            if isinstance(audio, list) and audio:
                location = audio[0].get('path', '')
            elif isinstance(audio, dict):
                location = audio.get('path', '')
        
        if location:
            return Path(location).name
        return None


class ShardProcessor:
    """Processes individual shard files to create subset shards."""

    def __init__(self, output_dir: str, update_metadata: bool, dataset_name: Optional[str] = None):
        self.output_dir = output_dir
        self.update_metadata = update_metadata
        self.dataset_name = dataset_name

    def process(self, shard_path: str, filter_data: Dict[str, List[Dict]]) -> Tuple[Optional[Dict], List[ManifestEntry]]:
        """
        Process a single shard: match entries against filter and write output.
        
        Args:
            shard_path: Path to source .ndjson shard.
            filter_data: The loaded filter dictionary.
            
        Returns:
            Tuple of (shard_metadata, list_of_entries). metadata is None if empty.
        """
        shard_name = os.path.basename(shard_path)
        # logger.info(f"Processing shard: {shard_name}")
        
        # 1. Read Source Entries
        source_audio_map = self._load_source_audio_map(shard_path)
        
        # 2. Generate Subset Entries
        final_entries = self._generate_subset_entries(source_audio_map, filter_data)
        
        if not final_entries:
            return None, []

        # 3. Write Output Shard
        output_path = os.path.join(self.output_dir, shard_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        shard_meta = self._write_shard(output_path, final_entries, shard_name)
        return shard_meta, final_entries

    def _load_source_audio_map(self, shard_path: str) -> Dict[str, Dict[str, Any]]:
        """Map filename -> source entry (or audio dict) for the shard.
        
        For multi-audio entries, uses the first audio path as the key.
        """
        audio_map = {}
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    entry = json.loads(line)
                    
                    # We map by filename to match the filter keys
                    # Handle both single-audio (dict) and multi-audio (list)
                    audio = entry.get('audio', {})
                    if isinstance(audio, list) and audio:
                        # Multi-audio: use first audio path as key
                        audio_path = audio[0].get('path', '')
                    elif isinstance(audio, dict):
                        # Single-audio
                        audio_path = audio.get('path', '')
                    else:
                        audio_path = ''
                    
                    if audio_path:
                        key = Path(audio_path).name
                        # We store the full entry to access 'audio' or other original fields
                        audio_map[key] = entry
        except Exception as e:
            logger.error(f"Error parsing shard {shard_path}: {e}")
        return audio_map

    def _generate_subset_entries(self, source_map: Dict[str, Dict], filter_data: Dict[str, List[Dict]]) -> List[ManifestEntry]:
        """Join source data with filter to produce final ManifestEntries."""
        entries = []
        
        # Iterate over source files that exist in filter
        # We iterate source keys to maintain roughly the original order and shard distribution
        for key, source_entry in source_map.items():
            if key not in filter_data:
                continue
                
            # Get the crucial audio metadata (tar pointers) from source
            source_audio = source_entry.get('audio', {})
            
            if self.update_metadata:
                # Case A: Drive by Filter (Update Metadata)
                # If filter has multiple entries (e.g. multiple captions), we emit multiple entries
                for filter_entry in filter_data[key]:
                    entries.append(self._create_entry_from_filter(filter_entry, source_audio, key))
            else:
                # Case B: Drive by Source (Preserve Metadata)
                # We keep the source entry as-is, effectively just filtering the file list
                try:
                    # Clean entry to match ManifestEntry constructor signature
                    entries.append(self._create_entry_from_source(source_entry))
                except Exception as e:
                    logger.warning(f"Skipping invalid source entry {key}: {e}")
                    
        return entries

    def _create_entry_from_filter(self, filter_entry: Dict, source_audio, key: str) -> ManifestEntry:
        """Construct a ManifestEntry using filter metadata and source audio.
        
        Args:
            filter_entry: Entry from the filter manifest
            source_audio: Audio data from source (can be Dict or List[Dict] for multi-audio)
            key: Filename key used for matching
        """
        # Use provided dataset name override, or fallback to filter's dataset, or 'dataset'
        dataset = self.dataset_name or filter_entry.get('dataset', 'dataset')
        
        orig_id = filter_entry.get('original_file_id') or Path(key).stem
        
        # Determine Audio ID
        audio_id = filter_entry.get('audio_id')
        # Re-generate audio_id if dataset name changed or missing, to ensure uniqueness
        if not audio_id or self.dataset_name:
            audio_id = f"{dataset}_{orig_id}"
        
        # Determine if this is a multi-audio entry
        is_multi_audio = isinstance(source_audio, list)
            
        return ManifestEntry(
            audio_id=audio_id,
            text=filter_entry.get('caption') or filter_entry.get('text') or "",
            conversations=filter_entry.get('conversations'),
            audio=source_audio, # Reuse existing tarball pointers (can be Dict or List)
            dataset=dataset,
            original_file_id=orig_id,
            speaker_id=filter_entry.get('speaker_id') or filter_entry.get('speaker'),
            metadata=filter_entry.get('metadata', {}),
            is_multi_audio=is_multi_audio
        )

    def _create_entry_from_source(self, source_entry: Dict) -> ManifestEntry:
        """Convert a raw dictionary to ManifestEntry, filtering strictly for valid fields."""
        valid_keys = ManifestEntry.__annotations__.keys()
        clean_data = {k: v for k, v in source_entry.items() if k in valid_keys}
        
        # Update dataset name if override provided
        if self.dataset_name:
            clean_data['dataset'] = self.dataset_name
            # Should we also update audio_id if it contains the old dataset name?
            # For now, we trust user to be aware, or we could regenerate audio_id.
            # Usually safer to regenerate if dataset changes to avoid confusion.
            if 'original_file_id' in source_entry:
                clean_data['audio_id'] = f"{self.dataset_name}_{source_entry['original_file_id']}"
        
        # Ensure metadata is a dict
        if 'metadata' in source_entry and isinstance(source_entry['metadata'], dict):
            clean_data['metadata'] = source_entry['metadata']
        
        # Preserve is_multi_audio flag if present
        if 'is_multi_audio' in source_entry:
            clean_data['is_multi_audio'] = source_entry['is_multi_audio']
        # Or infer from audio type
        elif 'audio' in source_entry and isinstance(source_entry['audio'], list):
            clean_data['is_multi_audio'] = True
            
        return ManifestEntry(**clean_data)

    def _write_shard(self, output_path: str, entries: List[ManifestEntry], filename: str) -> Dict[str, Any]:
        """Write entries to NDJSON and calculate shard statistics."""
        # Calculate statistics (handles both single and multi-audio)
        durations = [e.get_total_duration() for e in entries if e.get_total_duration() > 0]
        
        # Collect all audio dicts for encoding stats
        all_audios = []
        for e in entries:
            all_audios.extend(e.get_audio_list())
        encodings = [a.get("encoding", "unknown") for a in all_audios if a.get("encoding")]
        caption_lengths = [len(e.text) for e in entries if e.text]

        shard_meta = {
            "filename": filename,
            "num_examples": len(entries),
            "total_duration_hours": round(sum(durations) / 3600, 3) if durations else 0,
        }
        
        if encodings:
            shard_meta["encodings"] = dict(Counter(encodings).most_common())

        if durations:
            shard_meta["audio_duration_seconds"] = {
                "min": round(float(np.min(durations)), 2),
                "max": round(float(np.max(durations)), 2),
                "mean": round(float(np.mean(durations)), 2),
                "median": round(float(np.median(durations)), 2)
            }

        if caption_lengths:
             non_empty = [l for l in caption_lengths if l > 0]
             if non_empty:
                shard_meta["caption_char_length"] = {
                    "min": int(np.min(non_empty)),
                    "max": int(np.max(non_empty)),
                    "mean": round(float(np.mean(non_empty)), 1),
                    "median": round(float(np.median(non_empty)), 1)
                }

        # Write file
        hasher = hashlib.md5()
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                json_str = json.dumps(entry.to_dict(), ensure_ascii=False)
                f.write(json_str + "\n")
                hasher.update(json_str.encode('utf-8'))
        
        shard_meta["size_bytes"] = os.path.getsize(output_path)
        shard_meta["checksum_md5"] = hasher.hexdigest()
        
        logger.info(f"Wrote {filename}: {len(entries)} entries")
        return shard_meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a subset manifest from existing shards, preserving audio metadata."
    )
    parser.add_argument("--input_subset_manifest", type=str, required=True,
                        help="Input quality-filtered raw manifest (file or directory)")
    parser.add_argument("--source_manifest_dir", type=str, required=True,
                        help="Directory containing existing processed manifests (.ndjson)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for new manifests")
    parser.add_argument("--prefix", type=str, default="subset",
                        help="Prefix for the output index file")
    parser.add_argument("--dataset_name", type=str,
                        help="Optional: Override dataset name in output entries to track filter strategy.")
    parser.add_argument("--update_metadata", action="store_true",
                        help="Update text/captions using the input filter. If False, preserves source metadata.")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Load Filter
    filter_data = FilterLoader.load(args.input_subset_manifest)
    if not filter_data:
        sys.exit(1)
        
    # 2. Identify Source Shards
    source_files = sorted(glob.glob(os.path.join(args.source_manifest_dir, "*.ndjson")))
    if not source_files:
        logger.error(f"No .ndjson files found in {args.source_manifest_dir}")
        sys.exit(1)
        
    logger.info(f"Found {len(source_files)} source shards to process")
    
    # 3. Process Shards
    processor = ShardProcessor(args.output_dir, args.update_metadata, args.dataset_name)
    all_shard_info = []
    all_entries = []
    
    for shard_path in tqdm(source_files, desc="Processing shards"):
        shard_meta, entries = processor.process(shard_path, filter_data)
        if shard_meta:
            all_shard_info.append(shard_meta)
            all_entries.extend(entries)
            
    if not all_entries:
        logger.error("No entries produced! Check if input filter matches source filenames.")
        sys.exit(1)
        
    # 4. Write Index File
    writer = ManifestWriter()
    output_path = Path(args.output_dir)
    
    writer.write_index_file(
        all_entries,
        output_path,
        args.prefix,
        all_shard_info
    )
    
    print("\n" + "=" * 60)
    print(f"SUBSET MANIFEST CREATION COMPLETE")
    print(f"Total Entries: {len(all_entries)}")
    print(f"Shards Created: {len(all_shard_info)}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
