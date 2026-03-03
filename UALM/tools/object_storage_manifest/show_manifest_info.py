# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
"""
Display comprehensive information about generated manifests.
Quick way to verify manifest creation results.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def print_section(title: str, width: int = 60):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def show_manifest_info(manifest_dir: Path):
    """Display detailed information about a manifest directory."""
    
    # Find index file
    index_files = list(manifest_dir.glob("*.index"))
    if not index_files:
        print(f"❌ No index file found in {manifest_dir}")
        return False
    
    index_file = index_files[0]
    prefix = index_file.stem
    
    # Load index
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    # Display header
    print_section(f"Manifest: {manifest_dir.name}")
    
    # Basic info
    print(f"\n📁 Directory: {manifest_dir}")
    print(f"📄 Prefix: {prefix}")
    
    # Dataset info
    print("\n📊 Dataset Statistics:")
    print(f"  • Examples: {index_data.get('num_examples', 'N/A'):,}")
    print(f"  • Duration: {index_data.get('total_duration_hours', 0):.2f} hours")
    print(f"  • Datasets: {', '.join(index_data.get('datasets', []))}")
    
    # Multi-audio indicator (check from features schema or sample entry)
    features = index_data.get('features', {})
    if 'is_multi_audio' in features:
        print(f"  • Multi-audio: ✅ Yes")
    
    # Shard info
    shards = index_data.get('shards', [])
    if shards:
        print(f"\n📦 Shards ({len(shards)} files):")
        total_size = 0
        for i, shard in enumerate(shards[:3], 1):  # Show first 3
            shard_path = manifest_dir / shard['filename']
            if shard_path.exists():
                size = shard_path.stat().st_size
                total_size += size
                print(f"  {i}. {shard['filename']}: {shard['num_examples']:,} entries ({format_size(size)})")
        
        if len(shards) > 3:
            print(f"  ... and {len(shards) - 3} more shards")
            # Calculate total size
            for shard in shards[3:]:
                shard_path = manifest_dir / shard['filename']
                if shard_path.exists():
                    total_size += shard_path.stat().st_size
        
        print(f"  📏 Total manifest size: {format_size(total_size)}")
    
    # Caption statistics
    caption_stats = index_data.get('caption_char_length', {})
    if caption_stats:
        print("\n✏️ Caption Statistics:")
        print(f"  • Length: {caption_stats.get('min', 0)}-{caption_stats.get('max', 0)} chars")
        print(f"  • Average: {caption_stats.get('mean', 0):.1f} ± {caption_stats.get('std', 0):.1f} chars")
        print(f"  • Median: {caption_stats.get('median', 0):.1f} chars")
    
    # Audio statistics
    audio_stats = index_data.get('audio_duration_seconds', {})
    if audio_stats:
        print("\n🎵 Audio Statistics:")
        print(f"  • Duration: {audio_stats.get('min', 0):.1f}-{audio_stats.get('max', 0):.1f} seconds")
        print(f"  • Average: {audio_stats.get('mean', 0):.1f} ± {audio_stats.get('std', 0):.1f} seconds")
        print(f"  • Median: {audio_stats.get('median', 0):.1f} seconds")

    # Audio format info
    audio_meta = index_data.get('audio_metadata', {})
    if audio_meta:
        print("\n🔊 Audio Formats:")
        for key in ['sampling_rates', 'channels', 'encodings']:
            if key in audio_meta:
                dist = audio_meta[key]
                # Format as a clean one-liner
                dist_str = ", ".join([f"{k}: {v:,}" for k, v in dist.items()])
                print(f"  • {key.replace('_', ' ').title()}: {dist_str}")
    
    # Sample entry
    if shards:
        first_shard = manifest_dir / shards[0]['filename']
        if first_shard.exists():
            with open(first_shard, 'r') as f:
                first_line = f.readline()
                if first_line:
                    entry = json.loads(first_line)
                    print("\n📝 Sample Entry:")
                    # Pretty print the first sample entry
                    print(json.dumps(entry, indent=2))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Display information about generated manifests"
    )
    parser.add_argument(
        "manifest_dir",
        type=str,
        nargs='?',
        help="Path to manifest directory (or parent directory to show all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all manifests in directory"
    )
    
    args = parser.parse_args()
    
    if not args.manifest_dir:
        print("Usage: python show_manifest_info.py <manifest_dir>")
        print("       python show_manifest_info.py --all <parent_dir>")
        return
    
    path = Path(args.manifest_dir)
    
    if not path.exists():
        print(f"❌ Path not found: {path}")
        return
    
    if args.all or not list(path.glob("*.index")):
        # Show all manifests in subdirectories
        manifest_dirs = [d for d in path.iterdir() if d.is_dir() and list(d.glob("*.index"))]
        
        if not manifest_dirs:
            print(f"❌ No manifest directories found in {path}")
            return
        
        print_section(f"Found {len(manifest_dirs)} Manifests", 60)
        
        for manifest_dir in sorted(manifest_dirs):
            show_manifest_info(manifest_dir)
    else:
        # Show single manifest
        if not show_manifest_info(path):
            return
    
    print("\n" + "=" * 60)
    print(" ✅ Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
