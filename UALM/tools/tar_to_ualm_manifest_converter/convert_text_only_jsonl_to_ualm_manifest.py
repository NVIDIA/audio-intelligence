# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
"""
Production-grade conversion tool for Text-Only UALM Manifests.

This script converts text-only JSONL files into the UALM training format. 
It performs the following key functions:

1.  **Length Estimation**: Estimates total token count for bucket sorting.
2.  **Metadata Unification**: Consolidates all datasets into a single random-access index 
    (LMDB or JSONL) to minimize file handle overhead during training.
3.  **Task Injection**: Injects 'text_only' task metadata to guide the runtime data 
    loader's chat template construction.
4.  **Conversation Format Conversion**: Converts input/output format to conversations format.
5.  **Weight Handling**: Calculates and outputs suggested training arguments for 
    weighted sampling via `DataIteratorFactory`.

Input JSONL Format:
    Each line should be a JSON object with the following structure:
    {"input": [{"role": "user", "content": <question>}], "output": <output_text>}
    
    Additional metadata fields (category, generator, etc.) are preserved but optional.

Architecture:
    - `Config`: Handles configuration parsing and validation.
    - `ManifestProcessor`: Manages parallel processing of source files.
    - `MetadataBuilder`: Standardizes and cleans raw entries into UALM metadata.
    - `OutputWriter`: Handles writing artifacts (Manifests, Stats, LMDB/JSONL).

Usage:
    python convert_text_only_jsonl_to_ualm_manifest.py --config config.yaml --output-dir exp/manifest
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False

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
    """Processing options for a dataset."""
    data_weight: float = 1.0
    # Key names for input/output extraction
    input_keyname: str = "input"
    output_keyname: str = "output"
    # Optional: custom ID field name (if not present, will generate hash-based ID)
    id_keyname: Optional[str] = None
    
    def __init__(self, **kwargs):
        self.data_weight = kwargs.get("data_weight", 1.0)
        self.input_keyname = kwargs.get("input_keyname", "input")
        self.output_keyname = kwargs.get("output_keyname", "output")
        self.id_keyname = kwargs.get("id_keyname", None)


@dataclass
class DatasetConfig:
    """Configuration for a dataset source."""
    id: str
    source_files: List[Path]
    options: ProcessingOptions


@dataclass
class GlobalConfig:
    """Global script configuration."""
    text_tokenizer: str
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
                
                # Collect source files
                source_files = []
                src = item.get("manifest_source", {})
                
                # Support both direct file list and directory + pattern
                if "files" in src:
                    # Direct file list
                    for f in src["files"]:
                        path = Path(f)
                        if path.exists():
                            source_files.append(path)
                        else:
                            logger.warning(f"File not found: {f}")
                elif "base_dir" in src:
                    # Directory with patterns
                    base_dir = Path(src["base_dir"])
                    patterns = src.get("patterns", ["*.jsonl", "*.json", "*.ndjson"])
                    for pattern in patterns:
                        source_files.extend(base_dir.glob(pattern))
                        # Also search recursively
                        source_files.extend(base_dir.glob(f"**/{pattern}"))
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_files = []
                    for f in source_files:
                        if f not in seen:
                            seen.add(f)
                            unique_files.append(f)
                    source_files = unique_files
                
                ds = DatasetConfig(
                    id=item["id"],
                    source_files=source_files,
                    options=opts
                )
                datasets.append(ds)
        
        return cls(
            text_tokenizer=args.text_tokenizer,
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
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class MetadataBuilder:
    """Encapsulates logic for cleaning and normalizing metadata entries."""
    
    @staticmethod
    def extract_conversations(entry: Dict, options: ProcessingOptions) -> List[Dict]:
        """
        Extract and convert input/output format to conversations format.
        
        Input format: {"input": [{"role": "user", "content": <text>}], "output": <text>}
        Output format: [{"from": "human", "value": <text>}, {"from": "gpt", "value": <text>}]
        """
        conversations = []
        
        # Process input messages
        input_messages = entry.get(options.input_keyname, [])
        for msg in input_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map role to conversation format
            if role in ["user", "human"]:
                conversations.append({"from": "human", "value": content})
            elif role in ["assistant", "gpt", "model"]:
                conversations.append({"from": "gpt", "value": content})
            elif role == "system":
                # System messages can be prepended to the first user message
                # or handled separately - here we add as a separate entry
                conversations.append({"from": "system", "value": content})
            else:
                # Default to human for unknown roles
                conversations.append({"from": "human", "value": content})
        
        # Process output
        output_text = entry.get(options.output_keyname, "")
        if output_text:
            conversations.append({"from": "gpt", "value": output_text})
        
        return conversations
    
    @staticmethod
    def calculate_length(conversations: List[Dict], tokenizer: TokenizerWrapper) -> int:
        """
        Estimates total sequence length (tokens).
        """
        total_tokens = 1  # <bos>
        
        for msg in conversations:
            content = msg.get("value", "")
            total_tokens += 3  # Overhead (<role> <modality> <eos>)
            if content:
                total_tokens += tokenizer.count_tokens(content)
        
        return total_tokens

    @staticmethod
    def generate_sample_id(entry: Dict, line_idx: int, options: ProcessingOptions) -> str:
        """
        Generate a unique sample ID.
        
        Priority:
        1. Use custom ID field if specified and present
        2. Generate hash-based ID from content
        """
        # Try custom ID field
        if options.id_keyname and options.id_keyname in entry:
            return str(entry[options.id_keyname])
        
        # Generate hash-based ID from content
        content_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()[:12]
        return f"line_{line_idx}_{content_hash}"

    @staticmethod
    def build_unified_entry(sid: str, length: int, conversations: List[Dict],
                           raw_entry: Dict) -> Tuple[str, int, Dict]:
        """
        Constructs the standardized metadata entry.
        
        Returns: (sample_id, length, metadata_dict)
        """
        # Text Content
        text_content = {
            "conversations": conversations,
        }
        
        # Preserve additional metadata if present
        extra_meta = {}
        for key in ["category", "generator", "license", "version", "system_prompt"]:
            if key in raw_entry:
                extra_meta[key] = raw_entry[key]
        
        # Unified Entry
        metadata = {
            "id": sid,
            "ualm_task": "text_only",
            "text": text_content,
        }
        
        if extra_meta:
            metadata["extra"] = extra_meta
        
        return sid, length, metadata


_worker_tokenizer = None


def _init_worker(tokenizer_name):
    """Initializer function for each worker process - called once per worker."""
    global _worker_tokenizer
    _worker_tokenizer = TokenizerWrapper(tokenizer_name)


def process_file_worker(args):
    """Multiprocessing worker function."""
    global _worker_tokenizer
    file_path, options, tokenizer_name, dataset_id = args
    
    tokenizer = _worker_tokenizer
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    
                    # Generate sample ID
                    local_id = MetadataBuilder.generate_sample_id(entry, line_idx, options)
                    global_id = f"{dataset_id}::{local_id}"
                    
                    # Extract conversations
                    conversations = MetadataBuilder.extract_conversations(entry, options)
                    
                    if not conversations:
                        continue
                    
                    # Calculate length
                    length = MetadataBuilder.calculate_length(conversations, tokenizer)
                    
                    # Build metadata
                    result = MetadataBuilder.build_unified_entry(
                        global_id, length, conversations, entry
                    )
                    results.append(result)
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error in {file_path}:{line_idx}: {e}")
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
        self.use_lmdb = use_lmdb and LMDB_AVAILABLE
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)
        
        self.meta_filename = "_metadata.lmdb" if self.use_lmdb else "_metadata.jsonl"

    def save_manifest(self, dataset_id: str, sample_ids: List[str]):
        """Saves the dataset-specific manifest JSON."""
        manifest_path = self.output_dir / f"{dataset_id}_manifest.json"
        
        data = {
            "data_entry": [
                {
                    "name": "text",
                    "path": str(self.output_dir / self.meta_filename),
                    "reader": "tarball_dialogue"  # Reuse TarballDialogueReader with TextOnlyTemplate
                }
            ],
            "samples": sample_ids
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(data, f, indent=2)
        return manifest_path

    def save_stats(self, dataset_id: str, stats: Dict[str, int]):
        """Saves length statistics for bucketing."""
        path = self.stats_dir / f"stats_text_only_{dataset_id}.jsonl"
        with open(path, 'w') as f:
            for sid, length in stats.items():
                f.write(json.dumps({sid: length}) + "\n")

    def save_unified_metadata(self, all_metadata: Dict[str, Dict]):
        """Saves the unified random-access index."""
        # 1. JSONL (Always save for inspection)
        jsonl_path = self.output_dir / "_metadata.jsonl"
        logger.info(f"Saving unified metadata JSONL to {jsonl_path}...")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in all_metadata.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
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
║              TEXT-ONLY UALM MANIFEST CONVERSION TOOL                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This tool converts text-only JSONL files into the UALM training format.     ║
║                                                                              ║
║  What it does:                                                               ║
║    1. Reads text data from source JSONL files                                ║
║    2. Converts input/output format to conversations format                   ║
║    3. Estimates sequence lengths (text tokens) for bucket sorting            ║
║    4. Unifies all datasets into a single LMDB/JSONL index for fast lookup    ║
║    5. Injects 'text_only' task label                                         ║
║    6. Generates per-dataset manifest JSONs for training                      ║
║                                                                              ║
║  Input format (per line):                                                    ║
║    {"input": [{"role": "user", "content": <text>}], "output": <text>}        ║
║                                                                              ║
║  Output files:                                                               ║
║    - <dataset_id>_manifest.json : Per-dataset sample index for DataLoader    ║
║    - _metadata.jsonl / .lmdb    : Unified metadata store (all datasets)      ║
║    - stats/text_only_<id>.jsonl : Length stats for bucket-based batching     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)

    def _print_config_plan(self):
        """Print configuration and processing plan."""
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        print(f"  Output Directory   : {self.config.output_dir}")
        print(f"  Text Tokenizer     : {self.config.text_tokenizer}")
        print(f"  Parallel Workers   : {self.config.num_workers}")
        print(f"  LMDB Output        : {'Enabled' if not self.config.disable_lmdb and LMDB_AVAILABLE else 'Disabled (JSONL only)'}")
        
        print("\n" + "-"*80)
        print("DATASETS TO PROCESS")
        print("-"*80)
        
        for ds_config in self.config.datasets:
            ds_id = ds_config.id
            num_files = len(ds_config.source_files)
            
            print(f"\n  [{ds_id}]")
            print(f"    • Task Type      : text_only")
            print(f"    • Source Files   : {num_files} JSONL file(s) found")
            print(f"    • Data Weight    : {ds_config.options.data_weight}")
            
            # Show first few files
            if num_files > 0:
                print(f"    • Files:")
                for i, f in enumerate(ds_config.source_files[:3]):
                    print(f"        - {f}")
                if num_files > 3:
                    print(f"        ... and {num_files - 3} more files")
        
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
        print(f"    ✓ {ds_id}")
        print(f"        Task     : text_only")
        print(f"        Samples  : {num_samples:,}")
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
        meta_type = 'lmdb' if not self.config.disable_lmdb and LMDB_AVAILABLE else 'jsonl'
        print(f"    • Unified Index  : {self.config.output_dir}/_metadata.{meta_type}")
        print(f"    • Length Stats   : {self.writer.stats_dir}/stats_text_only_<dataset_id>.jsonl")
        print("="*80)

    def run(self):
        # Print startup banner
        self._print_banner()
        
        # Phase 1: Print configuration
        self._print_phase(1, "DISCOVERY", 
            "Scanning source directories for JSONL files...")
        
        # Print configuration plan
        self._print_config_plan()
        
        total_files = sum(len(ds.source_files) for ds in self.config.datasets)
        logger.info(f"Discovery complete: {total_files} source files across {len(self.config.datasets)} datasets.")
        
        # Phase 2: Parallel processing
        self._print_phase(2, "PROCESSING", 
            "Reading source JSONL files, converting to conversations format, estimating token lengths.\n"
            "  For each sample:\n"
            "    - Convert input/output to conversations format\n"
            "    - Estimate sequence length = text_tokens\n"
            "    - Assign global sample ID: <dataset_id>::<local_id>")
        
        work_items = self._collect_work_items()
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
            self.writer.save_stats(ds_id, stats)
            
            # Print dataset summary
            self._print_dataset_summary(ds_id, ds_config, len(samples), manifest_path)
            
            # Generate CLI suggestion
            arg = f"text_only:{ds_id}:{manifest_path}:{ds_config.options.data_weight}"
            suggested_args.append(arg)
        
        # Phase 4: Write unified metadata
        print(f"\n  Writing unified metadata index ({len(all_metadata):,} total entries)...")
        self.writer.save_unified_metadata(all_metadata)
        
        # Print final summary
        self._print_final_summary(len(all_metadata), processed_datasets)
        
        # Print training suggestions
        self._print_suggestions(suggested_args)

    def _collect_work_items(self):
        """Collects all work items for parallel processing."""
        items = []
        for ds in self.config.datasets:
            for f in ds.source_files:
                items.append((
                    str(f), 
                    ds.options, 
                    self.config.text_tokenizer,
                    ds.id
                ))
        return items

    def _process_parallel(self, work_items):
        """Runs workers and groups results by dataset ID."""
        results = {ds.id: [] for ds in self.config.datasets}
        
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
        ds_id = args[3]
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
        description="Convert text-only JSONL files to UALM format."
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for manifests and metadata")
    parser.add_argument("--text-tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace tokenizer for text length estimation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--disable-lmdb", action="store_true",
                        help="Disable LMDB output (JSONL only)")
    return parser


if __name__ == "__main__":
    main()
