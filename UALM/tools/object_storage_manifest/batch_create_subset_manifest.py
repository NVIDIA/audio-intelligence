# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
"""
Batch script to create subset manifests sequentially from a YAML configuration.
Reuses the create_subset_manifest.py script for each dataset.
Designed for lightweight, sequential processing with detailed planning and progress tracking.

Features:
- Reads from a YAML config listing multiple datasets.
- Validates inputs before starting.
- Displays a comprehensive execution plan.
- Runs processing sequentially.
- Reports summary of successes/failures.
"""

import argparse
import os
import sys
import time
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION MODELS
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset subset operation."""
    name: str
    input_manifest: str
    source_manifest_dir: str
    output_dir: str
    prefix: str = "train"
    update_metadata: bool = True
    dataset_name: Optional[str] = None  # Override dataset name in entries

    def validate(self) -> List[str]:
        """Validate dataset configuration."""
        errors = []
        if not os.path.exists(self.input_manifest):
            errors.append(f"{self.name}: Input manifest not found: {self.input_manifest}")
        if not os.path.isdir(self.source_manifest_dir):
            errors.append(f"{self.name}: Source manifest dir not found: {self.source_manifest_dir}")
        return errors

@dataclass
class BatchConfig:
    """Complete batch processing configuration."""
    datasets: List[DatasetConfig]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BatchConfig':
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        datasets_list = data.get('datasets', [])
        datasets = []
        for ds in datasets_list:
            # Map YAML keys to Dataclass fields if needed, or direct unpacking
            datasets.append(DatasetConfig(
                name=ds['name'],
                input_manifest=ds['input_manifest'],
                source_manifest_dir=ds['source_manifest_dir'],
                output_dir=ds['output_dir'],
                prefix=ds.get('prefix', 'train'),
                update_metadata=ds.get('update_metadata', True),
                dataset_name=ds.get('dataset_name', None)
            ))
        return cls(datasets=datasets)

    def validate(self) -> List[str]:
        errors = []
        for ds in self.datasets:
            errors.extend(ds.validate())
        return errors

# =============================================================================
# 2. ORCHESTRATOR
# =============================================================================

class BatchOrchestrator:
    def __init__(self, script_path: str):
        self.script_path = script_path

    def print_plan(self, config: BatchConfig):
        print("\n" + "=" * 60)
        print("📋 BATCH PROCESSING PLAN")
        print("=" * 60)
        print(f"Script: {self.script_path}")
        print(f"Datasets to process: {len(config.datasets)}\n")

        for i, ds in enumerate(config.datasets, 1):
            print(f"{i}. {ds.name}")
            print(f"   Input Filter: {ds.input_manifest}")
            print(f"   Source Shards: {ds.source_manifest_dir}")
            print(f"   Output Dir: {ds.output_dir}")
            print(f"   Prefix: {ds.prefix}")
            print(f"   Update Metadata: {ds.update_metadata}")
            if ds.dataset_name:
                print(f"   Override Dataset Name: {ds.dataset_name}")
            print("-" * 40)
        print("=" * 60 + "\n")

    def run(self, config: BatchConfig) -> bool:
        success_count = 0
        failures = []

        print("🚀 Starting Batch Execution...\n")

        # Total progress bar
        for ds in tqdm(config.datasets, desc="Overall Batch Progress", unit="dataset"):
            print(f"\n[Processing] {ds.name}...")
            
            start_time = time.time()
            cmd = [
                sys.executable, self.script_path,
                "--input_subset_manifest", ds.input_manifest,
                "--source_manifest_dir", ds.source_manifest_dir,
                "--output_dir", ds.output_dir,
                "--prefix", ds.prefix,
            ]

            if ds.update_metadata:
                cmd.append("--update_metadata")
            
            # If dataset_name override is provided, pass it.
            # If not, we default to using the config 'name' as the dataset name? 
            # Or does the underlying script handle it?
            # The prompt usage example used --dataset_name explicitly.
            # Let's use ds.dataset_name if set, otherwise maybe default to ds.name?
            # The user's previous manual run used `--dataset_name name`.
            # So it's safer to pass it if we want the manifest entries to have this specific name.
            
            target_dataset_name = ds.dataset_name if ds.dataset_name else ds.name
            cmd.extend(["--dataset_name", target_dataset_name])

            try:
                # Run subprocess
                # We allow stdout to flow to console so user sees the underlying script's progress bars
                result = subprocess.run(cmd, check=True)
                duration = time.time() - start_time
                print(f"✅ Completed {ds.name} in {duration:.2f}s")
                success_count += 1

            except subprocess.CalledProcessError as e:
                duration = time.time() - start_time
                print(f"❌ Failed {ds.name} in {duration:.2f}s")
                failures.append((ds.name, str(e)))
            except Exception as e:
                print(f"❌ Error executing {ds.name}: {e}")
                failures.append((ds.name, str(e)))

        # Summary
        print("\n" + "=" * 60)
        print("🏁 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total: {len(config.datasets)}")
        print(f"Success: {success_count}")
        print(f"Failed: {len(failures)}")
        
        if failures:
            print("\nFailures:")
            for name, err in failures:
                print(f"  - {name}")
        
        return len(failures) == 0

# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch create subset manifests.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--dry-run", action="store_true", help="Only show the plan, do not execute.")
    args = parser.parse_args()

    # Locate the single-subset script
    script_dir = Path(__file__).parent.resolve()
    subset_script = script_dir / "create_subset_manifest.py"

    if not subset_script.exists():
        print(f"Error: Could not find {subset_script}")
        sys.exit(1)

    # Load Config
    try:
        batch_config = BatchConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        sys.exit(1)

    # Validate
    errors = batch_config.validate()
    if errors:
        print("Configuration Errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    orchestrator = BatchOrchestrator(str(subset_script))
    orchestrator.print_plan(batch_config)

    if args.dry_run:
        print("Dry run complete. Exiting.")
        sys.exit(0)

    success = orchestrator.run(batch_config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

