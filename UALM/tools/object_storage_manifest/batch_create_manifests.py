# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
"""
Production-grade batch pipeline to create sharded, tarball-based datasets from YAML configuration.

Architecture:
- Validates configuration upfront before processing
- Provides clear progress tracking and error reporting
- Supports skipping already-processed datasets
- Modular design with clear separation of concerns
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import shutil

import yaml
from tqdm import tqdm


# =============================================================================
# 1. CONFIGURATION MODELS
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    input: Optional[str] = None
    audio_source_dir: Optional[str] = None
    location_key: Optional[str] = "location"
    caption_key: Optional[str] = "text"
    conversation_key: Optional[str] = "conversations"
    s3_dataset_dir: Optional[str] = None
    prefix: Optional[str] = None
    is_multi_audio: bool = False  # Whether location_key points to list of audio paths
    
    def validate(self) -> List[str]:
        """Validate dataset configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not self.input and not self.audio_source_dir:
            errors.append(f"{self.name}: Must provide 'input' manifest or 'audio_source_dir'")
        
        if self.input and not os.path.exists(self.input):
            errors.append(f"{self.name}: Input manifest not found: {self.input}")
        
        if self.audio_source_dir and not os.path.isdir(self.audio_source_dir):
            errors.append(f"{self.name}: Audio source directory not found: {self.audio_source_dir}")
        
        return errors


@dataclass
class CommonConfig:
    """Common configuration applied to all datasets."""
    manifest_output_dir: str
    tarball_output_dir: str
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    convert_to_s3: bool = False
    verify_audio: bool = True
    num_workers: int = 4
    shard_size: int = 10000
    timeout_seconds: int = 0  # 0 = no timeout (recommended for large datasets)
    verbose_logging: bool = False
    auto_convert_wav: bool = False
    auto_convert_wav_remove_after: bool = False
    audio_dest_dir: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate common configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not self.manifest_output_dir:
            errors.append("common.manifest_output_dir is required")
        
        if not self.tarball_output_dir:
            errors.append("common.tarball_output_dir is required")
        
        if self.auto_convert_wav and not self.audio_dest_dir:
            errors.append("common.audio_dest_dir is required when auto_convert_wav is enabled")
        
        return errors


@dataclass
class BatchConfig:
    """Complete batch processing configuration."""
    common: CommonConfig
    datasets: List[DatasetConfig]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BatchConfig':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            BatchConfig instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        common_dict = data.get('common', {})
        common = CommonConfig(**common_dict)
        
        datasets_list = data.get('datasets', [])
        datasets = [DatasetConfig(**ds) for ds in datasets_list]
        
        return cls(common=common, datasets=datasets)
    
    def validate(self) -> List[str]:
        """Validate entire batch configuration.
        
        Returns:
            List of all error messages (empty if valid)
        """
        errors = []
        
        # Validate common config
        errors.extend(self.common.validate())
        
        # Validate each dataset
        for dataset in self.datasets:
            errors.extend(dataset.validate())
        
        # Check for duplicate dataset names
        names = [ds.name for ds in self.datasets]
        duplicates = set([name for name in names if names.count(name) > 1])
        if duplicates:
            errors.append(f"Duplicate dataset names found: {', '.join(duplicates)}")
        
        return errors


# =============================================================================
# 2. CONFIGURATION VALIDATOR
# =============================================================================

class ConfigValidator:
    """Validate batch configuration before processing."""
    
    @staticmethod
    def validate_and_report(config: BatchConfig) -> bool:
        """Validate configuration and print detailed error report.
        
        Args:
            config: Batch configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        errors = config.validate()
        
        if errors:
            print("=" * 60)
            print("❌ CONFIGURATION ERRORS")
            print("=" * 60)
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
            print("=" * 60)
            print(f"\nFound {len(errors)} error(s). Please fix and try again.")
            return False
        
        return True


# =============================================================================
# 3. DATASET PROCESSOR
# =============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single dataset."""
    dataset_name: str
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    skipped: bool = False


class DatasetProcessor:
    """Process a single dataset."""
    
    def __init__(self, script_path: str = "create_manifest.py"):
        """Initialize dataset processor.
        
        Args:
            script_path: Path to the create_manifest script
        """
        self.script_path = script_path
    
    def process_dataset(
        self,
        dataset_config: DatasetConfig,
        common_config: CommonConfig
    ) -> ProcessingResult:
        """Process a single dataset.
        
        Args:
            dataset_config: Dataset-specific configuration
            common_config: Common configuration
            
        Returns:
            ProcessingResult with outcome
        """
        start_time = time.time()
        
        try:
            # Build command
            cmd = self._build_command(dataset_config, common_config)
            
            # Run subprocess
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=False,
                timeout=common_config.timeout_seconds if common_config.timeout_seconds > 0 else None
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return ProcessingResult(
                    dataset_name=dataset_config.name,
                    success=True,
                    duration_seconds=duration
                )
            else:
                return ProcessingResult(
                    dataset_name=dataset_config.name,
                    success=False,
                    duration_seconds=duration,
                    error_message=f"Process exited with code {result.returncode}"
                )
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ProcessingResult(
                dataset_name=dataset_config.name,
                success=False,
                duration_seconds=duration,
                error_message=f"Timeout after {common_config.timeout_seconds}s"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                dataset_name=dataset_config.name,
                success=False,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    def _build_command(
        self,
        dataset_config: DatasetConfig,
        common_config: CommonConfig
    ) -> List[str]:
        """Build command-line arguments for create_manifest.py.
        
        Args:
            dataset_config: Dataset-specific configuration
            common_config: Common configuration
            
        Returns:
            List of command arguments
        """
        cmd = [sys.executable, self.script_path]
        
        # Add input if provided
        if dataset_config.input:
            cmd.extend(["--input", dataset_config.input])
        
        # Output directories
        manifest_output_base = common_config.manifest_output_dir
        tarball_output_base = common_config.tarball_output_dir
        
        cmd.extend([
            "--output_dir", manifest_output_base,
            "--tarball_output_dir", tarball_output_base,
            "--dataset_name", dataset_config.name,
            "--create_tarballs"  # Always create tarballs in batch mode
        ])
        
        # Determine prefix
        if dataset_config.prefix:
            prefix = dataset_config.prefix
        elif dataset_config.input:
            filename = Path(dataset_config.input).stem.lower()
            prefix = self._detect_split_prefix(filename)
        else:
            prefix = "train"
        
        cmd.extend(["--prefix", prefix])
        
        # Add common arguments
        self._add_common_args(cmd, common_config)
        
        # Add dataset-specific arguments
        self._add_dataset_args(cmd, dataset_config)
        
        return cmd
    
    def _add_common_args(self, cmd: List[str], config: CommonConfig):
        """Add common configuration arguments to command."""
        if config.s3_bucket:
            cmd.extend(["--s3_bucket", config.s3_bucket])
        
        if config.s3_prefix:
            cmd.extend(["--s3_prefix", config.s3_prefix])
        
        if config.convert_to_s3:
            cmd.append("--convert_to_s3")
        
        if config.verify_audio:
            cmd.append("--verify_audio")
        
        if config.auto_convert_wav:
            cmd.append("--auto_convert_wav")
        
        if config.audio_dest_dir:
            cmd.extend(["--audio_dest_dir", config.audio_dest_dir])
        
        if config.verbose_logging:
            cmd.append("--verbose_logging")
        
        cmd.extend([
            "--num_workers", str(config.num_workers),
            "--shard_size", str(config.shard_size)
        ])
    
    def _add_dataset_args(self, cmd: List[str], config: DatasetConfig):
        """Add dataset-specific arguments to command."""
        if config.audio_source_dir:
            cmd.extend(["--audio_source_dir", config.audio_source_dir])
        
        if config.location_key:
            cmd.extend(["--location_key", config.location_key])
        
        if config.caption_key:
            cmd.extend(["--caption_key", config.caption_key])
        
        if config.conversation_key:
            cmd.extend(["--conversation_key", config.conversation_key])
        
        if config.s3_dataset_dir:
            cmd.extend(["--s3_dataset_dir", config.s3_dataset_dir])
        
        if config.is_multi_audio:
            cmd.append("--is_multi_audio")
    
    @staticmethod
    def _detect_split_prefix(filename: str) -> str:
        """Detect dataset split from filename.
        
        Args:
            filename: Input filename (lowercased)
            
        Returns:
            Detected split name ('train', 'val', 'test') or 'train'
        """
        for split in ['train', 'val', 'valid', 'test']:
            if split in filename:
                return split if split != 'valid' else 'val'
        return 'train'


# =============================================================================
# 4. BATCH ORCHESTRATOR
# =============================================================================

class BatchOrchestrator:
    """Orchestrate batch processing of multiple datasets."""
    
    def __init__(self, processor: DatasetProcessor):
        """Initialize batch orchestrator.
        
        Args:
            processor: Dataset processor instance
        """
        self.processor = processor
        self.common_config = None  # Set during run()
    
    def run(self, config: BatchConfig, skip_existing: bool = True) -> int:
        """Run batch processing.
        
        Args:
            config: Batch configuration
            skip_existing: Whether to skip datasets with existing output
            
        Returns:
            Exit code (0 for success, 1 for failures)
        """
        # Store common config for use in helper methods
        self.common_config = config.common
        
        # Print configuration summary
        self._print_config_summary(config)
        
        # Determine what to process
        to_process, to_skip = self._plan_processing(config, skip_existing)
        
        if not to_process:
            print("\n✅ All datasets already exist. Nothing to process.")
            return 0
        
        # Print plan
        self._print_processing_plan(to_process, to_skip)
        
        # Process datasets
        results = []
        try:
            for dataset_config in tqdm(to_process, desc="Overall Progress", unit="dataset"):
                print(f"\n{'=' * 60}")
                print(f"Processing: {dataset_config.name}")
                print(f"{'=' * 60}")
                
                result = self.processor.process_dataset(dataset_config, config.common)
                results.append(result)
                
                if result.success:
                    print(f"✅ {dataset_config.name} completed in {result.duration_seconds:.1f}s")
                else:
                    print(f"❌ {dataset_config.name} failed: {result.error_message}")
                
                # delete temporary converted audio if applicable
                if config.common.auto_convert_wav and config.common.audio_dest_dir and config.common.auto_convert_wav_remove_after:
                    converted_dir = os.path.join(config.common.audio_dest_dir, dataset_config.name)
                    if os.path.isdir(converted_dir):
                        print(f"🗑️  Removing temporary converted audio directory: {converted_dir}")
                        shutil.rmtree(converted_dir)
        
        except KeyboardInterrupt:
            print("\n\n⛔ Batch processing interrupted by user!")
            self._print_summary(results, to_skip)
            return 1
        
        # Print final summary
        self._print_summary(results, to_skip)
        
        # Return appropriate exit code
        failures = [r for r in results if not r.success]
        return 1 if failures else 0
    
    def _plan_processing(
        self,
        config: BatchConfig,
        skip_existing: bool
    ) -> Tuple[List[DatasetConfig], List[str]]:
        """Determine which datasets to process and which to skip.
        
        Args:
            config: Batch configuration
            skip_existing: Whether to skip existing datasets
            
        Returns:
            Tuple of (datasets_to_process, skipped_dataset_names)
        """
        to_process = []
        to_skip = []
        
        for dataset_config in config.datasets:
            if skip_existing and self._output_exists(dataset_config, config.common):
                to_skip.append(dataset_config.name)
            else:
                to_process.append(dataset_config)
        
        return to_process, to_skip
    
    def _output_exists(self, dataset_config: DatasetConfig, common_config: CommonConfig) -> bool:
        """Check if output already exists for a dataset.
        
        Args:
            dataset_config: Dataset configuration
            common_config: Common configuration
            
        Returns:
            True if output exists, False otherwise
        """
        # Check if tarball output directory exists and has content
        output_dir = os.path.join(common_config.tarball_output_dir, dataset_config.name)
        
        if os.path.isdir(output_dir):
            # Check if it has tarball files
            tar_files = list(Path(output_dir).glob("*.tar"))
            if tar_files:
                return True
        
        return False
    
    def _print_config_summary(self, config: BatchConfig):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("BATCH PROCESSING CONFIGURATION")
        print("=" * 60)
        print(f"Datasets: {len(config.datasets)}")
        print(f"Workers: {config.common.num_workers}")
        print(f"Shard size: {config.common.shard_size}")
        print(f"Verify audio: {config.common.verify_audio}")
        print(f"Auto-convert WAV: {config.common.auto_convert_wav}")
        print(f"Convert to S3: {config.common.convert_to_s3}")
        print(f"\nOutput directories:")
        print(f"  Manifests: {config.common.manifest_output_dir}")
        print(f"  Tarballs: {config.common.tarball_output_dir}")
        if config.common.audio_dest_dir:
            print(f"  Converted audio: {config.common.audio_dest_dir}")
    
    def _print_processing_plan(self, to_process: List[DatasetConfig], to_skip: List[str]):
        """Print processing plan."""
        print("\n" + "=" * 60)
        print("PROCESSING PLAN")
        print("=" * 60)
        
        if to_process:
            print(f"\n📋 To process ({len(to_process)}):")
            for i, ds in enumerate(to_process, 1):
                print(f"\n  {i}. {ds.name}")
                print(f"     {'─' * 50}")
                
                # Determine input type and display accordingly
                if ds.input:
                    input_ext = Path(ds.input).suffix.lower()
                    if input_ext == ".txt":
                        print(f"     📄 Input type: Text file (absolute paths)")
                        print(f"     📂 File: {ds.input}")
                    elif input_ext in [".jsonl", ".json", ".ndjson"]:
                        print(f"     📄 Input type: Manifest ({input_ext})")
                        print(f"     📂 File: {ds.input}")
                    else:
                        print(f"     📄 Input type: File")
                        print(f"     📂 File: {ds.input}")
                else:
                    print(f"     📄 Input type: Directory scan")
                    print(f"     📂 Directory: {ds.audio_source_dir}")
                
                # Show audio source directory if provided
                if ds.audio_source_dir:
                    print(f"     🎵 Audio source: {ds.audio_source_dir}")
                
                # Show multi-audio mode
                if ds.is_multi_audio:
                    print(f"     🎵 Multi-audio: ENABLED (location_key is list of paths)")
                
                # Show what will happen
                print(f"     ⚙️  Actions:")
                if self.common_config and self.common_config.auto_convert_wav:
                    print(f"        • Convert audio to WAV")
                    print(f"        • Save to: {self.common_config.audio_dest_dir}/{ds.s3_dataset_dir or ds.name}")
                    print(f"        ⚠️  Consider disabling auto_convert_wav - VirtualFileSection supports MP3/FLAC directly!")
                else:
                    print(f"        • Use source audio directly")
                
                if self.common_config and self.common_config.verify_audio:
                    print(f"        • Extract audio metadata")
                
                print(f"        • Create tarballs")
                print(f"        • Generate sharded manifests")
                
                # Show output locations
                if self.common_config:
                    print(f"     📤 Output:")
                    print(f"        • Manifests: {self.common_config.manifest_output_dir}/{ds.name}")
                    print(f"        • Tarballs: {self.common_config.tarball_output_dir}/{ds.name}")
                    print(f"        • Prefix: {ds.prefix or 'train'}")
        
        if to_skip:
            print(f"\n⏩ Skipping ({len(to_skip)}) - already exist:")
            for name in to_skip:
                print(f"  • {name}")
        
        print("=" * 60)
    
    def _print_summary(self, results: List[ProcessingResult], skipped: List[str]):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        
        total = len(results) + len(skipped)
        print(f"\nTotal datasets: {total}")
        print(f"  ✅ Succeeded: {len(successes)}")
        if skipped:
            print(f"  ⏩ Skipped: {len(skipped)}")
        if failures:
            print(f"  ❌ Failed: {len(failures)}")
        
        if successes:
            print(f"\n✅ Successfully processed:")
            for result in successes:
                print(f"  • {result.dataset_name} ({result.duration_seconds:.1f}s)")
        
        if skipped:
            print(f"\n⏩ Skipped (already exist):")
            for name in skipped:
                print(f"  • {name}")
        
        if failures:
            print(f"\n❌ Failed:")
            for result in failures:
                print(f"  • {result.dataset_name}: {result.error_message}")
        
        if results:
            total_time = sum(r.duration_seconds for r in results)
            print(f"\nTotal processing time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        
        print("=" * 60)


# =============================================================================
# 5. MAIN & CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch create sharded, tarball-based datasets from YAML config (refactored production version)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML configuration file"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process all datasets, even if output already exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show plan without processing"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    try:
        config = BatchConfig.from_yaml(args.config)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Validate configuration
    if not ConfigValidator.validate_and_report(config):
        return 1
    
    print("✅ Configuration is valid")
    
    # Determine the path to create_manifest.py (in same directory as this script)
    script_dir = Path(__file__).parent.resolve()
    create_manifest_script = script_dir / "create_manifest.py"
    
    if not create_manifest_script.exists():
        print(f"❌ Error: create_manifest.py not found at {create_manifest_script}")
        return 1
    
    # Dry run mode
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No actual processing will occur")
        processor = DatasetProcessor(str(create_manifest_script))
        orchestrator = BatchOrchestrator(processor)
        orchestrator.common_config = config.common  # Set for display
        
        to_process, to_skip = orchestrator._plan_processing(config, skip_existing=not args.no_skip_existing)
        orchestrator._print_processing_plan(to_process, to_skip)
        
        if to_process:
            print("\n💻 Command preview for first dataset:")
            cmd = processor._build_command(to_process[0], config.common)
            print(f"\n  {' '.join(cmd)}\n")
        
        return 0
    
    # Run batch processing
    processor = DatasetProcessor(str(create_manifest_script))
    orchestrator = BatchOrchestrator(processor)
    
    skip_existing = not args.no_skip_existing
    exit_code = orchestrator.run(config, skip_existing=skip_existing)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

