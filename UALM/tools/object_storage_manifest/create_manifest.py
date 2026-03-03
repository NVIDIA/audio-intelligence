# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
"""
Production-grade script to create sharded, tarball-based datasets.

This script supports three input modes:
1. Manifest-based: Reads an existing .jsonl/.json manifest with audio paths and captions.
2. File list: Reads a .txt file with absolute audio paths (one per line).
3. Audio-only: Scans a directory of audio files directly.

Architecture:
- Modular classes with single responsibilities
- Clear separation of concerns
- Dependency injection for testability
- Comprehensive error handling and logging
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import io
import struct
import tarfile
import hashlib
import subprocess
import tempfile
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import Counter
from functools import partial
from itertools import repeat

import numpy as np
import soundfile as sf
from tqdm import tqdm


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONFIGURATION & DATA MODELS
# =============================================================================

@dataclass
class AudioInfo:
    """Audio file metadata for S3 byte-range reads."""
    sampling_rate: int
    channels: int
    duration: float
    samples: int
    bytes_per_sample: Optional[int] = None
    data_offset: Optional[int] = None
    encoding: str = "PCM_S"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for manifest."""
        result = {
            "sampling_rate": self.sampling_rate,
            "channels": self.channels,
            "duration": self.duration,
            "encoding": self.encoding
        }
        if self.bytes_per_sample is not None:
            result["bytes_per_sample"] = self.bytes_per_sample
        if self.data_offset is not None:
            result["data_offset"] = self.data_offset
        return result


@dataclass
class AudioSegment:
    """Audio segment with time boundaries."""
    offset: float = 0.0
    duration: Optional[float] = None
    
    @classmethod
    def from_entry(cls, entry: Dict[str, Any], total_duration: Optional[float] = None) -> "AudioSegment":
        """Create segment from manifest entry."""
        start_time, end_time = cls._resolve_time_bounds(entry)
        
        if end_time is not None:
            duration = end_time - start_time
        elif total_duration is not None:
            duration = total_duration - start_time
        else:
            duration = None
            
        return cls(offset=start_time, duration=duration)
    
    @staticmethod
    def _resolve_time_bounds(entry: Dict[str, Any]) -> Tuple[float, Optional[float]]:
        """Resolve start and end times from heterogeneous manifest keys."""
        def _parse_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        
        sources = [entry, entry.get("metadata", {})]
        
        start_time = None
        for source in sources:
            for key in ("start", "start_time", "offset", "start_sec"):
                start_time = _parse_float(source.get(key))
                if start_time is not None:
                    break
            if start_time is not None:
                break
        
        end_time = None
        for source in sources:
            for key in ("end", "end_time", "stop", "stop_time", "end_sec"):
                end_time = _parse_float(source.get(key))
                if end_time is not None:
                    break
            if end_time is not None:
                break
        
        if end_time is None:
            for source in sources:
                duration = _parse_float(source.get("duration"))
                if duration is not None and start_time is not None:
                    end_time = start_time + duration
                    break
        
        if start_time is None:
            start_time = 0.0
        
        return start_time, end_time


@dataclass
class ManifestEntry:
    """Unified manifest entry format for object storage.
    
    Supports both single-audio and multi-audio entries:
    - Single-audio: audio is Dict[str, Any], is_multi_audio=False (default)
    - Multi-audio: audio is List[Dict[str, Any]], is_multi_audio=True
    """
    audio_id: str
    audio: Union[Dict[str, Any], List[Dict[str, Any]]]  # Dict for single, List for multi
    dataset: str
    original_file_id: str
    is_multi_audio: bool = False  # Explicit flag for multi-audio entries
    text: Optional[str] = None
    conversations: Optional[Any] = None  # Can be str or List[Dict] for multi-audio
    speaker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None/empty values."""
        d = asdict(self)
        result = {k: v for k, v in d.items() if v is not None}
        if not result.get("metadata"):
            result.pop("metadata", None)
        if not result.get("speaker_id"):
            result.pop("speaker_id", None)
        if "text" in result and not result["text"]:
            result.pop("text", None)
        # Only include is_multi_audio if True (backward compatibility)
        if not result.get("is_multi_audio"):
            result.pop("is_multi_audio", None)
        return result
    
    def get_audio_list(self) -> List[Dict[str, Any]]:
        """Get audio as a list, regardless of single/multi format."""
        if self.is_multi_audio:
            return self.audio if isinstance(self.audio, list) else [self.audio]
        else:
            return [self.audio] if isinstance(self.audio, dict) else self.audio
    
    def get_total_duration(self) -> float:
        """Get total audio duration across all audio files."""
        audio_list = self.get_audio_list()
        return sum(a.get("duration", 0) for a in audio_list if a)
    
    def get_all_paths(self) -> List[str]:
        """Get all audio paths from this entry."""
        audio_list = self.get_audio_list()
        return [a.get("path") for a in audio_list if a and a.get("path")]


@dataclass
class ProcessingConfig:
    """Configuration for processing manifest entries."""
    dataset_name: str
    location_key: str = "location"
    caption_key: str = "captions"
    conversation_key: str = "conversations"
    convert_to_s3: bool = False
    dataset_dir: Optional[str] = None
    audio_source_dir: Optional[str] = None
    is_multi_audio: bool = False  # Whether to treat location_key as list of audio paths
    
    @property
    def effective_dataset_dir(self) -> str:
        """Get the effective dataset directory (dataset_dir or dataset_name)."""
        return self.dataset_dir or self.dataset_name


@dataclass
class AudioConverterConfig:
    """Configuration for audio conversion."""
    num_workers: int = 4
    temp_dir: Optional[str] = None
    skip_existing: bool = True
    target_format: str = "wav"
    target_encoding: str = "PCM_16"
    verbose: bool = False


# =============================================================================
# 2. AUDIO METADATA EXTRACTOR
# =============================================================================

class AudioMetadataExtractor:
    """Extract metadata from audio files (WAV, FLAC, MP3, etc.)."""
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize metadata extractor.
        
        Args:
            cache_enabled: Whether to cache metadata to avoid repeated reads
        """
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, Optional[AudioInfo]] = {}
    
    def extract(self, audio_path: str) -> Optional[AudioInfo]:
        """Extract audio metadata from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioInfo object or None if extraction fails
        """
        if self.cache_enabled and audio_path in self.cache:
            return self.cache[audio_path]
        
        try:
            ext = Path(audio_path).suffix.lower()
            
            if ext == ".wav":
                audio_info = self._extract_wav_metadata(audio_path)
            else:
                audio_info = self._extract_compressed_metadata(audio_path, ext)
            
            if self.cache_enabled:
                self.cache[audio_path] = audio_info
            
            return audio_info
            
        except Exception as e:
            logger.warning(f"Failed to read audio info for {audio_path}: {e}")
            return None
    
    def _extract_wav_metadata(self, audio_path: str) -> Optional[AudioInfo]:
        """Extract metadata from WAV file."""
        with sf.SoundFile(audio_path) as f:
            return AudioInfo(
                sampling_rate=f.samplerate,
                channels=f.channels,
                duration=f.frames / f.samplerate,
                samples=f.frames,
                bytes_per_sample=self._determine_bytes_per_sample(f.subtype),
                data_offset=self._get_wav_data_offset(audio_path),
                encoding=self._determine_encoding(f.subtype)
            )
    
    def _extract_compressed_metadata(self, audio_path: str, ext: str) -> Optional[AudioInfo]:
        """Extract metadata from non-WAV files."""
        ext_clean = ext.lstrip('.').upper()
        
        # Try soundfile first (works for FLAC, OGG)
        try:
            with sf.SoundFile(audio_path) as f:
                return AudioInfo(
                    sampling_rate=f.samplerate,
                    channels=f.channels,
                    duration=f.frames / f.samplerate,
                    samples=f.frames,
                    bytes_per_sample=None,
                    data_offset=None,
                    encoding=ext_clean
                )
        except:
            # Try soxi for MP3 and others
            metadata = self._get_audio_metadata_soxi(audio_path)
            if metadata:
                return AudioInfo(
                    sampling_rate=int(metadata['sample_rate']),
                    channels=int(metadata['channels']),
                    duration=metadata['duration'],
                    samples=int(metadata['duration'] * metadata['sample_rate']),
                    bytes_per_sample=None,
                    data_offset=None,
                    encoding=ext_clean
                )
        
        return None
    
    def _get_audio_metadata_soxi(self, audio_path: str) -> Optional[Dict[str, float]]:
        """Get audio metadata using soxi command."""
        try:
            result = subprocess.run(
                ["soxi", "-D", "-c", "-r", audio_path],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 3 and lines[0] and lines[1] and lines[2]:
                    return {
                        'duration': float(lines[0]),
                        'channels': int(lines[1]),
                        'sample_rate': int(lines[2])
                    }
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None
    
    def _get_wav_data_offset(self, audio_path: str) -> int:
        """Find the data chunk offset in a WAV file."""
        try:
            with open(audio_path, "rb") as f:
                if f.read(4) != b"RIFF":
                    return 0
                f.read(4)
                if f.read(4) != b"WAVE":
                    return 0
                
                while True:
                    chunk_id = f.read(4)
                    if not chunk_id:
                        break
                    chunk_size_bytes = f.read(4)
                    if len(chunk_size_bytes) < 4:
                        break
                    chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
                    if chunk_id == b"data":
                        return f.tell()
                    f.seek(chunk_size, 1)
        except Exception:
            pass
        return 0
    
    def _determine_bytes_per_sample(self, subtype: str) -> int:
        """Determine bytes per sample from soundfile subtype."""
        if 'PCM_16' in subtype or '16' in subtype:
            return 2
        elif 'PCM_24' in subtype or '24' in subtype:
            return 3
        elif 'PCM_32' in subtype or '32' in subtype:
            return 4
        elif 'PCM_08' in subtype or '8' in subtype:
            return 1
        elif 'FLOAT' in subtype:
            return 4
        elif 'DOUBLE' in subtype:
            return 8
        else:
            return 2
    
    def _determine_encoding(self, subtype: str) -> str:
        """Determine PCM encoding from soundfile subtype."""
        if 'FLOAT' in subtype or 'DOUBLE' in subtype:
            return "PCM_F"
        else:
            return "PCM_S"


# =============================================================================
# 3. AUDIO CONVERTER
# =============================================================================

class AudioConverter:
    """Convert audio files to a standard format (WAV)."""
    
    def __init__(self, config: AudioConverterConfig):
        """Initialize audio converter.
        
        Args:
            config: Converter configuration
        """
        self.config = config
        
        # Determine temp directory
        if config.temp_dir:
            self.temp_dir = config.temp_dir
        elif os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK):
            self.temp_dir = '/dev/shm'
        else:
            self.temp_dir = '/tmp'
        
        logger.info(f"Using {self.temp_dir} for temporary files ({'RAM' if 'shm' in self.temp_dir else 'disk'})")
    
    def convert_batch(
        self,
        source_files: List[str],
        source_dir: str,
        dest_dir: str,
        dataset_name: str
    ) -> Tuple[int, int, int]:
        """Convert a batch of audio files to WAV format.
        
        Args:
            source_files: List of source file paths (can be relative or absolute)
            source_dir: Base directory for source files
            dest_dir: Base directory for destination files
            dataset_name: Dataset name for organizing output
            
        Returns:
            Tuple of (converted_count, skipped_count, failed_count)
        """
        logger.info(f"Converting {len(source_files)} unique audio files with {self.config.num_workers} workers")
        
        # Build conversion tasks
        tasks = []
        directories = set()
        
        for source_path in source_files:
            # Determine actual source path
            if os.path.isabs(source_path):
                actual_source = source_path
            else:
                actual_source = os.path.join(source_dir, source_path)
            
            # Determine relative path
            try:
                relative_path = os.path.relpath(actual_source, source_dir)
                if ".." in relative_path:
                    relative_path = Path(actual_source).name
            except (ValueError, OSError):
                relative_path = Path(actual_source).name
            
            # Construct destination path
            rel_no_ext = os.path.splitext(relative_path)[0]
            output_dir = os.path.join(dest_dir, dataset_name)
            dest_path = os.path.join(output_dir, f"{rel_no_ext}.wav")
            
            directories.add(os.path.dirname(dest_path))
            tasks.append((actual_source, dest_path))

            # print("actual_source", actual_source)
            # print("source_dir", source_dir)
            # print("relative_path", relative_path)
            # print("rel_no_ext", rel_no_ext)
            # print("dest_path", dest_path)
            # sys.exit()
        
        # Pre-create directories
        for dir_path in sorted(directories):
            os.makedirs(dir_path, exist_ok=True)
        
        # Convert files in parallel
        converted = skipped = failed = 0
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [executor.submit(self._convert_single_file, src, dst) for src, dst in tasks]
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Converting", unit="files"):
                try:
                    result = future.result()
                    if result == 'converted':
                        converted += 1
                    elif result == 'skipped':
                        skipped += 1
                    else:
                        failed += 1
                except:
                    failed += 1
        
        logger.info(f"Converted {converted}, skipped {skipped}, failed {failed}")
        return converted, skipped, failed
    
    def _convert_single_file(self, source_path: str, target_path: str) -> str:
        """Convert a single audio file to WAV."""
        if self.config.verbose:
            print(f"[CONVERT] {source_path} -> {target_path}")
        
        # Check if already converted
        if self.config.skip_existing:
            try:
                size = os.path.getsize(target_path)
                if size > 10240:  # > 10KB
                    return 'skipped'
                elif size > 0:
                    os.remove(target_path)
            except:
                pass
        
        # Convert using sox
        if not os.path.exists(source_path):
            return 'failed'
        
        temp_wav = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix='sox_',
                suffix='.wav',
                dir=self.temp_dir,
                delete=False
            ) as tmp_file:
                temp_wav = tmp_file.name
            
            cmd = ["sox", source_path, "-b", "16", "-e", "signed-integer", temp_wav]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(temp_wav):
                shutil.move(temp_wav, target_path)
                try:
                    os.chmod(target_path, 0o644)
                except OSError:
                    pass
                return 'converted'
        except Exception:
            pass
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
        
        return 'failed'


# =============================================================================
# 4. PATH RESOLVER
# =============================================================================

class PathResolver:
    """Centralized path resolution logic."""
    
    @staticmethod
    def resolve_source_path(location: str, audio_source_dir: Optional[str]) -> str:
        """Get absolute path to source audio file."""
        if os.path.isabs(location):
            return location
        if audio_source_dir:
            return os.path.join(audio_source_dir, location)
        return location
    
    @staticmethod
    def compute_relative_path(source_path: str, base_dir: str) -> str:
        """Compute relative path, handling edge cases."""
        if not base_dir or not os.path.isabs(source_path):
            return source_path
        
        try:
            relative = os.path.relpath(source_path, base_dir)
            if ".." in relative:
                return source_path # Return absolute path if it's outside base_dir
            return relative
        except (ValueError, OSError):
            return source_path


# =============================================================================
# 5. ENTRY PROCESSOR
# =============================================================================

class EntryProcessor:
    """Process individual manifest entries."""
    
    def __init__(
        self,
        metadata_extractor: AudioMetadataExtractor,
        path_resolver: PathResolver,
        verbose: bool = False,
        auto_convert: bool = False,
        audio_dest_dir: Optional[str] = None
    ):
        """Initialize entry processor.
        
        Args:
            metadata_extractor: Audio metadata extractor
            path_resolver: Path resolution utility
            verbose: Enable verbose logging
            auto_convert: Whether audio conversion is enabled
            audio_dest_dir: Base destination directory for converted audio
        """
        self.metadata = metadata_extractor
        self.paths = path_resolver
        self.verbose = verbose
        self.auto_convert = auto_convert
        self.audio_dest_dir = audio_dest_dir
    
    def process_entry(
        self,
        entry: Dict[str, Any],
        config: ProcessingConfig
    ) -> Optional[ManifestEntry]:
        """Process a single manifest entry.
        
        Args:
            entry: Raw manifest entry
            config: Processing configuration
            
        Returns:
            ManifestEntry or None if processing fails
        """
        # Branch based on multi-audio mode
        if config.is_multi_audio:
            return self._process_multi_audio_entry(entry, config)
        else:
            return self._process_single_audio_entry(entry, config)
    
    def _process_single_audio_entry(
        self,
        entry: Dict[str, Any],
        config: ProcessingConfig
    ) -> Optional[ManifestEntry]:
        """Process a single-audio manifest entry.
        
        Delegates audio processing to _process_single_audio_path to avoid code duplication.
        """
        try:
            loc_key = config.location_key or "location"
            cap_key = config.caption_key or "text"
            conversation_key = config.conversation_key or "conversations"
            
            location = entry.get(loc_key)
            if not location:
                if self.verbose:
                    print(f"[PROCESS] Skipping entry, no location: {entry}")
                return None
            
            # Handle case where single-audio mode receives a list (take first element)
            if isinstance(location, list):
                if self.verbose:
                    logger.warning(f"is_multi_audio=False but location is a list - using first element")
                location = location[0] if location else None
                if not location:
                    return None
            
            text = self._extract_text(entry, cap_key)
            conversations = self._extract_conversation(entry, conversation_key)
            
            # Delegate to shared audio processing method (avoids code duplication)
            audio_dict = self._process_single_audio_path(location, entry, config, 0)
            
            # Reject entry if audio processing failed
            if audio_dict is None:
                return None
            
            # Build segment for audio_id generation
            audio_info_duration = audio_dict.get("duration")
            segment = AudioSegment(offset=audio_dict.get("offset", 0.0), duration=audio_info_duration)
            
            # Build manifest entry
            return ManifestEntry(
                audio_id=self._generate_audio_id(config.dataset_name, location, segment),
                text=text or "",
                conversations=conversations or "",
                audio=audio_dict,
                dataset=entry.get("dataset", config.dataset_name),
                original_file_id=Path(location).stem,
                speaker_id=entry.get("speaker_id") or entry.get("speaker"),
                metadata=self._preserve_metadata(entry, loc_key, cap_key, conversation_key),
                is_multi_audio=False
            )
            
        except Exception as e:
            logger.error(f"Failed to process single-audio entry: {e}\nEntry: {entry}")
            return None
    
    def _process_multi_audio_entry(
        self,
        entry: Dict[str, Any],
        config: ProcessingConfig
    ) -> Optional[ManifestEntry]:
        """Process a multi-audio manifest entry.
        
        For multi-audio entries:
        - location_key points to a list of audio paths
        - Each audio is processed and stored in audio list
        - List order matches <sound-1>, <sound-2>, etc. placeholders
        """
        try:
            loc_key = config.location_key or "location"
            cap_key = config.caption_key or "text"
            conversation_key = config.conversation_key or "conversations"
            
            locations = entry.get(loc_key)
            if not locations:
                if self.verbose:
                    print(f"[PROCESS] Skipping multi-audio entry, no locations: {entry}")
                return None
            
            # Validate that locations is a list
            if not isinstance(locations, list):
                logger.error(f"is_multi_audio=True but {loc_key} is not a list: {type(locations)}")
                return None
            
            if len(locations) == 0:
                logger.error(f"is_multi_audio=True but {loc_key} is empty list")
                return None
            
            text = self._extract_text(entry, cap_key)
            conversations = self._extract_conversation(entry, conversation_key)
            
            # Process each audio file
            audio_list = []
            failed_indices = []
            for idx, location in enumerate(locations):
                audio_dict = self._process_single_audio_path(location, entry, config, idx)
                if audio_dict:
                    audio_list.append(audio_dict)
                else:
                    failed_indices.append(idx + 1)  # 1-indexed for <sound-N> reference
            
            if not audio_list:
                logger.error(f"No valid audio files processed for multi-audio entry")
                return None
            
            # STRICT: Reject entire entry if ANY audio failed
            # This prevents mismatched <sound-1>, <sound-2> references in conversation text
            if failed_indices:
                entry_id = entry.get("id") or entry.get("audio_id") or "unknown"
                logger.warning(
                    f"Rejecting multi-audio entry {entry_id} - {len(failed_indices)}/{len(locations)} "
                    f"audio files failed (indices: {failed_indices}). "
                    f"Partial entries would break <sound-N> references."
                )
                return None
            
            # Generate audio_id from entry id or first file
            entry_id = entry.get("id") or entry.get("audio_id") or Path(locations[0]).stem
            audio_id = f"{config.dataset_name}_{entry_id}"
            
            # Build manifest entry with audio as list
            return ManifestEntry(
                audio_id=audio_id,
                text=text or "",
                conversations=conversations,  # Keep as-is for multi-audio (usually List[Dict])
                audio=audio_list,  # List of audio dicts
                dataset=entry.get("dataset", config.dataset_name),
                original_file_id=str(entry_id),
                speaker_id=entry.get("speaker_id") or entry.get("speaker"),
                metadata=self._preserve_metadata(entry, loc_key, cap_key, conversation_key),
                is_multi_audio=True
            )
            
        except Exception as e:
            logger.error(f"Failed to process multi-audio entry: {e}\nEntry: {entry}")
            return None
    
    def _process_single_audio_path(
        self,
        location: str,
        entry: Dict[str, Any],
        config: ProcessingConfig,
        index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Process a single audio path and return audio dict.
        
        Used by multi-audio processing to handle each audio file.
        """
        try:
            if config.audio_source_dir:
                if self.auto_convert and self.audio_dest_dir:
                    location_no_ext = os.path.splitext(location)[0]
                    converted_relative = f"{location_no_ext}.wav"
                    dataset_subdir = config.effective_dataset_dir

                    if location.startswith('/') and config.audio_source_dir == "/":
                        metadata_source = os.path.join(self.audio_dest_dir, dataset_subdir, '.' + converted_relative)
                        output_path = os.path.join(dataset_subdir, '.' + converted_relative)
                        metadata_source = os.path.normpath(metadata_source)
                        output_path = os.path.normpath(output_path)
                    else:
                        metadata_source = os.path.join(self.audio_dest_dir, dataset_subdir, converted_relative)
                        output_path = os.path.join(dataset_subdir, converted_relative)
                else:
                    full_source = self.paths.resolve_source_path(location, config.audio_source_dir)
                    metadata_source = full_source
                    output_path = self.paths.compute_relative_path(full_source, config.audio_source_dir)
            else:
                metadata_source = location
                output_path = Path(location).name
            
            # Extract metadata
            audio_info = self.metadata.extract(metadata_source)
            segment = AudioSegment.from_entry(entry, audio_info.duration if audio_info else None)
            
            # Build audio dict
            audio_dict = self._build_audio_dict(
                output_path if config.convert_to_s3 else metadata_source,
                audio_info,
                segment
            )
            
            return audio_dict
            
        except Exception as e:
            logger.warning(f"Failed to process audio path {location}: {e}")
            return None
    
    def _extract_text(self, entry: Dict[str, Any], caption_key: str) -> Optional[str]:
        """Extract text/caption from entry."""
        caption = entry.get(caption_key)
        if not caption:
            return None
        if isinstance(caption, list):
            return caption[0] if caption else None
        return caption
    
    def _extract_conversation(self, entry: Dict[str, Any], conversation_key: str) -> Optional[str]:
        """Extract conversation from entry."""
        conversation = entry.get(conversation_key)
        if not conversation:
            return None
        return conversation
    
    def _generate_audio_id(self, dataset_name: str, filename: str, segment: AudioSegment) -> str:
        """Generate unique audio ID."""
        path = Path(filename)
        
        if os.path.isabs(filename):
            base_id = path.stem
        elif "/" in filename:
            base_id = str(path.parent / path.stem).replace("/", "_")
        else:
            base_id = path.stem
        
        if segment.offset > 0:
            return f"{dataset_name}_{base_id}_{int(segment.offset)}"
        return f"{dataset_name}_{base_id}"
    
    def _build_audio_dict(
        self,
        path: str,
        audio_info: Optional[AudioInfo],
        segment: AudioSegment
    ) -> Optional[Dict[str, Any]]:
        """Build audio metadata dictionary.
        
        Returns None if critical audio metadata is missing (failed extraction).
        This ensures only valid, loadable entries make it into the manifest.
        """
        # STRICT VALIDATION: Reject entries without critical metadata
        if audio_info is None:
            logger.warning(f"Rejecting entry - audio metadata extraction failed: {path}")
            return None
        
        if not audio_info.sampling_rate or not audio_info.channels:
            logger.warning(f"Rejecting entry - missing sampling_rate or channels: {path}")
            return None
        
        audio_dict = {"path": path}
        audio_dict.update(audio_info.to_dict())
        
        audio_dict["offset"] = segment.offset
        if segment.duration is not None:
            audio_dict["duration"] = segment.duration
        elif audio_info.duration:
            audio_dict["duration"] = audio_info.duration - segment.offset
        
        # Final validation: ensure duration exists
        if not audio_dict.get("duration") or audio_dict["duration"] <= 0:
            logger.warning(f"Rejecting entry - invalid or missing duration: {path}")
            return None
        
        return audio_dict
    
    def _preserve_metadata(
        self,
        entry: Dict[str, Any],
        location_key: str,
        caption_key: str,
        conversation_key: str
    ) -> Dict[str, Any]:
        """Preserve extra metadata from original entry."""
        handled_keys = {
            location_key, caption_key, conversation_key, "start", "end", "offset", "duration",
            "speaker", "speaker_id", "dataset"
        }
        
        extra_keys = set(entry.keys()) - handled_keys
        if extra_keys:
            return {k: entry[k] for k in extra_keys}
        return {}


# =============================================================================
# 6. MANIFEST LOADER
# =============================================================================

class ManifestLoader:
    """Load manifest data from files or directories."""
    
    @staticmethod
    def load_from_file(input_path: str, location_key: str = "location", caption_key: str = "text", conversation_key: str = "conversations") -> List[Dict[str, Any]]:
        """Load entries from a manifest file (.jsonl, .json, or .txt).
        
        Args:
            input_path: Path to manifest file
            location_key: Key to use for audio path in synthetic entries
            caption_key: Key to use for caption in synthetic entries
            conversation_key: Key to use for conversations in synthetic entries
            
        Returns:
            List of raw manifest entries
        """
        logger.info(f"Loading manifest from file: {input_path}")
        
        # Check file extension
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext == ".txt":
            # Text file with absolute paths (one per line)
            entries = ManifestLoader._load_from_txt_file(input_path, location_key, caption_key, conversation_key)
        else:
            # JSON or JSONL format
            with open(input_path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == "[":
                    # JSON array format
                    data = json.load(f)
                    entries = data if isinstance(data, list) else [data]
                else:
                    # JSONL format
                    entries = [json.loads(line) for line in f if line.strip()]
        
        logger.info(f"Loaded {len(entries)} entries from manifest")
        return entries
    
    @staticmethod
    def _load_from_txt_file(
        txt_path: str,
        location_key: str = "location",
        caption_key: str = "text",
        conversation_key: str = "conversations"
    ) -> List[Dict[str, Any]]:
        """Load audio file paths from a text file (one absolute path per line).
        
        Args:
            txt_path: Path to text file containing audio paths
            location_key: Key to use for audio path in synthetic entries
            caption_key: Key to use for caption in synthetic entries
            conversation_key: Key to use for conversations in synthetic entries
            
        Returns:
            List of synthetic manifest entries
        """
        logger.info(f"Loading audio paths from text file: {txt_path}")
        
        entries = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                
                # Validate path exists
                if not os.path.exists(line):
                    logger.warning(f"Line {line_num}: File not found: {line}")
                    continue
                
                # Create synthetic manifest entry
                entries.append({
                    location_key: line,  # Absolute path
                    caption_key: "",     # Empty caption for audio-only
                    conversation_key: "" # Empth conversations for audio-only
                })
        
        logger.info(f"Loaded {len(entries)} audio file paths from text file")
        return entries
    
    @staticmethod
    def load_from_directory(
        audio_dir: str,
        location_key: str = "location",
        caption_key: str = "text",
        conversation_key: str = "conversations"
    ) -> List[Dict[str, Any]]:
        """Load entries by scanning an audio directory.
        
        Args:
            audio_dir: Directory containing audio files
            location_key: Key to use for audio path in synthetic entries
            caption_key: Key to use for caption in synthetic entries
            conversation_key: Key to use for conversations in synthetic entries
            
        Returns:
            List of synthetic manifest entries
        """
        logger.info(f"Scanning audio directory: {audio_dir}")
        
        audio_files = []
        extensions = ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.opus", "*.m4a"]
        for ext in extensions:
            audio_files.extend(glob.glob(os.path.join(audio_dir, "**", ext), recursive=True))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        entries = []
        for audio_file in audio_files:
            relative_path = os.path.relpath(audio_file, audio_dir)
            entries.append({
                location_key: relative_path,
                caption_key: "",
                conversation_key: ""
            })
        
        return entries


# =============================================================================
# 7. MANIFEST STATISTICS
# =============================================================================

class ManifestStatistics:
    """Calculate comprehensive statistics for manifest entries."""
    
    @staticmethod
    def calculate(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate all statistics.
        
        Args:
            entries: List of manifest entries
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'audio_duration': ManifestStatistics._audio_duration_stats(entries),
            'audio_metadata': ManifestStatistics._audio_metadata_stats(entries),
            'caption_char_length': ManifestStatistics._caption_char_stats(entries),
            'caption_word_count': ManifestStatistics._caption_word_stats(entries),
            'conversation_rounds': ManifestStatistics._conversation_stats(entries),
            'segmentation': ManifestStatistics._segmentation_stats(entries),
            'quality_metrics': ManifestStatistics._quality_metrics(entries),
            'dataset_distribution': ManifestStatistics._dataset_distribution(entries),
            'num_speakers': ManifestStatistics._count_speakers(entries)
        }
        
        # Add multi-audio statistics
        multi_audio_stats = ManifestStatistics._multi_audio_stats(entries)
        if multi_audio_stats:
            stats['multi_audio'] = multi_audio_stats
        
        return stats
    
    @staticmethod
    def _get_all_audio_dicts(entries: List[ManifestEntry]) -> List[Dict[str, Any]]:
        """Helper to get all audio dicts from entries (handles both single and multi)."""
        all_audios = []
        for e in entries:
            all_audios.extend(e.get_audio_list())
        return all_audios
    
    @staticmethod
    def _audio_duration_stats(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate audio duration statistics (per-entry total duration)."""
        durations = [e.get_total_duration() for e in entries if e.get_total_duration() > 0]
        if not durations:
            return {}
        
        return {
            "min": round(float(np.min(durations)), 2),
            "max": round(float(np.max(durations)), 2),
            "mean": round(float(np.mean(durations)), 2),
            "std": round(float(np.std(durations)), 2),
            "median": round(float(np.median(durations)), 2),
            "percentiles": {
                "25": round(float(np.percentile(durations, 25)), 2),
                "75": round(float(np.percentile(durations, 75)), 2),
                "90": round(float(np.percentile(durations, 90)), 2),
                "95": round(float(np.percentile(durations, 95)), 2),
                "99": round(float(np.percentile(durations, 99)), 2)
            }
        }
    
    @staticmethod
    def _audio_metadata_stats(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate audio metadata distribution (across all audio files)."""
        all_audios = ManifestStatistics._get_all_audio_dicts(entries)
        
        sample_rates = [a.get("sampling_rate", 0) for a in all_audios if a.get("sampling_rate")]
        channels = [a.get("channels", 0) for a in all_audios if a.get("channels")]
        encodings = [a.get("encoding", "unknown") for a in all_audios if a.get("encoding")]
        
        result = {}
        if sample_rates:
            result["sampling_rates"] = {str(k): v for k, v in Counter(sample_rates).most_common()}
        if channels:
            result["channels"] = {str(k): v for k, v in Counter(channels).most_common()}
        if encodings:
            result["encodings"] = {str(k): v for k, v in Counter(encodings).most_common()}
        
        return result
    
    @staticmethod
    def _multi_audio_stats(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate multi-audio specific statistics."""
        multi_entries = [e for e in entries if e.is_multi_audio]
        single_entries = [e for e in entries if not e.is_multi_audio]
        
        if not multi_entries:
            return {}
        
        # Count audio files per multi-audio entry
        audio_counts = [len(e.get_audio_list()) for e in multi_entries]
        
        result = {
            "total_multi_audio_entries": len(multi_entries),
            "total_single_audio_entries": len(single_entries),
            "total_audio_files": sum(len(e.get_audio_list()) for e in entries),
            "audio_files_per_entry": {
                "min": int(np.min(audio_counts)),
                "max": int(np.max(audio_counts)),
                "mean": round(float(np.mean(audio_counts)), 2),
                "median": round(float(np.median(audio_counts)), 2)
            },
            "audio_count_distribution": dict(Counter(audio_counts).most_common())
        }
        
        return result
    
    @staticmethod
    def _caption_char_stats(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate caption character length statistics."""
        lengths = [len(e.text) for e in entries if e.text]
        if not lengths:
            return {}
        
        return {
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": round(float(np.mean(lengths)), 2),
            "std": round(float(np.std(lengths)), 2),
            "median": round(float(np.median(lengths)), 2),
            "percentiles": {
                "25": round(float(np.percentile(lengths, 25)), 2),
                "75": round(float(np.percentile(lengths, 75)), 2),
                "90": round(float(np.percentile(lengths, 90)), 2),
                "95": round(float(np.percentile(lengths, 95)), 2),
                "99": round(float(np.percentile(lengths, 99)), 2)
            }
        }
    
    @staticmethod
    def _caption_word_stats(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate caption word count statistics."""
        word_counts = [len(e.text.split()) for e in entries if e.text]
        if not word_counts:
            return {}
        
        return {
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "mean": round(float(np.mean(word_counts)), 2),
            "std": round(float(np.std(word_counts)), 2),
            "median": round(float(np.median(word_counts)), 2)
        }
    
    @staticmethod
    def _conversation_stats(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate caption word count statistics."""
        # print(entries[0])
        # sys.exit()
        n_rounds = [len(e.conversations) for e in entries if e.conversations]
        if not n_rounds:
            return {}
        
        return {
            "min": int(np.min(n_rounds)),
            "max": int(np.max(n_rounds)),
            "mean": round(float(np.mean(n_rounds)), 2),
            "std": round(float(np.std(n_rounds)), 2),
            "median": round(float(np.median(n_rounds)), 2)
        }
    
    @staticmethod
    def _segmentation_stats(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate segmentation statistics."""
        all_audios = ManifestStatistics._get_all_audio_dicts(entries)
        segmented = sum(1 for a in all_audios if a.get("offset", 0) > 0)
        total = len(all_audios)
        
        return {
            "total_segmented": segmented,
            "total_full_files": total - segmented,
            "segmentation_ratio": round(segmented / total, 3) if total > 0 else 0
        }
    
    @staticmethod
    def _quality_metrics(entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        all_audios = ManifestStatistics._get_all_audio_dicts(entries)
        caption_lengths = [len(e.text) for e in entries if e.text]
        
        # Count entries with missing duration (check total duration per entry)
        missing_duration = sum(1 for e in entries if e.get_total_duration() == 0)
        
        # Audio-level metrics
        short_audio = sum(1 for a in all_audios if 0 < a.get("duration", 0) < 1.0)
        long_audio = sum(1 for a in all_audios if a.get("duration", 0) > 600)
        byte_seekable = sum(1 for a in all_audios if a.get("encoding") in ["PCM_S", "PCM_F"])
        compressed = sum(1 for a in all_audios if a.get("encoding") not in ["PCM_S", "PCM_F", None])
        
        return {
            "empty_captions": sum(1 for e in entries if not e.text or len(e.text.strip()) == 0),
            "missing_audio_duration": missing_duration,
            "short_audio_below_1s": short_audio,
            "long_audio_above_600s": long_audio,
            "short_captions_below_10chars": sum(1 for c in caption_lengths if c < 10),
            "long_captions_above_200chars": sum(1 for c in caption_lengths if c > 200),
            "byte_seekable_wav": byte_seekable,
            "compressed_formats": compressed
        }
    
    @staticmethod
    def _dataset_distribution(entries: List[ManifestEntry]) -> Dict[str, int]:
        """Calculate dataset distribution."""
        datasets = [e.dataset for e in entries if e.dataset]
        if not datasets:
            return {}
        return {k: v for k, v in Counter(datasets).most_common()}
    
    @staticmethod
    def _count_speakers(entries: List[ManifestEntry]) -> int:
        """Count unique speakers."""
        speakers = set(e.speaker_id for e in entries if e.speaker_id)
        return len(speakers)


# =============================================================================
# 8. TARBALL CREATOR
# =============================================================================

def create_tarball_shard(args):
    """Create a single tarball shard (for multiprocessing).
    
    Supports both single-audio and multi-audio entries.
    
    Args:
        args: Tuple of (shard_entries, shard_idx, num_shards, prefix, 
                       manifest_output_dir, tarball_output_dir, audio_source_dir)
    
    Returns:
        Path to created manifest file or None on failure
    """
    shard_entries, shard_idx, num_shards, prefix, manifest_output_dir, tarball_output_dir, audio_source_dir = args
    
    shard_num_width = max(3, len(str(num_shards)))
    shard_name = f"{prefix}.{shard_idx + 1:0{shard_num_width}d}-of-{num_shards:0{shard_num_width}d}"
    tar_filename = f"{shard_name}.tar"
    tar_filepath = os.path.join(tarball_output_dir, tar_filename)
    output_manifest_path = os.path.join(manifest_output_dir, f"{shard_name}.ndjson")
    
    os.makedirs(manifest_output_dir, exist_ok=True)
    os.makedirs(tarball_output_dir, exist_ok=True)
    
    # Phase 1: Create de-duplicated tarball
    # Collect all unique audio paths from both single and multi-audio entries
    unique_files = set()
    for entry in shard_entries:
        for path in entry.get_all_paths():
            unique_files.add(path)
    
    added_files = set()
    
    try:
        with tarfile.open(tar_filepath, 'w') as tar_file:
            for file_path in unique_files:
                # If file_path is absolute, use it directly as source.
                # Otherwise join with audio_source_dir.
                if os.path.isabs(file_path):
                    full_audio_path = file_path
                else:
                    full_audio_path = os.path.join(audio_source_dir, file_path)
                
                if full_audio_path not in added_files:
                    if os.path.exists(full_audio_path):
                        tar_file.add(full_audio_path, arcname=file_path)
                        added_files.add(full_audio_path)
    except Exception as e:
        logger.error(f"Error creating tarball for shard {shard_name}: {e}")
        return None
    
    # Phase 2: Re-open tar to get offsets and write manifest
    try:
        with tarfile.open(tar_filepath, 'r') as tar_file:
            # Note: tarfile strips leading '/' from arcnames, so we need to handle both
            # /path/to/file.mp3 -> path/to/file.mp3 in the tarball
            member_map = {member.name: member for member in tar_file.getmembers()}
            
            def lookup_member(path):
                """Look up tar member, handling leading slash stripping."""
                if path in member_map:
                    return member_map[path]
                # Try without leading slash (tarfile strips it)
                stripped = path.lstrip('/')
                if stripped in member_map:
                    return member_map[stripped]
                return None
            
            with open(output_manifest_path, 'w') as f:
                for entry in shard_entries:
                    if entry.is_multi_audio:
                        # Multi-audio: update each audio dict in the list
                        for audio_dict in entry.audio:
                            audio_path = audio_dict.get('path')
                            member = lookup_member(audio_path) if audio_path else None
                            if member:
                                audio_dict['tar_path'] = tar_filename
                                audio_dict['tar_offset'] = member.offset_data
                                audio_dict['tar_size'] = member.size
                        f.write(json.dumps(entry.to_dict()) + '\n')
                    else:
                        # Single-audio: update the single audio dict
                        audio_path = entry.audio['path']
                        member = lookup_member(audio_path)
                        if member:
                            entry.audio['tar_path'] = tar_filename
                            entry.audio['tar_offset'] = member.offset_data
                            entry.audio['tar_size'] = member.size
                            f.write(json.dumps(entry.to_dict()) + '\n')
        return output_manifest_path
    except Exception as e:
        logger.error(f"Error generating manifest for shard {shard_name}: {e}")
        return None


class TarballCreator:
    """Create tarball datasets from manifest entries."""
    
    def __init__(self, num_workers: int = 4):
        """Initialize tarball creator.
        
        Args:
            num_workers: Number of parallel workers
        """
        self.num_workers = num_workers
    
    def create_tarball_dataset(
        self,
        entries: List[ManifestEntry],
        prefix: str,
        shard_size: int,
        manifest_output_dir: str,
        tarball_output_dir: str,
        audio_source_dir: str
    ) -> List[str]:
        """Create tarball dataset with sharded archives and manifests.
        
        Args:
            entries: List of manifest entries
            prefix: Output filename prefix
            shard_size: Number of entries per shard
            manifest_output_dir: Directory for output manifests
            tarball_output_dir: Directory for tarball files
            audio_source_dir: Source directory for audio files
            
        Returns:
            List of created manifest file paths
        """
        logger.info("Creating tarball dataset")
        
        # Shard entries
        num_entries = len(entries)
        num_shards = (num_entries + shard_size - 1) // shard_size
        shards = [entries[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
        
        # Prepare tasks
        tasks = [
            (shards[i], i, num_shards, prefix, manifest_output_dir, tarball_output_dir, audio_source_dir)
            for i in range(num_shards)
        ]
        
        # Process shards in parallel
        shard_files = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(create_tarball_shard, tasks),
                total=len(tasks),
                desc="Creating tarballs"
            ))
            shard_files = [res for res in results if res]
        
        logger.info(f"Created {len(shard_files)} tarball shards")
        return shard_files


# =============================================================================
# 9. MANIFEST WRITER
# =============================================================================

class ManifestWriter:
    """Write manifest files and index."""
    
    def write_sharded_manifests(
        self,
        entries: List[ManifestEntry],
        output_dir: Path,
        prefix: str,
        shard_size: int = 10000
    ) -> List[str]:
        """Write manifest entries to sharded NDJSON files.
        
        Args:
            entries: List of manifest entries
            output_dir: Output directory
            prefix: Filename prefix
            shard_size: Maximum entries per shard
            
        Returns:
            List of created shard file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_entries = len(entries)
        num_shards = (num_entries + shard_size - 1) // shard_size
        
        if num_shards == 0:
            logger.warning("No entries to write")
            return []
        
        logger.info(f"Writing {num_entries} entries to {num_shards} shards")
        
        shard_files = []
        shard_info = []
        width = max(3, len(str(num_shards)))
        
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, num_entries)
            shard_entries = entries[start_idx:end_idx]
            
            shard_num = f"{shard_idx + 1:0{width}d}"
            total_num = f"{num_shards:0{width}d}"
            filename = output_dir / f"{prefix}.{shard_num}-of-{total_num}.ndjson"
            
            # Calculate shard statistics (handles both single and multi-audio)
            shard_durations = [e.get_total_duration() for e in shard_entries if e.get_total_duration() > 0]
            all_audios = []
            for e in shard_entries:
                all_audios.extend(e.get_audio_list())
            shard_encodings = [a.get("encoding", "unknown") for a in all_audios if a.get("encoding")]
            
            shard_metadata = {
                "filename": filename.name,
                "num_examples": len(shard_entries),
                "total_duration_hours": round(sum(shard_durations) / 3600, 3) if shard_durations else 0,
            }
            
            if shard_encodings:
                encoding_counts = Counter(shard_encodings)
                shard_metadata["encodings"] = {k: v for k, v in encoding_counts.most_common()}
            
            if shard_durations:
                shard_metadata["audio_duration_seconds"] = {
                    "min": round(float(np.min(shard_durations)), 2),
                    "max": round(float(np.max(shard_durations)), 2),
                    "mean": round(float(np.mean(shard_durations)), 2),
                    "median": round(float(np.median(shard_durations)), 2)
                }
            
            # Write shard
            hasher = hashlib.md5()
            with open(filename, "w", encoding="utf-8") as f:
                for entry in shard_entries:
                    json_str = json.dumps(entry.to_dict(), ensure_ascii=False)
                    f.write(json_str + "\n")
                    hasher.update(json_str.encode('utf-8'))
            
            shard_metadata["size_bytes"] = filename.stat().st_size
            shard_metadata["checksum_md5"] = hasher.hexdigest()
            
            logger.info(f"Wrote {len(shard_entries)} entries to {filename}")
            shard_files.append(str(filename))
            shard_info.append(shard_metadata)
        
        # Write index file
        self.write_index_file(entries, output_dir, prefix, shard_info)
        
        return shard_files
    
    def rebuild_shard_info_from_files(
        self,
        manifest_dir: Path,
        entries: List[ManifestEntry],
        prefix: str,
        shard_size: int
    ) -> List[Dict[str, Any]]:
        """Rebuild shard info by reading actual manifest files.
        
        This is used when manifests are created by TarballCreator and we need
        to reconstruct the metadata for the index file.
        
        Args:
            manifest_dir: Directory containing manifest files
            entries: List of manifest entries
            prefix: Filename prefix
            shard_size: Shard size used
            
        Returns:
            List of shard metadata dictionaries
        """
        num_entries = len(entries)
        num_shards = (num_entries + shard_size - 1) // shard_size
        width = max(3, len(str(num_shards)))
        
        shard_info = []
        
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, num_entries)
            shard_entries = entries[start_idx:end_idx]
            
            shard_num = f"{shard_idx + 1:0{width}d}"
            total_num = f"{num_shards:0{width}d}"
            filename = f"{prefix}.{shard_num}-of-{total_num}.ndjson"
            filepath = manifest_dir / filename
            
            # Calculate shard statistics (handles both single and multi-audio)
            shard_durations = [e.get_total_duration() for e in shard_entries if e.get_total_duration() > 0]
            all_audios = []
            for e in shard_entries:
                all_audios.extend(e.get_audio_list())
            shard_encodings = [a.get("encoding", "unknown") for a in all_audios if a.get("encoding")]
            shard_caption_lengths = [len(e.text) if e.text else 0 for e in shard_entries]
            
            shard_metadata = {
                "filename": filename,
                "num_examples": len(shard_entries),
                "total_duration_hours": round(sum(shard_durations) / 3600, 3) if shard_durations else 0,
            }
            
            # Add encoding distribution
            if shard_encodings:
                encoding_counts = Counter(shard_encodings)
                shard_metadata["encodings"] = {k: v for k, v in encoding_counts.most_common()}
            
            # Add duration statistics
            if shard_durations:
                shard_metadata["audio_duration_seconds"] = {
                    "min": round(float(np.min(shard_durations)), 2),
                    "max": round(float(np.max(shard_durations)), 2),
                    "mean": round(float(np.mean(shard_durations)), 2),
                    "median": round(float(np.median(shard_durations)), 2)
                }
            
            # Add caption statistics
            if shard_caption_lengths and any(l > 0 for l in shard_caption_lengths):
                non_empty = [l for l in shard_caption_lengths if l > 0]
                if non_empty:
                    shard_metadata["caption_char_length"] = {
                        "min": int(np.min(non_empty)),
                        "max": int(np.max(non_empty)),
                        "mean": round(float(np.mean(non_empty)), 1),
                        "median": round(float(np.median(non_empty)), 1)
                    }
            
            # Get file size and checksum if file exists
            if filepath.exists():
                shard_metadata["size_bytes"] = filepath.stat().st_size
                
                # Calculate checksum
                hasher = hashlib.md5()
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        hasher.update(line.encode('utf-8'))
                shard_metadata["checksum_md5"] = hasher.hexdigest()
            else:
                shard_metadata["size_bytes"] = 0
                shard_metadata["checksum_md5"] = ""
            
            shard_info.append(shard_metadata)
        
        return shard_info
    
    def write_index_file(
        self,
        entries: List[ManifestEntry],
        output_dir: Path,
        prefix: str,
        shard_info: List[Dict]
    ):
        """Write comprehensive index file."""
        index_path = output_dir / f"{prefix}.index"
        
        # Calculate statistics
        stats = ManifestStatistics.calculate(entries)
        
        # Get unique datasets
        datasets = sorted(list(set(e.dataset for e in entries if e.dataset)))
        
        # Check if any entries are multi-audio
        has_multi_audio = any(e.is_multi_audio for e in entries)
        
        # Build features schema
        audio_schema = {
            "path": "str",
            "sampling_rate": "int",
            "channels": "int",
            "offset": "float",
            "duration": "float",
            "bytes_per_sample": "int",
            "encoding": "str",
            "data_offset": "int"
        }
        
        features = {
            "audio_id": "str",
            "text": "str",
            "audio": audio_schema if not has_multi_audio else f"Dict | List[Dict]",
            "dataset": "str",
            "original_file_id": "str"
        }
        
        if has_multi_audio:
            features["is_multi_audio"] = "bool"
        if any(e.speaker_id for e in entries):
            features["speaker_id"] = "str"
        if any(e.metadata for e in entries):
            features["metadata"] = "dict"
        
        # Calculate total duration and size (handles both single and multi-audio)
        total_duration = sum(e.get_total_duration() for e in entries)
        total_size = sum(s.get("size_bytes", 0) for s in shard_info)
        
        # Build index data
        index_data = {
            "features": features,
            "shards": shard_info,
            "num_examples": len(entries),
            "total_duration_hours": round(total_duration / 3600, 2),
            "total_size_bytes": total_size,
            "datasets": datasets,
            "dataset_distribution": stats['dataset_distribution'],
            "num_speakers": stats['num_speakers'],
            "caption_char_length": stats['caption_char_length'],
            "caption_word_count": stats['caption_word_count'],
            "audio_duration_seconds": stats['audio_duration'],
            "audio_metadata": stats['audio_metadata'],
            "segmentation": stats['segmentation'],
            "quality_metrics": stats['quality_metrics'],
            "processing_info": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "processor_version": "2.1",  # Updated version for multi-audio support
            },
            "manifest_version": "1.1"  # Updated for multi-audio support
        }
        
        # Add multi-audio stats if present
        if 'multi_audio' in stats:
            index_data['multi_audio'] = stats['multi_audio']
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"Wrote index file: {index_path}")


# =============================================================================
# 10. MANIFEST PIPELINE (Orchestrator)
# =============================================================================

class ManifestPipeline:
    """High-level orchestration of manifest creation pipeline."""
    
    def __init__(
        self,
        verify_audio: bool = True,
        auto_convert: bool = False,
        audio_dest_dir: Optional[str] = None,
        num_workers: int = 4,
        verbose: bool = False
    ):
        """Initialize pipeline.
        
        Args:
            verify_audio: Whether to verify audio and extract metadata
            auto_convert: Whether to auto-convert non-WAV files
            audio_dest_dir: Destination directory for converted audio
            num_workers: Number of parallel workers
            verbose: Enable verbose logging
        """
        self.verify_audio = verify_audio
        self.auto_convert = auto_convert
        self.audio_dest_dir = audio_dest_dir
        self.num_workers = num_workers
        self.verbose = verbose
        
        # Initialize components
        self.metadata_extractor = AudioMetadataExtractor(cache_enabled=True) if verify_audio else None
        self.path_resolver = PathResolver()
        
        if auto_convert:
            converter_config = AudioConverterConfig(
                num_workers=num_workers,
                verbose=verbose
            )
            self.converter = AudioConverter(converter_config)
        else:
            self.converter = None
        
        self.entry_processor = EntryProcessor(
            self.metadata_extractor if self.metadata_extractor else AudioMetadataExtractor(cache_enabled=False),
            self.path_resolver,
            verbose=verbose,
            auto_convert=auto_convert,
            audio_dest_dir=audio_dest_dir
        )
        
        self.manifest_loader = ManifestLoader()
        self.manifest_writer = ManifestWriter()
    
    def run(
        self,
        input_path: Optional[str],
        config: ProcessingConfig
    ) -> List[ManifestEntry]:
        """Run the complete manifest processing pipeline.
        
        Args:
            input_path: Path to input manifest (or None for audio-only mode)
            config: Processing configuration
            
        Returns:
            List of processed manifest entries
        """
        # Phase 1: Load raw entries
        raw_entries = self._phase1_load_entries(input_path, config)
        
        # Phase 2: Audio source configuration
        logger.info("=" * 60)
        logger.info("PHASE 2: Audio source configuration")
        logger.info("=" * 60)
        
        if self.auto_convert and config.audio_source_dir and self.audio_dest_dir:
            logger.info("Audio conversion: ENABLED")
            logger.info(f"Source: {config.audio_source_dir}")
            logger.info(f"Destination: {self.audio_dest_dir}/{config.effective_dataset_dir}")
            self._phase2_convert_audio(raw_entries, config)
            logger.info("Audio conversion complete")
        else:
            logger.info("Audio conversion: DISABLED")
            if config.audio_source_dir:
                logger.info(f"Using source audio directly from: {config.audio_source_dir}")
            else:
                logger.info("Using audio paths from manifest")
        
        # Phase 3: Process entries and build manifest
        processed_entries = self._phase3_process_entries(raw_entries, config)
        
        return processed_entries
    
    def _phase1_load_entries(
        self,
        input_path: Optional[str],
        config: ProcessingConfig
    ) -> List[Dict[str, Any]]:
        """Phase 1: Load raw entries from file or directory."""
        logger.info("=" * 60)
        logger.info("PHASE 1: Loading entries")
        logger.info("=" * 60)
        
        if input_path:
            raw_entries = self.manifest_loader.load_from_file(
                input_path,
                config.location_key or "location",
                config.caption_key or "text",
                config.conversation_key or "conversations"
            )
        elif config.audio_source_dir:
            raw_entries = self.manifest_loader.load_from_directory(
                config.audio_source_dir,
                config.location_key or "location",
                config.caption_key or "text",
                config.conversation_key or "conversations"
            )
        else:
            raise ValueError("Either input manifest or audio_source_dir must be provided")
        
        logger.info(f"Loaded {len(raw_entries)} entries")
        return raw_entries
    
    def _phase2_convert_audio(
        self,
        raw_entries: List[Dict],
        config: ProcessingConfig
    ):
        """Phase 2: Convert unique audio files."""
        logger.info("=" * 60)
        logger.info("PHASE 2: Converting audio files")
        logger.info("=" * 60)
        
        # Extract unique files (handles both single and multi-audio)
        unique_files = set()
        loc_key = config.location_key or "location"
        for entry in raw_entries:
            location = entry.get(loc_key)
            if location:
                if config.is_multi_audio and isinstance(location, list):
                    # Multi-audio: add all paths in the list
                    for loc in location:
                        if loc:
                            unique_files.add(loc)
                else:
                    # Single audio
                    unique_files.add(location)
        
        # Convert
        converted, skipped, failed = self.converter.convert_batch(
            list(unique_files),
            config.audio_source_dir,
            self.audio_dest_dir,
            config.effective_dataset_dir
        )
        
        logger.info(f"Conversion complete: {converted} converted, {skipped} skipped, {failed} failed")
    
    def _handle_caption_duplicates(
        self,
        results: List[Optional[ManifestEntry]],
        raw_entries: List[Dict],
        config: ProcessingConfig
    ) -> List[ManifestEntry]:
        """Handle multiple captions for same audio file.
        
        Args:
            results: Processed manifest entries (may contain None)
            raw_entries: Original raw entries
            config: Processing configuration
            
        Returns:
            List of processed entries with unique audio_ids
        """
        file_caption_tracker = {}
        processed_entries = []
        loc_key = config.location_key or "location"
        
        for result, original_entry in zip(results, raw_entries):
            if not result:
                continue
            
            # Build file key for de-duplication
            if result.is_multi_audio:
                # For multi-audio, use all paths joined as key
                paths = result.get_all_paths()
                file_key = "_".join(sorted(paths))
            else:
                # For single audio, use path and offset
                file_key = f"{original_entry.get(loc_key, '')}_{result.audio.get('offset', 0)}"
            
            if file_key in file_caption_tracker:
                caption_idx = file_caption_tracker[file_key]
                result.audio_id = f"{result.audio_id}_{caption_idx}"
                file_caption_tracker[file_key] += 1
            else:
                file_caption_tracker[file_key] = 1
            
            processed_entries.append(result)
        
        return processed_entries
    
    def _phase3_process_entries(
        self,
        raw_entries: List[Dict],
        config: ProcessingConfig
    ) -> List[ManifestEntry]:
        """Phase 3: Process entries and build manifest."""
        logger.info("=" * 60)
        logger.info("PHASE 3: Processing entries")
        logger.info("=" * 60)
        
        # Process entries
        if self.num_workers > 1:
            # Parallel processing
            process_fn = partial(
                self.entry_processor.process_entry,
                config=config
            )
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(tqdm(
                    executor.map(process_fn, raw_entries),
                    total=len(raw_entries),
                    desc="Processing entries",
                    unit="entries"
                ))
        else:
            # Sequential processing
            results = []
            for entry in tqdm(raw_entries, desc="Processing entries", unit="entries"):
                result = self.entry_processor.process_entry(entry, config)
                results.append(result)
        
        # Handle caption duplicates
        processed_entries = self._handle_caption_duplicates(results, raw_entries, config)
        
        # Report rejection statistics
        rejected_count = sum(1 for r in results if r is None)
        if rejected_count > 0:
            logger.warning(
                f"⚠️  {rejected_count}/{len(raw_entries)} entries rejected due to missing/invalid audio metadata. "
                f"Check logs above for details."
            )
        
        logger.info(f"Successfully processed {len(processed_entries)} entries (rejected {rejected_count})")
        return processed_entries
    
    def get_tarball_audio_source(self, config: ProcessingConfig) -> str:
        """Get the audio source directory for tarball creation.
        
        Args:
            config: Processing configuration
            
        Returns:
            Base directory for audio files (entry paths include subdirectories)
        """
        if self.auto_convert and self.audio_dest_dir:
            # Converted files are in audio_dest_dir (entry paths include dataset subdirectory)
            return self.audio_dest_dir
        else:
            # Original files in source directory
            return config.audio_source_dir


# =============================================================================
# 11. MAIN & CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create sharded, tarball-based datasets (refactored production version)"
    )
    
    # Input/Output
    parser.add_argument(
        "--input", 
        type=str, 
        help="Input file path: .jsonl/.json (manifest with captions), .txt (list of audio paths), or omit to scan --audio_source_dir"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for manifests")
    parser.add_argument("--dataset_name", type=str, help="Dataset name")
    
    # Dataset config
    parser.add_argument("--location_key", type=str, default="location", help="Key for audio location")
    parser.add_argument("--caption_key", type=str, default="text", help="Key for caption/text")
    parser.add_argument("--conversation_key", type=str, default="text", help="Key for conversations")
    parser.add_argument("--audio_source_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--is_multi_audio", action="store_true", 
                        help="Treat location_key as list of audio paths (multi-audio mode)")
    
    # S3 config
    parser.add_argument("--s3_bucket", type=str, help="S3 bucket name")
    parser.add_argument("--s3_prefix", type=str, help="S3 prefix path")
    parser.add_argument("--s3_dataset_dir", type=str, help="Dataset-specific S3 directory")
    parser.add_argument("--convert_to_s3", action="store_true", help="Convert paths to S3")
    
    # Processing options
    parser.add_argument("--shard_size", type=int, default=10000, help="Entries per shard")
    parser.add_argument("--verify_audio", action="store_true", help="Verify audio and extract metadata")
    parser.add_argument("--auto_convert_wav", action="store_true", help="Auto-convert to WAV")
    parser.add_argument("--audio_dest_dir", type=str, help="Destination for converted audio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--verbose_logging", action="store_true", help="Enable verbose logging")
    
    # Tarball options
    parser.add_argument("--create_tarballs", action="store_true", help="Create tarball dataset")
    parser.add_argument("--tarball_output_dir", type=str, help="Output directory for tarballs")
    parser.add_argument("--prefix", type=str, help="Output filename prefix")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Infer dataset name if not provided
    if not args.dataset_name:
        if args.input:
            args.dataset_name = Path(args.input).stem
        else:
            args.dataset_name = "dataset"
        logger.info(f"Inferred dataset name: {args.dataset_name}")
    
    # Set prefix
    if not args.prefix:
        args.prefix = args.dataset_name
    
    # Set tarball output dir
    if args.create_tarballs and not args.tarball_output_dir:
        args.tarball_output_dir = args.output_dir
    
    # Warning about auto_convert_wav being resource-heavy
    if args.auto_convert_wav:
        logger.warning(
            "⚠️  auto_convert_wav is ENABLED. This is resource-intensive and may not be necessary.\n"
            "    Our VirtualFileSection loader now supports arbitrary audio formats (MP3, FLAC, etc.)\n"
            "    directly from tarballs without WAV conversion. Consider disabling auto_convert_wav\n"
            "    to save disk space and processing time."
        )
    
    # Log multi-audio mode
    if args.is_multi_audio:
        logger.info("🎵 Multi-audio mode ENABLED: location_key will be treated as list of audio paths")
    
    # Initialize pipeline
    pipeline = ManifestPipeline(
        verify_audio=args.verify_audio,
        auto_convert=args.auto_convert_wav,
        audio_dest_dir=args.audio_dest_dir,
        num_workers=args.num_workers,
        verbose=args.verbose_logging
    )
    
    # Create processing config
    config = ProcessingConfig(
        dataset_name=args.dataset_name,
        location_key=args.location_key,
        caption_key=args.caption_key,
        conversation_key=args.conversation_key,
        convert_to_s3=args.convert_to_s3,
        dataset_dir=args.s3_dataset_dir,
        audio_source_dir=args.audio_source_dir,
        is_multi_audio=args.is_multi_audio
    )
    
    # Run pipeline
    entries = pipeline.run(args.input, config)
    
    if not entries:
        logger.error("No valid entries found!")
        sys.exit(1)
    
    # Determine the actual audio source directory for tarball creation
    actual_audio_source = pipeline.get_tarball_audio_source(config)
    logger.info(f"Audio source for tarballs: {actual_audio_source}")
    
    # Write output
    output_dir = Path(args.output_dir)
    
    if args.create_tarballs:
        # Create tarball dataset
        logger.info("=" * 60)
        logger.info("PHASE 4: Creating tarball dataset")
        logger.info("=" * 60)
        
        manifest_output_dir = os.path.join(args.output_dir, args.dataset_name)
        tarball_output_dir = os.path.join(args.tarball_output_dir, args.dataset_name)
        
        tarball_creator = TarballCreator(num_workers=args.num_workers)
        shard_files = tarball_creator.create_tarball_dataset(
            entries,
            args.prefix,
            args.shard_size,
            manifest_output_dir,
            tarball_output_dir,
            actual_audio_source
        )
        
        # Reload entries from created manifests to get tarball info
        final_entries = []
        for manifest_path in shard_files:
            with open(manifest_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    final_entries.append(ManifestEntry(**data))
        
        # Rebuild shard info from actual manifest files
        shard_info = pipeline.manifest_writer.rebuild_shard_info_from_files(
            Path(manifest_output_dir),
            final_entries,
            args.prefix,
            args.shard_size
        )
        
        # Write index with proper shard info
        pipeline.manifest_writer.write_index_file(
            final_entries,
            Path(manifest_output_dir),
            args.prefix,
            shard_info
        )

        # --- Self-Verification ---
        print("\n" + "=" * 60)
        print("🔍 SELF-VERIFICATION")
        print("=" * 60)
        try:
            # Import standalone loader dynamically to avoid top-level import issues if not present
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from standalone_loader import SimpleAudioLoader
            
            # Use the first shard manifest for verification
            if shard_files:
                manifest_to_verify = shard_files[0]
                print(f"Verifying first shard: {manifest_to_verify}")
                
                with open(manifest_to_verify, 'r') as f:
                    verify_entries = [json.loads(line) for line in f if line.strip()]
                
                if verify_entries:
                    # Check first 5 entries
                    loader = SimpleAudioLoader(sampling_rate=48000)
                    
                    def prepare_entry_for_verification(entry, tarball_dir):
                        """Prepare entry for verification by resolving tar paths."""
                        test_entry = json.loads(json.dumps(entry))
                        is_multi_audio = test_entry.get('is_multi_audio', False)
                        
                        # Also infer from audio type if flag not set
                        if not is_multi_audio and isinstance(test_entry.get('audio'), list):
                            is_multi_audio = True
                        
                        if is_multi_audio and isinstance(test_entry.get('audio'), list):
                            # Multi-audio: update each audio dict
                            has_tar = False
                            for audio_dict in test_entry['audio']:
                                if 'tar_path' in audio_dict:
                                    audio_dict['tar_path'] = os.path.join(tarball_dir, audio_dict['tar_path'])
                                    has_tar = True
                            if has_tar:
                                test_entry['storage_backend'] = 'tarball_lustre'
                            else:
                                test_entry['storage_backend'] = 'lustre'
                        elif isinstance(test_entry.get('audio'), dict):
                            # Single-audio
                            if 'tar_path' in test_entry['audio']:
                                test_entry['audio']['tar_path'] = os.path.join(tarball_dir, test_entry['audio']['tar_path'])
                                test_entry['storage_backend'] = 'tarball_lustre'
                            else:
                                test_entry['storage_backend'] = 'lustre'
                        
                        return test_entry
                    
                    verified_count = 0
                    for i, entry in enumerate(verify_entries[:5]):
                        try:
                            test_entry = prepare_entry_for_verification(entry, tarball_output_dir)
                            
                            # Attempt load
                            result = loader.load_audio(test_entry)
                            if result and result.get("wav") is not None:
                                wav = result["wav"]
                                if result.get("is_multi_audio", False) and isinstance(wav, list):
                                    shapes = [w.shape for w in wav]
                                    print(f"  ✅ Entry {i+1}: Multi-audio loaded ({len(wav)} files, shapes: {shapes})")
                                else:
                                    print(f"  ✅ Entry {i+1}: Audio loaded successfully (shape: {wav.shape})")
                                verified_count += 1
                            else:
                                print(f"  ❌ Entry {i+1}: Failed to load audio")
                        except Exception as e:
                            print(f"  ❌ Entry {i+1}: Error: {e}")
                    
                    if verified_count == min(5, len(verify_entries)):
                        print("\n✨ Verification successful! Sample entries are readable.")
                    else:
                        print("\n⚠️  Verification had failures. Please check paths and file accessibility.")
                else:
                    print("No entries to verify.")
        except ImportError:
            print("Could not import standalone_loader for verification. Skipping.")
        except Exception as e:
            print(f"Verification failed with error: {e}")
        print("=" * 60)
    else:
        # Write sharded manifests only
        shard_files = pipeline.manifest_writer.write_sharded_manifests(
            entries,
            output_dir,
            args.prefix,
            args.shard_size
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("MANIFEST CREATION COMPLETE")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Total entries: {len(entries)}")
    print(f"Shards created: {len(shard_files) if shard_files else 0}")
    print(f"Dataset: {args.dataset_name}")
    if entries:
        print("\nSample entry:")
        sample = entries[0].to_dict()
        sample_str = json.dumps(sample, indent=2)
        print(sample_str)
    print("=" * 60)


if __name__ == "__main__":
    main()

