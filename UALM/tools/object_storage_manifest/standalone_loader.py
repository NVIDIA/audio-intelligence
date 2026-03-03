# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

"""
Standalone audio loader for object storage datasets.
Extracted from stable_audio_tools to allow independent testing of manifests.
"""

import os
import io
import random
import logging
import json
from typing import Any, Dict, Generator, Set, Tuple, Optional, Union

import numpy as np
import soundfile as sf
import librosa

# Optional dependencies
try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    import sphn
except ImportError:
    sphn = None

logger = logging.getLogger(__name__)

# ==============================================================================
# Audio Utils
# ==============================================================================

# TypeAlias for cleaner function signatures
NumpyAudioDType = type[Union[np.signedinteger, np.floating]]
DTypeAndFactor = Tuple[NumpyAudioDType, float]

# Mappings for data types based on byte depth
PCM_S_DTYPE_MAP: Dict[int, NumpyAudioDType] = {
    1: np.int8,
    2: np.int16,
    3: np.int32,  # 24-bit audio is typically read into 32-bit integers
    4: np.int32,
    8: np.int64,
}

PCM_F_DTYPE_MAP: Dict[int, NumpyAudioDType] = {
    4: np.float32,
    8: np.float64,
}

# Supported audio formats
BYTE_SEEKABLE_FORMATS: Set[str] = {"wav", "raw", "pcm"}

def is_byte_seekable_format(format_str: str) -> bool:
    """Check if the audio format supports byte-range seeking."""
    if format_str:
        return format_str.lower() in BYTE_SEEKABLE_FORMATS
    return False

def get_format_from_path(path: str) -> str:
    """Extract audio format from file path."""
    if '.' in path:
        return path.rsplit('.', 1)[-1].lower()
    return 'wav'  # Default to WAV if no extension

def get_audio_processing_params(encoding: str, bytes_per_sample: int) -> DTypeAndFactor:
    """
    Determines the NumPy dtype and normalization factor from audio metadata.
    """
    if encoding == "PCM_S":
        dtype = PCM_S_DTYPE_MAP.get(bytes_per_sample)
        if dtype is None:
            raise ValueError(f"Unsupported byte depth for signed PCM: {bytes_per_sample}")
        # Normalization factor for signed integers is 2^(bits - 1)
        norm_factor = 2.0 ** (bytes_per_sample * 8 - 1)

    elif encoding == "PCM_F":
        dtype = PCM_F_DTYPE_MAP.get(bytes_per_sample)
        if dtype is None:
            raise ValueError(f"Unsupported byte depth for float PCM: {bytes_per_sample}")
        # Floating point audio is already in the [-1.0, 1.0] range
        norm_factor = 1.0

    else:
        raise ValueError(f"Unsupported audio encoding: {encoding}")

    return dtype, norm_factor

# ==============================================================================
# VirtualFileSection Object: supports soundfile.read() directly on the tarball file handle for non-wav files (e.g. mp3)
# ==============================================================================

class VirtualFileSection:
    """
    A file-like object that restricts access to a specific section of a real file.
    It supports seeking relative to the section start.
    """
    def __init__(self, file_obj, start_offset, length):
        self.file_obj = file_obj
        self.start_offset = start_offset
        self.length = length
        self.end_offset = start_offset + length
        # Initialize position at the start of the section
        self.file_obj.seek(self.start_offset)

    def read(self, size=-1):
        current_pos = self.file_obj.tell()
        
        # Calculate how many bytes are left in our section
        bytes_left = self.end_offset - current_pos
        
        if bytes_left <= 0:
            return b""
            
        # If size is -1 (read all) or greater than what's left, cap it
        if size < 0 or size > bytes_left:
            size = bytes_left
            
        return self.file_obj.read(size)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            # Seek relative to the start of our section
            target = self.start_offset + offset
        elif whence == os.SEEK_CUR:
            # Seek relative to current real position
            target = self.file_obj.tell() + offset
        elif whence == os.SEEK_END:
            # Seek relative to the end of our section
            target = self.end_offset + offset
        else:
            raise ValueError(f"Invalid whence argument: {whence}")

        # (Optional) Clamp target to bounds if strictly necessary, 
        # though standard files allow seeking past bounds.
        # For safety with decoding libraries, we simply perform the seek.
        self.file_obj.seek(target)
        return self.tell()

    def tell(self):
        # Report position relative to our virtual start
        return self.file_obj.tell() - self.start_offset

    def flush(self):
        pass
        
    def close(self):
        # Do not close the underlying file, as it might be used elsewhere
        pass
    

# ==============================================================================
# Simple Audio Loader
# ==============================================================================

class SimpleAudioLoader:
    """
    A standalone class that handles audio loading from various sources (S3, Lustre, Tarballs).
    Replicates the core logic of AudioPreprocessor without stable_audio_tools dependencies.
    """

    def __init__(self, sampling_rate: int = 48000, seed: int = 42, storage_backend: str = "tarball_lustre", s3_client_config: Optional[Dict] = None):
        self.sampling_rate = sampling_rate
        self.rng = random.Random(seed)
        self.storage_backend = storage_backend
        
        # Initialize client placeholder to None.
        self.s3_client = None
        self.s3_client_config = s3_client_config
        self.tar_handles = {} # TarFile or raw file object handles
        self.random_crop_sample_size = None # Set manually if needed

    def _init_s3_client(self):
        if self.s3_client is None and boto3 is not None:
            logger.debug("Initializing Boto3 S3 client...")
            
            # Default config
            boto_config_args = {
                'signature_version': 's3v4',
                's3': {'use_accelerate_endpoint': False}
            }
            
            # Override if s3_client_config provided
            if self.s3_client_config:
                # Merge configs if needed, for now just passing what's there if compatible
                pass

            boto_config = BotoConfig(**boto_config_args)
            
            client_args = {
                "service_name": "s3",
                "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "region_name": os.getenv("AWS_REGION"),
                "config": boto_config
            }
            
            # Apply overrides from s3_client_config
            if self.s3_client_config:
                if "endpoint_url" in self.s3_client_config:
                    client_args["endpoint_url"] = self.s3_client_config["endpoint_url"]
                if "profile_name" in self.s3_client_config:
                    # If profile is specified, we might need a session
                    pass 

            self.s3_client = boto3.client(**client_args)
        elif boto3 is None:
            logger.warning("Boto3 not installed, S3 functionality will fail.")

    def _get_frames_to_read(
        self, 
        original_samplerate: int,
        duration_sec: float,
        offset_sec: float = 0.0
    ) -> tuple[int, int]:
        """
        Calculate start frame and number of frames to read.
        
        Automatically applies random crop if self.random_crop_sample_size is set.
        This helper function is shared across all audio loading methods.
        
        Args:
            original_samplerate: Sample rate of the audio file
            duration_sec: Total duration of audio available in seconds
            offset_sec: Existing offset into the audio in seconds (default: 0.0)
            
        Returns:
            Tuple of (start_frame, frames_to_read) where:
            - start_frame: Frame offset to start reading from
            - frames_to_read: Number of frames to read, or -1 for all remaining
        """
        # If random crop is not enabled, return original behavior
        if self.random_crop_sample_size is None:
            start_frame = int(offset_sec * original_samplerate)
            frames_to_read = int(duration_sec * original_samplerate) if duration_sec is not None else -1
            return start_frame, frames_to_read
        
        # Calculate available and needed samples at original sample rate
        total_frames_available = int(duration_sec * original_samplerate)
        # Account for resampling: calculate target frames at original sample rate
        target_frames_at_original_sr = int(self.random_crop_sample_size * original_samplerate / self.sampling_rate)
        
        # Start with the existing offset
        start_frame = int(offset_sec * original_samplerate)
        
        # Only apply random crop if audio is longer than needed
        if total_frames_available > target_frames_at_original_sr:
            # Random crop: pick a random start position within available range
            max_start_frame = total_frames_available - target_frames_at_original_sr
            crop_start_frame = self.rng.randint(0, max_start_frame)
            
            start_frame += crop_start_frame
            frames_to_read = target_frames_at_original_sr
            
        else:
            # Audio is shorter than or equal to target, load all
            frames_to_read = total_frames_available
        
        return start_frame, frames_to_read
    
    def _load_s3_wav_with_byte_seeking(self, bucket: str, key: str, audio_metadata: dict) -> np.ndarray:
        """
        Load WAV audio from S3 using efficient byte-range requests.
        
        This method performs precise byte-seeking for WAV files, downloading only
        the exact audio segment needed. This is highly efficient for large files
        and random access patterns.
        
        With random crop support (when self.random_crop_sample_size is set):
        - If audio is longer than needed, randomly selects a crop window
        - Only downloads that portion (precise byte-range request)
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            audio_metadata: Dict containing sampling_rate, channels, bytes_per_sample,
                          encoding, data_offset, offset, and duration
        
        Returns:
            numpy array of shape [channels, time] with audio data normalized to [-1, 1]
        
        Raises:
            ValueError: If duration is invalid (<=0)
            Exception: If S3 read fails
        """
        sampling_rate = audio_metadata["sampling_rate"]
        channels = audio_metadata["channels"]
        bytes_per_sample = audio_metadata["bytes_per_sample"]
        encoding = audio_metadata["encoding"]
        data_offset = audio_metadata["data_offset"]
        offset_sec = audio_metadata["offset"]
        duration_sec = audio_metadata["duration"]
        
        # Define data type and normalization factor based on byte depth
        dtype, norm_factor = get_audio_processing_params(encoding, bytes_per_sample)
        
        # Calculate byte range for the audio segment (with automatic random crop if enabled)
        frame_size_bytes = channels * bytes_per_sample
        start_sample, num_samples = self._get_frames_to_read(
            sampling_rate, duration_sec, offset_sec
        )
        
        if num_samples <= 0:
            raise ValueError(
                f"Invalid duration for S3 audio: s3://{bucket}/{key}\n"
                f"start_sec: {offset_sec}, duration_sec: {duration_sec}"
            )
        
        start_byte = data_offset + start_sample * frame_size_bytes
        duration_bytes = num_samples * frame_size_bytes
        end_byte = start_byte + duration_bytes - 1
        byte_range = f"bytes={start_byte}-{end_byte}"
        
        # Download the specific byte range from S3
        response = self.s3_client.get_object(Bucket=bucket, Key=key, Range=byte_range)
        segment_data = response["Body"].read()
        
        # Convert raw bytes to normalized float numpy array
        array = np.frombuffer(segment_data, dtype=dtype)
        array = array.reshape(-1, channels).T  # [channels, time]
        array = array.astype(np.float32) / norm_factor
        
        # Resample if needed using high quality resampler
        if sampling_rate != self.sampling_rate:
            array = librosa.resample(
                array, orig_sr=sampling_rate, target_sr=self.sampling_rate,
                res_type="soxr_vhq"  # Very high quality resampling
            )
        
        return array

    def _load_tarball_local_audio_byteseek(self, audio_metadata: dict) -> np.ndarray:
        """
        [OPTIMIZED] Load a sub-segment of audio from a local TAR archive using direct byte seeking with soundfile.

        This is the most performant method for local tarballs. It works by:
        1. Caching the raw file handle (`open(path, 'rb')`) for the tarball.
        2. Creating a `VirtualFileSection` wrapper that presents only the relevant file segment.
        3. Calling `soundfile.read()` on this wrapper. This works for both WAV and compressed
           formats like MP3/FLAC, as soundfile can seek within the wrapper to decode.
        
        With random crop support (when self.random_crop_sample_size is set):
        - If audio is longer than needed, randomly selects a crop window
        - Only loads that portion from tarball (byte-level seeking)
        - Significantly reduces memory usage and loading time for long audio files
        
        Args:
            audio_metadata: Dict containing tar_path, tar_offset, tar_size, and all other
                          audio properties for byte calculation.

        Returns:
            numpy array of shape [channels, time] with audio data normalized to [-1, 1]
        """
        tar_path = audio_metadata["tar_path"]
        offset = audio_metadata["tar_offset"]
        tar_size = audio_metadata["tar_size"]
        
        # Get audio format details from manifest for direct reading
        samplerate = audio_metadata["sampling_rate"]

        # Check if file handle is cached for this worker
        if tar_path not in self.tar_handles:
            if not os.path.exists(tar_path):
                raise FileNotFoundError(f"Tarball not found: {tar_path}")
            # Cache the raw file handle
            self.tar_handles[tar_path] = open(tar_path, 'rb')

        handle = self.tar_handles[tar_path]

        # Create a VirtualFileSection view for the embedded audio file
        # This supports both WAV and non-WAV (MP3, FLAC, etc.) by providing a virtual file interface
        file_view = VirtualFileSection(handle, offset, tar_size)

        # Calculate start and number of frames to read (with automatic random crop if enabled)
        duration_sec = audio_metadata.get("duration")
        offset_sec = audio_metadata.get("offset", 0.0)
        start_frame, frames_to_read = self._get_frames_to_read(
            samplerate, duration_sec, offset_sec
        )

        # Use soundfile.read directly on the file view.
        # soundfile reads the header to determine format, so we don't pass format, samplerate, or channels.
        try:
            array, read_samplerate = sf.read(
                file_view, 
                start=start_frame, 
                frames=frames_to_read, 
                dtype='float32', 
                always_2d=True
            )
        except Exception as e:
            # Enhance error message with context
            raise RuntimeError(f"Failed to decode audio from tarball {tar_path} at offset {offset}: {e}")

        array = array.T # soundfile.read returns (frames, channels), we need (channels, frames)
        
        # Resample if needed
        if read_samplerate != self.sampling_rate:
            array = librosa.resample(
                array, orig_sr=read_samplerate, target_sr=self.sampling_rate,
                res_type="soxr_vhq"
            )
        
        return array

    def _load_tarball_s3_audio_byteseek(self, audio_metadata: dict) -> np.ndarray:
        """
        [OPTIMIZED] Load a sub-segment of audio from an S3-hosted TAR archive using a single, precise byte-range request.

        This is the most performant method for S3 tarballs. It works by:
        1. Calculating the absolute byte range of the final audio *sub-segment*
           by combining the tar_offset, the audio file's internal data_offset (e.g., for WAV headers),
           and the sub-segment's frame offset and duration.
        2. Making a single S3 `GetObject` request with this precise byte range.
        3. Converting the resulting raw bytes directly into a NumPy array.

        This approach minimizes both network latency (one request) and data transfer.
        
        With random crop support (when self.random_crop_sample_size is set):
        - If audio is longer than needed, randomly selects a crop window
        - Only downloads that portion (precise byte-range request)
        - Significantly reduces S3 data transfer and loading time
        
        Args:
            audio_metadata: Dict containing tar_path (s3://bucket/key.tar), 
                          tar_offset, tar_size, and all other properties.
        
        Returns:
            numpy array of shape [channels, time] with audio data normalized to [-1, 1]
        
        Raises:
            Exception: If S3 requests or audio decoding fails.
        """
        s3_path = audio_metadata["tar_path"]
        tar_offset = audio_metadata["tar_offset"]
        
        # Get audio format details from manifest for precise byte calculation
        sampling_rate = audio_metadata["sampling_rate"]
        channels = audio_metadata["channels"]
        bytes_per_sample = audio_metadata["bytes_per_sample"]
        encoding = audio_metadata["encoding"]
        data_offset = audio_metadata.get("data_offset", 0) # Offset within the audio file (e.g., WAV header)
        offset_sec = audio_metadata.get("offset", 0.0)
        duration_sec = audio_metadata.get("duration")

        _, _, bucket, key = s3_path.split("/", 3)

        # Calculate the precise byte range for the sub-segment within the S3 object.
        # This allows us to fetch only the needed audio data in a single request.
        dtype, norm_factor = get_audio_processing_params(encoding, bytes_per_sample)
        frame_size_bytes = channels * bytes_per_sample
        
        # Get start sample and number of samples (with automatic random crop if enabled)
        start_sample, num_samples = self._get_frames_to_read(
            sampling_rate, duration_sec, offset_sec
        )
        
        # Handle edge case: no duration specified
        if num_samples == -1:
            # Calculate from tar_size
            total_samples = (audio_metadata["tar_size"] - data_offset) // frame_size_bytes
            num_samples = total_samples - start_sample
        
        if num_samples <= 0:
            raise ValueError(f"Invalid duration for S3 audio: {s3_path}, num_samples={num_samples}")

        # tar_offset (start of audio data in tar) + 
        # data_offset (WAV header etc.) + 
        # start_sample offset
        start_byte = tar_offset + data_offset + (start_sample * frame_size_bytes)
        duration_bytes = num_samples * frame_size_bytes
        end_byte = start_byte + duration_bytes - 1
        
        data_range = f"bytes={start_byte}-{end_byte}"
        
        # Perform the single, targeted GET request
        data_response = self.s3_client.get_object(Bucket=bucket, Key=key, Range=data_range)
        segment_data = data_response["Body"].read()

        # Convert raw bytes to normalized float numpy array
        array = np.frombuffer(segment_data, dtype=dtype)
        array = array.reshape(-1, channels).T  # [channels, time]
        array = array.astype(np.float32) / norm_factor

        # Resample if needed
        if sampling_rate != self.sampling_rate:
            array = librosa.resample(
                array, orig_sr=sampling_rate, target_sr=self.sampling_rate,
                res_type="soxr_vhq"
            )
            
        return array
    
    def _load_s3_compressed_audio(self, bucket: str, key: str, audio_metadata: dict, format_str: str) -> np.ndarray:
        """
        Load compressed audio from S3 with full download and decoding.
        
        This method downloads the entire file and uses pydub for decoding.
        Supports formats like AAC, MP3, OGG, FLAC, etc. Less efficient than
        byte-seeking but necessary for compressed formats.
        
        With random crop support (when self.random_crop_sample_size is set):
        - If audio is longer than needed, randomly selects a crop window
        - Applies crop after decoding (still needs full download for compressed formats)
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            audio_metadata: Dict optionally containing offset and duration for segments
            format_str: Audio format string (e.g., 'aac', 'mp3')
        
        Returns:
            numpy array of shape [channels, time] with audio data normalized to [-1, 1]
        
        Raises:
            Exception: If S3 download or audio decoding fails
        """
        # Download the entire file (required for compressed formats)
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        audio_data = response["Body"].read()
        
        # Decode using pydub
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_data),
            format=format_str
        )
        
        # Get original offset and duration
        offset_sec = audio_metadata.get("offset", 0.0)
        duration_sec = audio_metadata.get("duration")
        
        # Apply random crop if enabled (automatic check)
        # Note: For compressed formats, we work in seconds since we decode first
        if self.random_crop_sample_size is not None and duration_sec is not None:
            target_duration_sec = self.random_crop_sample_size / self.sampling_rate
            
            if duration_sec > target_duration_sec:
                # Random crop: pick random start position
                max_start_sec = duration_sec - target_duration_sec
                crop_start_sec = self.rng.random() * max_start_sec
                
                offset_sec += crop_start_sec
                duration_sec = target_duration_sec
                
                logger.debug(
                    f"Random crop (S3 compressed): {audio_metadata.get('duration', 0):.2f}s → {target_duration_sec:.2f}s "
                    f"(crop_start={crop_start_sec:.2f}s)"
                )
        
        # Apply offset and duration
        offset_ms = int(offset_sec * 1000)
        
        if duration_sec is not None:
            duration_ms = int(duration_sec * 1000)
            audio_segment = audio_segment[offset_ms:offset_ms + duration_ms]
        elif offset_ms > 0:
            audio_segment = audio_segment[offset_ms:]
        
        # Resample if needed
        if audio_segment.frame_rate != self.sampling_rate:
            audio_segment = audio_segment.set_frame_rate(self.sampling_rate)
        
        # Convert to numpy array preserving original channels
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        array = samples / (2 ** (8 * audio_segment.sample_width - 1))
        
        # Reshape to [channels, time] preserving original channel count
        if audio_segment.channels > 1:
            array = array.reshape(-1, audio_segment.channels).T
        else:
            array = array.reshape(1, -1)
        
        return array
    
    def _load_local_audio(self, audio_metadata: dict) -> np.ndarray:
        """
        Load audio from local filesystem with fallback mechanisms.
        
        Attempts to load audio using:
        1. sphn library (optimized for sphere format, TIMIT/LibriSpeech)
        2. pydub (fallback for common formats: mp3, wav, flac, etc.)
        
        With random crop support (when self.random_crop_sample_size is set):
        - If audio is longer than needed, randomly selects a crop window
        - Passes adjusted offset and duration to loaders
        
        Args:
            audio_metadata: Dict containing path, offset, and duration
        
        Returns:
            numpy array of shape [channels, time] with audio data normalized to [-1, 1]
        
        Raises:
            Exception: If both loading methods fail
        """
        # Get original offset and duration from metadata
        offset_sec = audio_metadata.get("offset", 0.0)
        duration_sec = audio_metadata.get("duration")
        
        # Apply random crop if enabled (automatic via _get_frames_to_read)
        # Note: For local files, we work in seconds since we load at target sample rate
        if self.random_crop_sample_size is not None and duration_sec is not None:
            target_duration_sec = self.random_crop_sample_size / self.sampling_rate
            
            if duration_sec > target_duration_sec:
                # Random crop: pick random start position
                max_start_sec = duration_sec - target_duration_sec
                crop_start_sec = self.rng.random() * max_start_sec
                
                offset_sec += crop_start_sec
                duration_sec = target_duration_sec
                
                logger.debug(
                    f"Random crop (local): {audio_metadata.get('duration', 0):.2f}s → {target_duration_sec:.2f}s "
                    f"(crop_start={crop_start_sec:.2f}s)"
                )
        
        try:
            # Attempt to read with sphn, suitable for sphere-formatted files
            array, _ = sphn.read(
                audio_metadata["path"],
                start_sec=offset_sec,
                duration_sec=duration_sec,
                sample_rate=self.sampling_rate,
            )  # [channels, time]
            return array
            
        except Exception as e:
            logger.warning(
                f"Failed to read audio with `sphn`:\n"
                f"\tpath: {audio_metadata['path']}\n"
                f"\tstart_sec: {offset_sec}, duration_sec: {duration_sec}\n"
                f"\tError: {e}\n"
                f"Attempting with `pydub`."
            )
            
            # Fallback to pydub for more common audio formats
            audio_segment = AudioSegment.from_file(
                audio_metadata["path"],
                start_second=offset_sec,
                duration=duration_sec
            ).set_frame_rate(self.sampling_rate)
            
            # Convert to numpy array preserving original channels
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            array = samples / (2 ** (8 * audio_segment.sample_width - 1))
            
            # Reshape to [channels, time] - preserve original channel count
            if audio_segment.channels > 1:
                array = array.reshape(-1, audio_segment.channels).T
            else:
                array = array.reshape(1, -1)
            
            return array

    def load_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads audio for a single example.
        
        Supports both single-audio and multi-audio entries:
        - Single-audio: example["audio"] is Dict, returns {"wav": array, "is_multi_audio": False}
        - Multi-audio: example["audio"] is List[Dict], returns {"wav": [array1, array2, ...], "is_multi_audio": True}
        
        The is_multi_audio flag is determined by:
        1. Explicit example.get("is_multi_audio") field (preferred)
        2. Fallback: isinstance(example["audio"], list)
        """
        # Initialize S3 if needed
        if not self.s3_client:
            self._init_s3_client()
        
        # Check for multi-audio mode
        is_multi_audio = example.get("is_multi_audio", False)
        
        # If not explicitly set, infer from audio type (fallback for compatibility)
        if not is_multi_audio and isinstance(example.get("audio"), list):
            is_multi_audio = True
        
        if is_multi_audio:
            return self._load_multi_audio(example)
        else:
            return self._load_single_audio(example)
    
    def _load_single_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Load a single audio file."""
        audio = example["audio"]
        array = self._load_audio_dict(audio)
        return {"wav": array, "is_multi_audio": False}
    
    def _load_multi_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Load multiple audio files from a multi-audio entry.
        
        Returns:
            Dict with:
            - wav: List of numpy arrays, one per audio file
            - is_multi_audio: True
            - num_audios: Number of audio files
        """
        audio_list = example["audio"]
        if not isinstance(audio_list, list):
            # Shouldn't happen, but handle gracefully
            audio_list = [audio_list]
        
        wav_list = []
        for audio_dict in audio_list:
            array = self._load_audio_dict(audio_dict)
            wav_list.append(array)
        
        return {
            "wav": wav_list,
            "is_multi_audio": True,
            "num_audios": len(wav_list)
        }
    
    def _load_audio_dict(self, audio: Dict[str, Any]) -> np.ndarray:
        """Load audio from a single audio dict. Core loading logic."""
        storage_backend = self.storage_backend

        if storage_backend == "tarball_lustre":
            array = self._load_tarball_local_audio_byteseek(audio)
        elif storage_backend == "tarball_s3":
            array = self._load_tarball_s3_audio_byteseek(audio)
        elif storage_backend == "lustre":
            array = self._load_local_audio(audio)
        elif storage_backend == "s3":
            if not self.s3_client:
                raise RuntimeError("S3 client not initialized (check credentials/install)")
            
            _, _, bucket, key = audio["path"].split("/", 3)
            format_str = get_format_from_path(key)
            
            if is_byte_seekable_format(format_str):
                array = self._load_s3_wav_with_byte_seeking(bucket, key, audio)
            else:
                array = self._load_s3_compressed_audio(bucket, key, audio, format_str)
        else:
            raise ValueError(f"Unknown storage_backend: '{storage_backend}'")
        
        return array


if __name__ == "__main__":
    import tarfile
    import shutil
    import tempfile
    import time
    import copy
    
    # Configure logging for the demo
    logging.basicConfig(level=logging.INFO)
    
    print("-" * 80)
    print("Running Standalone SimpleAudioLoader Demo")
    print("-" * 80)
    
    # 1. Setup Mock Dataset Directory
    base_dir = "./tmp/DEBUG_mock_dataset"
    if os.path.exists(base_dir):
        print(f"Cleaning up existing directory: {base_dir}")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Created mock dataset directory: {base_dir}")

    # 2. Generate Fake Audio Files (White Noise)
    sr = 48000
    channels = 2
    duration = 2.0 # seconds
    
    mock_wavs = []
    for i in range(3):
        # Generate white noise
        noise = np.random.uniform(-0.5, 0.5, size=(int(sr * duration), channels)).astype(np.float32)
        filename = f"mock_audio_{i}.wav"
        filepath = os.path.join(base_dir, filename)
        sf.write(filepath, noise, sr, subtype='PCM_16')
        mock_wavs.append(filename)
        print(f"  Generated mock WAV: {filename}")

    # 3. Create Tarball
    tar_filename = "train.001-of-001.tar"
    tar_path = os.path.join(base_dir, tar_filename)
    
    manifest_entries = []
    
    # Create tarball from generated wavs
    with tarfile.open(tar_path, "w") as tar:
        for wav_file in mock_wavs:
            full_path = os.path.join(base_dir, wav_file)
            tar.add(full_path, arcname=wav_file)
    
    print(f"Created tarball: {tar_path}")
    
    # 4. Inspect Tarball to Create Manifest (NDJSON)
    # We need precise byte offsets for the loader to work
    print("inspecting tarball for offsets...")
    
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name in mock_wavs:
                # Extract metadata using soundfile from the source file (for simplicity in this mock)
                full_path = os.path.join(base_dir, member.name)
                info = sf.info(full_path)
                
                # Determine data offset within the tar file
                # tarfile members usually have offset (header start). 
                # Data starts after header (512 bytes).
                # Check for offset_data attribute (Python 3+)
                if hasattr(member, 'offset_data'):
                    tar_offset = member.offset_data
                else:
                    tar_offset = member.offset + 512
                
                entry = {
                    "audio_id": f"mock_{member.name}",
                    "audio": {
                        "path": member.name,
                        "sampling_rate": info.samplerate,
                        "channels": info.channels,
                        "duration": info.duration,
                        "encoding": "PCM_S", # We saved as PCM_16
                        "bytes_per_sample": 2,
                        "data_offset": 44, # Standard WAV header size
                        "offset": 0.0,
                        "tar_path": tar_filename,
                        "tar_offset": tar_offset,
                        "tar_size": member.size
                    },
                    "dataset": "mock_dataset",
                    "metadata": {"id": member.name},
                    "conversations": [
                        {"from": "human", "value": "<sound>\nAnalyze this audio."},
                        {"from": "gpt", "value": "It sounds like white noise."}
                    ]
                }
                manifest_entries.append(entry)

    # 5. Write NDJSON Manifest
    manifest_path = os.path.join(base_dir, "train.001-of-001.ndjson")
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            json.dump(entry, f)
            f.write("\n")
            
    print(f"Created manifest: {manifest_path}")
    
    # 6. Create Index File (train.index)
    # This follows the schema required by the full dataset loader
    index_path = os.path.join(base_dir, "train.index")
    
    # Calculate simple stats for the mock data
    total_examples = len(mock_wavs)
    total_duration = total_examples * duration # 3 * 2.0 = 6.0s
    total_size = os.path.getsize(tar_path)
    
    index_content = {
        "features": {
            "audio_id": "str",
            "audio": {
                "path": "str",
                "sampling_rate": "int",
                "channels": "int",
                "offset": "float",
                "duration": "float",
                "bytes_per_sample": "int",
                "encoding": "str",
                "data_offset": "int",
                "tar_path": "str",
                "tar_offset": "int", 
                "tar_size": "int"
            },
            "dataset": "str",
            "metadata": "dict",
            "conversations": "list"
        },
        "shards": [
            {
                "filename": "train.001-of-001.ndjson",
                "num_examples": total_examples,
                "total_duration_hours": total_duration / 3600.0,
                "encodings": {
                    "PCM_S": total_examples
                },
                "audio_duration_seconds": {
                    "min": duration,
                    "max": duration,
                    "mean": duration,
                    "median": duration
                },
                "size_bytes": total_size,
                "checksum_md5": "mock_checksum" # skipped for demo
            }
        ],
        "num_examples": total_examples,
        "total_duration_hours": total_duration / 3600.0,
        "total_size_bytes": total_size,
        "datasets": [
            "mock_dataset"
        ],
        "dataset_distribution": {
            "mock_dataset": total_examples
        },
        "num_speakers": 0,
        "audio_duration_seconds": {
            "min": duration,
            "max": duration,
            "mean": duration,
            "median": duration
        },
        "audio_metadata": {
            "sampling_rates": {
                str(sr): total_examples
            },
            "channels": {
                str(channels): total_examples
            },
            "encodings": {
                "PCM_S": total_examples
            }
        },
        "processing_info": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
            "processor_version": "demo_1.0"
        },
        "manifest_version": "1.0"
    }
    
    with open(index_path, "w") as f:
        json.dump(index_content, f, indent=2)
        
    print(f"Created index: {index_path}")
    
    # 7. Test Loading with SimpleAudioLoader
    print("\n" + "=" * 40)
    print("Testing SimpleAudioLoader with Mock Data")
    print("=" * 40)
    
    loader = SimpleAudioLoader(sampling_rate=48000)
    
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)

            # Create a copy for loading so we don't mutate the original entry for display
            load_entry = copy.deepcopy(entry)

            # Resolve relative tar_path for the loader to work from current location
            # (In real usage, the dataset class handles joining with a root directory)
            if not os.path.isabs(load_entry["audio"]["tar_path"]):
                load_entry["audio"]["tar_path"] = os.path.join(base_dir, load_entry["audio"]["tar_path"])

            print(f"\nLoading Entry {i+1}: {entry['audio_id']}")
            
            try:
                start_time = time.time()
                result = loader.load_audio(load_entry)
                duration_ms = (time.time() - start_time) * 1000
                
                wav = result["wav"]
                print(f"  Success! Loaded in {duration_ms:.2f}ms")
                print(f"  Shape: {wav.shape}") # Expected [2, 96000] for 2s @ 48k
                print(f"  Value Range: [{wav.min():.3f}, {wav.max():.3f}]")
                
                print(f"  Entry Content:")
                print(json.dumps(entry, indent=2))
                
                # Verify properties
                assert wav.shape[0] == channels
                # Allow small tolerance for resampling/rounding
                assert abs(wav.shape[1] - (sr * duration)) < 100
                
            except Exception as e:
                print(f"  FAILED to load: {e}")
                import traceback
                traceback.print_exc()
                raise e

    print("\n" + "-" * 80)
    print("Demo Completed Successfully!")
    print("-" * 80)
