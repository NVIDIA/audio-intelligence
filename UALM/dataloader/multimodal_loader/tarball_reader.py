# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Tarball Metadata Reader base class and specialized readers for consolidated metadata.jsonl/lmdb.
"""

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import soundfile as sf
import librosa
import lmdb

# --- Optional Imports for Fallback Decoding ---
try:
    import torchcodec
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False
# ----------------------------------------------

logger = logging.getLogger(__name__)

class TarballMetadataReader:
    """
    Base reader that loads a unified metadata file (LMDB or JSONL) into a shared cache/handle.
    Prioritizes LMDB for efficient random access without loading everything into RAM.
    
    The cache is stored as a class-level dictionary keyed by absolute file path.
    For LMDB, it stores the environment handle. For JSONL, it stores the full dict.
    """
    
    # Class-level cache: {absolute_path: {id: entry} OR lmdb.Environment}
    _cache = {}
    
    def __init__(self, file_path: str, valid_ids: Optional[list] = None):
        self.file_path = str(Path(file_path).resolve())
        self.is_lmdb = self.file_path.endswith('.lmdb')
        self._env = None
        
        # Load/Open if not already cached
        if self.file_path not in self._cache:
            if self.is_lmdb:
                # self._open_lmdb(self.file_path)
                
                # Do NOT open LMDB here; workers will open it lazily.
                self.data = None
            else:
                # self._load_jsonl(self.file_path)

                # JSONL path: still cached at class level
                if self.file_path not in self._cache:
                    self._load_jsonl(self.file_path)
                self.data = self._cache[self.file_path]
            
        # self.data = self._cache[self.file_path]
        self.valid_ids_set = set(valid_ids) if valid_ids is not None else None

    def _load_jsonl(self, path):
        """Load the full metadata file into the cache."""
        logger.info(f"Loading unified metadata from {path}")
        data = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        item_id = item.get("id")
                        if item_id:
                            data[item_id] = item
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to load metadata file {path}: {e}")
            raise
            
        self._cache[path] = data
        logger.info(f"Cached {len(data)} entries from {path}")

    def _open_lmdb(self, path):
        """Open LMDB environment."""
        logger.info(f"Opening LMDB environment at {path}")
        # Open read-only, no lock (for multi-process reading safety)
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
        # self._cache[path] = env
        return env
    
    def _get_env(self):
        """Lazy-init LMDB env per process."""
        if self._env is None:
            self._env = self._open_lmdb(self.file_path)
        return self._env

    def _get_entry(self, key: str) -> Dict[str, Any]:
        """Helper to get filtered entry."""
        if self.valid_ids_set is not None and key not in self.valid_ids_set:
            raise KeyError(f"Sample ID {key} is not in valid_ids")

        if self.is_lmdb:
            env = self._get_env()
            with env.begin(write=False) as txn:
                value = txn.get(key.encode('utf-8'))
                if value is None:
                    raise KeyError(f"Sample ID {key} not found in LMDB")
                return pickle.loads(value)
        else:
            if key not in self.data:
                raise KeyError(f"Sample ID {key} not found in metadata")
            return self.data[key]

    def __contains__(self, key: str) -> bool:
        if self.valid_ids_set is not None and key not in self.valid_ids_set:
            return False
            
        if self.is_lmdb:
            env = self._get_env()
            with env.begin(write=False) as txn:
                # Check existence without loading full object
                cursor = txn.cursor()
                return cursor.set_key(key.encode('utf-8'))
        else:
            return key in self.data

    def __len__(self) -> int:
        if self.valid_ids_set is not None:
            # Exact count requires intersection, which is expensive for LMDB
            # For now, just return valid_ids set size as approximation or full count
            # If precise count is needed for iterating, keys() will handle it.
            return len(self.valid_ids_set)
            
        if self.is_lmdb:
            env = self._get_env()
            with env.begin() as txn:
                return txn.stat()['entries']
        else:
            return len(self.data)

    def keys(self):
        if self.valid_ids_set is not None:
            return iter(self.valid_ids_set)
            
        if self.is_lmdb:
            # Generator for all keys in LMDB
            env = self._get_env()
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    yield key.decode('utf-8')
        else:
            return self.data.keys()
    
    def __getstate__(self):
        """
        Remove non-picklable state (LMDB env, open file handles) when
        sending this object to DataLoader worker processes.
        """
        state = self.__dict__.copy()
        # LMDB env cannot be pickled
        if "_env" in state:
            state["_env"] = None
        # TarballAudioReader adds tar_handles; clear them on pickle
        if "tar_handles" in state:
            state["tar_handles"] = {}
        return state



# ==============================================================================
# Audio Reader Implementation (inherits from TarballMetadataReader)
# ==============================================================================

class VirtualFileSection:
    """Virtual file object for reading chunks from tarballs."""
    def __init__(self, file_obj, start_offset, length):
        self.file_obj = file_obj
        self.start_offset = start_offset
        self.length = length
        self.end_offset = start_offset + length
        self.file_obj.seek(self.start_offset)

    def read(self, size=-1):
        current_pos = self.file_obj.tell()
        bytes_left = self.end_offset - current_pos
        if bytes_left <= 0:
            return b""
        if size < 0 or size > bytes_left:
            size = bytes_left
        return self.file_obj.read(size)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            target = self.start_offset + offset
        elif whence == os.SEEK_CUR:
            target = self.file_obj.tell() + offset
        elif whence == os.SEEK_END:
            target = self.end_offset + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self.file_obj.seek(target)
        return self.tell()

    def tell(self):
        return self.file_obj.tell() - self.start_offset

    def flush(self): pass
    def close(self): pass


class TarballAudioReader(TarballMetadataReader):
    """
    Reads audio from tarballs using metadata from the unified index.
    Expects metadata entry to have an 'audio' dict with tar_path/offset.
    """
    def __init__(self, index_path: str, valid_ids: list = None, target_sample_rate: int = None):
        super().__init__(index_path, valid_ids)
        self.tar_handles = {}
        self.target_sample_rate = target_sample_rate

    def _get_tar_handle(self, tar_path: str):
        if tar_path not in self.tar_handles:
            if not os.path.exists(tar_path):
                raise FileNotFoundError(f"Tarball not found: {tar_path}")
            self.tar_handles[tar_path] = open(tar_path, "rb")
        return self.tar_handles[tar_path]
    
    def __getitem__(self, key: str) -> Tuple[np.ndarray, int]:
        try:
            return self._unstable_getitem(key)
        except Exception as e:
            logger.error(f"Failed to get item {key}: {e}")
            return None, None

    def _unstable_getitem(self, key: str) -> Tuple[np.ndarray, int]:
        entry = self._get_entry(key)
        
        # Extract audio metadata
        if "audio" in entry:
            meta = entry["audio"]
        else:
            raise KeyError(f"No audio metadata found for {key}")

        tar_path = meta["tar_path"]
        offset = meta["tar_offset"]
        size = meta["tar_size"]
        
        handle = self._get_tar_handle(tar_path)
        file_view = VirtualFileSection(handle, offset, size)
        
        # Calculate start and number of frames to read
        start_sec = meta.get("offset", 0.0)
        duration_sec = meta.get("duration")
        sample_rate = meta.get("sampling_rate")
        
        if sample_rate is None:
            try:
                info = sf.info(file_view)
                sample_rate = info.samplerate
                file_view.seek(0) # Reset
            except:
                raise ValueError(f"Missing sampling rate in metadata for {key}")

        start_frame = int(start_sec * sample_rate)
        frames_to_read = int(duration_sec * sample_rate) if duration_sec else -1

        try:
            audio, read_sample_rate = sf.read(
                file_view, 
                start=start_frame,
                frames=frames_to_read,
                dtype="float32", 
                always_2d=True
            )
            assert read_sample_rate == sample_rate, f"Read sample rate {read_sample_rate} does not match metadata sample rate {sample_rate}"
            audio = audio.T # soundfile.read returns (frames, channels), we need (channels, frames)
        except Exception as e:
            # Fallback logic implemented based on comments
            decoded = False
            
            # 1. Try TorchCodec
            if HAS_TORCHCODEC:
                try:
                    file_view.seek(0) # Reset file pointer
                    audio_dec = torchcodec.decoders.AudioDecoder(file_view, sample_rate=self.target_sample_rate)
                    if audio_dec.metadata.begin_stream_seconds_from_header:
                        offset_sec = audio_dec.metadata.begin_stream_seconds_from_header
                    else:
                        offset_sec = 0.0
                    start_seconds = offset_sec + start_sec
                    stop_seconds = offset_sec + start_sec + duration_sec
                    audio_samples = audio_dec.get_samples_played_in_range(start_seconds, stop_seconds)
                    audio = audio_samples.data.numpy()
                    read_sample_rate = audio_samples.sample_rate
                    decoded = True
                except Exception as tc_e:
                    logger.debug(f"TorchCodec fallback failed for {key}: {tc_e}")

            # 2. Try Torchaudio (if TorchCodec failed or wasn't available)
            if not decoded and HAS_TORCHAUDIO:
                try:
                    file_view.seek(0) # Reset file pointer
                    # Logic note: This will eventually be deprecated in favor of torchcodec
                    audio_tensor, read_sample_rate = torchaudio.load(file_view, frame_offset=start_frame, num_frames=frames_to_read)
                    audio = audio_tensor.numpy()
                    decoded = True
                except Exception as ta_e:
                    logger.debug(f"Torchaudio fallback failed for {key}: {ta_e}")
            
            if not decoded:
                raise RuntimeError(f"Failed to decode audio for {key} (tried sf, torchcodec, torchaudio): {e}")
            
        # Resample if needed
        if self.target_sample_rate and read_sample_rate != self.target_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=read_sample_rate, target_sr=self.target_sample_rate,
                res_type="soxr_vhq"
            )
            final_sample_rate = self.target_sample_rate
        else:
            final_sample_rate = read_sample_rate

        return audio, final_sample_rate
        
    def close(self):
        for h in self.tar_handles.values(): h.close()
        self.tar_handles.clear()


# ==============================================================================
# Modular Dialogue Reader
# ==============================================================================

class TaskTemplate(ABC):
    """Strategy for converting raw metadata entry to UALM messages."""
    @abstractmethod
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        pass

class DefaultTemplate(TaskTemplate):
    """Default template: Uses 'conversations' list directly if present."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        conversations = entry.get("conversations") or entry.get("messages")
        if not conversations:
            return []
        
        messages = []
        for turn in conversations:
            role = turn.get("from", "human")
            if role == "human": role = "user"
            elif role == "gpt": role = "assistant"
            
            content = turn.get("value", "")
            
            # # while this is general enough for dialogues, it doesn't handle the voice assistant case where content='<sound>'
            # parts = content.split("<sound>")
            # for i, part in enumerate(parts):
            #     if part.strip():
            #         messages.append([role, "text", part])
            #     if i < len(parts) - 1:
            #         messages.append([role, "audio", audio_placeholder])

            # # for now, let's fix just one text input and if the text is none, let it be " ".
            part = content.replace("<sound>", " ")
            messages.append([role, "text", part])
            if role == "user":
                messages.append([role, "audio", audio_placeholder])

        return messages

class AudioToCaptionTemplate(TaskTemplate):
    """Task: Audio Understanding/Captioning (User <sound> -> GPT <text>)."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        default_msgs = DefaultTemplate().convert(entry, audio_placeholder)
        if default_msgs: 
            return default_msgs
            
        text = entry.get("text") or entry.get("caption") or ""
        return [
            ["user", "text", "<|audio to caption|> Generate caption for this audio."],
            ["user", "audio", audio_placeholder],
            ["assistant", "text", text]
        ]

class AudioToConversationTemplate(TaskTemplate):
    """Task: Audio Conversation (User <sound> + <text> -> GPT <text>)."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        default_msgs = DefaultTemplate().convert(entry, audio_placeholder)
        assert default_msgs is not None and len(default_msgs) > 0  # must be conversations
        if default_msgs: 
            return default_msgs
            
        # Fallback if no conversations but we expect conversation structure
        # This typically shouldn't happen for reasoning data as it has 'conversations'
        # But if it does, treat as captioning
        # return AudioToCaptionTemplate().convert(entry, audio_placeholder)

class CaptionToAudioTemplate(TaskTemplate):
    """Task: Text-to-Audio (User <text> -> GPT <sound>)."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        default_msgs = DefaultTemplate().convert(entry, audio_placeholder)
        if default_msgs: 
            return default_msgs
            
        text = entry.get("text") or entry.get("caption") or ""
        return [
            ["user", "text", "<|text to audio|> Generate audio for this caption. " + text],
            ["assistant", "audio", audio_placeholder]
        ]

class AudioOnlyTemplate(TaskTemplate):
    """Task: Audio Continuation (autoregressive audio token modeling)."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        return [
            ["user", "text", "<|audio continuation|> Generate continuation of this audio."],
            ["assistant", "audio", audio_placeholder]
        ]

class TranscriptionToSpeechTemplate(TaskTemplate):
    """Task: TTS (User <text> -> GPT <sound>)."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        # Structurally same as CaptionToAudio
        # return CaptionToAudioTemplate().convert(entry, audio_placeholder)

        text = entry.get("text") or entry.get("transcription") or ""
        return [
            ["user", "text", "<|text to speech|> Generate speech for this transcription. " + text],
            ["assistant", "audio", audio_placeholder]
        ]

class SpeechToTranscriptionTemplate(TaskTemplate):
    """Task: ASR (User <sound> -> GPT <text>)."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        # Structurally same as AudioToCaption
        # return AudioToCaptionTemplate().convert(entry, audio_placeholder)

        default_msgs = DefaultTemplate().convert(entry, audio_placeholder)
        if default_msgs: 
            return default_msgs
            
        text = entry.get("text") or entry.get("transcription") or ""
        return [
            ["user", "text", "<|speech recognition|> Generate transcription for this speech."],
            ["user", "audio", audio_placeholder],
            ["assistant", "text", text]
        ]

class TextOnlyTemplate(TaskTemplate):
    """Task: Text-only dialogue (no audio involved)."""
    def convert(self, entry: Dict, audio_placeholder: str) -> List[List[str]]:
        # For text-only, we directly use conversations without audio
        conversations = entry.get("conversations") or entry.get("messages")
        if conversations:
            messages = []
            for turn in conversations:
                role = turn.get("from", "human")
                if role == "human": role = "user"
                elif role == "gpt": role = "assistant"
                
                content = turn.get("value", "")
                # Text-only: no audio placeholders, just text
                messages.append([role, "text", content])
            return messages
            
        # Fallback for simple text
        text = entry.get("text") or entry.get("caption") or ""
        return [["assistant", "text", text]]


class TarballDialogueReader(TarballMetadataReader):
    """
    Modular dialogue reader that selects a template based on 'ualm_task'.
    """
    
    def __init__(self, index_path: str, valid_ids: Optional[List[str]] = None):
        super().__init__(index_path, valid_ids)
        
        # Registry of templates
        self.templates = {
            "audio_to_caption": AudioToCaptionTemplate(),
            "caption_to_audio": CaptionToAudioTemplate(),
            # "text_to_audio": CaptionToAudioTemplate(),
            # "audio_to_text": AudioToCaptionTemplate(),
            "audio_to_conversation": AudioToConversationTemplate(),
            
            "transcription_to_speech": TranscriptionToSpeechTemplate(),
            "speech_to_transcription": SpeechToTranscriptionTemplate(),
            
            "audio_only": AudioOnlyTemplate(),
            "text_only": TextOnlyTemplate(),
            
            "default": DefaultTemplate()
        }

    def __getitem__(self, key: str) -> List[List[str]]:
        entry = self._get_entry(key)
        
        # Resolve content part of the entry
        if "text" in entry:
            content_data = entry["text"]
        else:
            raise KeyError(f"No text metadata found for {key}")
            
        # Resolve audio placeholder (path) - may not exist for text-only
        audio_meta = entry.get("audio", {})
        audio_path = audio_meta.get("tar_path", "placeholder_audio") if audio_meta else "placeholder_audio"
        
        # Determine task
        task = entry.get("ualm_task") or content_data.get("ualm_task")
        
        template = self.templates.get(task, self.templates["default"])
        
        messages = template.convert(content_data, audio_path)
        
        if not messages:
            # Absolute fallback
            messages = [["user", "audio", audio_path]]
            
        return messages
