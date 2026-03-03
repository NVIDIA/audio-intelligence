# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Task configuration definitions for multimodal UALM data loading."""

# List of supported entries for task configurations
SUPPORTED_ENTRIES = (
    [f"audio{i}" for i in range(1, 11)]  # audio1 to audio10
    + [f"text{i}" for i in range(1, 11)]  # text1 to text10
    + ["dialogue", "speaker"]
)


# Centralized task configuration mapping
TASK_CONFIGS = {
    # "text_to_audio": {
    #     "required_entries": ["text1", "audio1"],
    # },
    # "audio_to_text": {
    #     "required_entries": ["audio1", "text1"],
    # },
    "text_only": {
        "required_entries": ["text1", "text2"],
    },
    # "dialogue": {
    #     "required_entries": ["dialogue"],
    # },

    
    # added ualm_tasks
    "caption_to_audio": {
        "required_entries": ["text1", "audio1"],
    },
    "audio_to_caption": {
        "required_entries": ["text1", "audio1", "text2"],
    },
    "audio_to_conversation": {
        "required_entries": ["audio1", "text1", "text2"],
    },
    "audio_only": {
        "required_entries": ["text1", "audio1"],
    },
    
    "transcription_to_speech": {
        "required_entries": ["text1", "audio1"],
    },
    "speech_to_transcription": {
        "required_entries": ["text1", "audio1", "text2"],
    },
}


# Sanity check: ensure all entries in TASK_CONFIGS are supported
def _validate_task_configs():
    """Validate that all entries in TASK_CONFIGS are in SUPPORTED_ENTRIES."""
    for task_name, config in TASK_CONFIGS.items():
        for entry in config.get("required_entries", []):
            if entry not in SUPPORTED_ENTRIES:
                raise ValueError(
                    f"Invalid entry '{entry}' in task '{task_name}'. "
                    f"Must be one of: {SUPPORTED_ENTRIES}"
                )


_validate_task_configs()
