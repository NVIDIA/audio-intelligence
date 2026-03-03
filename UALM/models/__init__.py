# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Model module for UALM job templates and configurations."""

from models.ualm.ualm_job import UALMJobTemplate

_all_job_types = {"ualm": UALMJobTemplate}

__all__ = [
    _all_job_types,
]
