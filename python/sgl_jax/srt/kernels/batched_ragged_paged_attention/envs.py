# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Environment switches used by the vendored Batched RPA kernel."""

import os

USE_BATCHED_RPA_SEQ_ON_LANE = os.environ.get("SGL_JAX_BATCHED_RPA_SEQ_ON_LANE", "0").lower() in (
    "1",
    "true",
)
