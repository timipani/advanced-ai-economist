# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import json
import os
from datetime import datetime, timedelta

import GPUtil
import numpy as np

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.utils import verify_activation_code

try:
    num_gpus_available = len(GPUtil.getAvailable())
    print(f"Inside covid19_env.py: {num_gpus_available} GPUs are available.")
    if num_gpus_available == 0:
        print("No GPUs found! Running the simulation on a CPU.")
    else:
        from warp_drive.utils.constants import Constants
        from warp_drive.utils.data_feed import DataFeed

        _OBS