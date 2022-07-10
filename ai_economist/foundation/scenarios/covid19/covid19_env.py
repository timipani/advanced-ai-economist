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

        _OBSERVATIONS = Constants.OBSERVATIONS
        _ACTIONS = Constants.ACTIONS
        _REWARDS = Constants.REWARDS
except ModuleNotFoundError:
    print(
        "Warning: The 'WarpDrive' package is not found and cannot be used! "
        "If you wish to use WarpDrive, please run "
        "'pip install rl-warp-drive' first."
    )
except ValueError:
    print("No GPUs found! Running the simulation on a CPU.")


@scenario_registry.add
class CovidAndEconomyEnvironment(BaseEnvironment):
    """
    A simulation to model health and economy dynamics amidst the COVID-19 pandemic.
    The environment comprising 51 agents (each agent corresponding to a US state and
    Washington D.C.) and the Federal Government (planner). The state agents decide the
    stringency level of the policy response to the pandemic, while the federal
    government provides subsidies to eligible individuals.

    This simulation makes modeling assumptions. For details, see the technical paper:
    https://arxiv.org/abs/2108.02904

    Args:
        use_real_world_data (bool): Replay what happened in the real world.
            Real-world data comprises SIR (susceptible/