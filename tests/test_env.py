# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Unit tests for the wood and stone scenario + basic components
"""

import unittest

from ai_economist import foundation


class CreateEnv:
    """
    Create an environment instance based on a configuration
    """

    def __init__(self):
        self.env = None
        self.set_env_config()

    def set_env_config(self):
        """Set up a sample environment config"""
        self.env_config = {
            # ===== STANDARD ARGUMENTS ======
            "n_agents": 4,  # Number of non-planner agents
            "world_size": [15, 15],  # [Height, Width] of the env world
            "episode_length": 1000,  # Number 