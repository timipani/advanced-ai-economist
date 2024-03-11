# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Wrapper for making the gather-trade-build environment an OpenAI compatible environment.
This can then be used with reinforcement learning frameworks such as RLlib.
"""

import os
import pickle
import random
import warnings

import numpy as np
from ai_economist import foundation
from gym import spaces
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

_BIG_NUMBER = 1e20


def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError


def pretty_print(dictionary):
    for key in dictionary:
        print("{:15s}: {}".format(key, dictionary[key].shape))
    print("\n")


class RLlibEnvWrapper(MultiAgentEnv):
    """
