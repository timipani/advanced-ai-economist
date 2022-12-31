# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import random
from pathlib import Path

import numpy as np
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda_driver
import scipy
import scipy.stats
import torch
from pycuda.compiler import SourceModule
from torch.distributions import Categorical
from tqdm import tqdm

from .constants import (
    consumer_state_scaling_factors,
    firm_state_scaling_factors,
    govt_state_scaling_factors,
)
from .networks import DeterministicPolicy, IndependentPolicyNet, PolicyNet
from .util import expand_to_digit_form, size_after_digit_expansion

_NP_DTYPE = np.float32

# the below line is 'strangely' necessary to make PyTorch work with PyCUDA
pytorch_cuda_init_success = torch.cuda.FloatTensor(8)


# for opening source files within module
module_path = Path(__file__).parent


def interval_list_contains(interval_list, step):
    for (lower, upper_non_inclusive) in interval_list:
        if lower <= step < upper_non_inclusive:
            return True
    return False


class NoOpOptimizer:
    """
    Dummy Optimizer.
    """

    def __init__(self):
        pass

    def step(self):
        pass


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reverse_cumsum(x):
    # assumes summing along episode iteration dim
    return x + torch.sum(x, dim=-2, keepdims=True) - torch.cumsum(x, dim=-2)


def discounted_returns(rewards, gamma):
    maxt = rewards.shape[-2]
    cumulative_rewards = 0
    returns = torch.zeros_like(rewards)
    for t in reversed(range(maxt)):
        returns[:, t, :] = rewards[:, t, :] + gamma * cumulative_rewards
        cumulative_rewards = rewards[:, t, :] + cumulative_rewards
    return returns


def compute_theta_coef(hparams_dict, episode):
    anneal_dict = hparams_dict["agents"]["consumer_anneal_theta"]
    if anneal_dict["anneal_on"]:
        exp_decay_length_in_steps = anneal_dict["exp_decay_length_in_steps"]
        theta_coef = np.float32(1.0 - (np.exp(-episode / exp_decay_length_in_steps)))
    else:
        return np.float32(1.0)
    return theta_coef


def government_action_mask