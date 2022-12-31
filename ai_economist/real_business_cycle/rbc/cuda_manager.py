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
            return 