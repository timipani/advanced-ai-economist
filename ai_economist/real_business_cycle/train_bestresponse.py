# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from experiment_utils import cfg_dict_from_yaml
from rbc.cuda_manager import ConsumerFirmRunManagerBatchParallel


def check_if_ep_str_policy_exists(rollout_path, ep_str):
    return (
        rollout_path / Path("saved_models") / Path(f"consumer_policy_{ep_str}.pt")
    ).is_file()


def run_rollout(rollout_path, arguments):
    """
