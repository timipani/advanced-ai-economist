# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse
import os

from experiment_utils import (
    create_job_dir,
    run_experiment_batch_parallel,
    sweep_cfg_generator,
)
from rbc.constants import all_agents_short_export_experiment_template

train_param_sweeps = {
    "lr": [0.001],
    "entropy": [0.5],
    "batch_size": [128],
    "clip_grad_norm": [2.0],
    "base_seed": [2345],
    "should_boost_firm_reward": [False],
    "use_ppo": [True],
    "ppo_num_updates": [2, 4],
    "ppo_clip_param": [0.1],
}


agent_param_sweeps = {
    "consumer_lr_multiple": [1.0],
    "consumer_reward_scale": [5.0],
    "government_reward_scale": [5.0 * 100.0 * 2.0],
    "firm_reward_scale": [30000],
    "government_counts_firm_reward": [1],
    "government_lr_multiple": [0.05],
}


world_param_sweeps = {
    "initial_wages": [0.0],
    "interest_rate": [0.0],
    "importer_price": [500.0],
    "importer_quantity": [100.0],
    "use_impo