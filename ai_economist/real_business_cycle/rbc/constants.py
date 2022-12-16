# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import torch

_NP_DTYPE = np.float32


def all_agents_export_experiment_template(
    NUMFIRMS, NUMCONSUMERS, NUMGOVERNMENTS, episodes_const=30000
):
    consumption_choices = [
        np.array([0.0 + 1.0 * c for c in range(11)], dtype=_NP_DTYPE)
    ]
    work_choices = [
        np.array([0.0 + 20 * 13 * h for h in range(5)], dtype=_NP_DTYPE)
    ]  # specify dtype -- be consistent?

    consumption_choices = np.array(
        list(itertools.product(*consumption_choices)), dtype=_NP_DTYPE
    )
    w