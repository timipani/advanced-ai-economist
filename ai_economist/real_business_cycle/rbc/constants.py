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
    work_choices = np.array(list(itertools.product(*work_choices)), dtype=_NP_DTYPE)

    price_choices = np.array([0.0 + 500.0 * c for c in range(6)], dtype=_NP_DTYPE)
    wage_choices = np.array([0.0, 11.0, 22.0, 33.0, 44.0], dtype=_NP_DTYPE)
    capital_choices = np.array([0.1], dtype=_NP_DTYPE)
    price_and_wage = np.array(
        list(itertools.product(price_choices, wage_choices, capital_choices)),
        dtype=_NP_DTYPE,
    )

    # government action discretization
    income_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    corporate_taxation_choices = np.array(
        [0.0 + 0.2 * c for c in range(6)], dtype=_NP_DTYPE
    )
    tax_choices = np.array(
        list(itertools.product(income_taxation_choices, corporate_taxation_choices)),
        dtype=_NP_DTYPE,
    )
    global_state_dim = (
        NUMFIRMS  # prices
        + NUMFIRMS  # wages
        + NUMFIRMS  # stocks
        + NUMFIRMS  # was good overdemanded
        + 2 * NUMGOVERNMENTS  # tax rates
        + 1
    )  # time

    global_state_digit_dims = list(
  