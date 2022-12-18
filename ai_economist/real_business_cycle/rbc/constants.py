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
        range(2 * NUMFIRMS, 3 * NUMFIRMS)
    )  # stocks are the only global state var that can get huge
    consumer_state_dim = (
        global_state_dim + 1 + 1
    )  # budget  # theta, the disutility of work

    firm_state_dim = (
        global_state_dim
        + 1  # budget
        + 1  # capital
        + 1  # production alpha
        + NUMFIRMS  # onehot specifying which firm
    )

    episodes_to_anneal_firm = 100000
    episodes_to_anneal_government = 100000
    government_phase1_start = 100000
    government_state_dim = global_state_dim
    DEFAULT_CFG_DICT = {
        # actions_array key will be added below
        "agents": {
            "num_consumers": NUMCONSUMERS,
            "num_firms": NUMFIRMS,
            "num_governments": NUMGOVERNMENTS,
            "global_state_dim": global_state_dim,
            "consumer_state_dim": consumer_state_dim,
            # action vectors are how much consume from each firm,
            # how much to work, and which firm to choose
            "consumer_action_dim": NUMFIRMS + 1 + 1,
            "consumer_num_consume_actions": consumption_choices.shape[0],
            "consumer_num_work_actions": work_choices.shape[0],
            "consumer_num_whichfirm_actions": NUMFIRMS,
            "firm_state_dim": firm_state_dim,  # what are observations?
            # actions are price and wage for own firm, and capital choices
            "firm_action_dim": 3,
            "firm_num_actions": price_and_wage.shape[0],
            "government_state_dim": government_state_dim,
            "government_action_dim": 2,
            "government_num_actions": tax_choices.shape[0],
            "max_possible_consumption": float(consumption_choices.max()),
            "max_possible_hours_worked": float(work_choices.max()),
            "max_possible_wage": float(wage_choices.max()),
            "max_possible_price": float(price_choices.max()),
            # these are dims which, due to being on a large scale,
            # have to be expanded to a digit representation
            "consumer_digit_dims": global_state_digit_dims
            + [global_state_dim],  # global state + consumer budget
            # global state + firm budget (do we need capital?)
            "firm_digit_dims": global_state_digit_dims + [global_state_dim],
            # g