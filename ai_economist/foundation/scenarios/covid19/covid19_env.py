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
            Real-world data comprises SIR (susceptible/infected/recovered),
            unemployment, government policy, and vaccination numbers.
            This setting also sets use_real_world_policies=True.
        use_real_world_policies (bool): Run the environment with real-world policies
            (stringency levels and subsidies). With this setting and
            use_real_world_data=False, SIR and economy dynamics are still
            driven by fitted models.
        path_to_data_and_fitted_params (dirpath): Full path to the directory containing
            the data, fitted parameters and model constants. This defaults to
            "ai_economist/datasets/covid19_datasets/data_and_fitted_params".
            For details on obtaining these parameters, please see the notebook
            "ai-economist-foundation/ai_economist/datasets/covid19_datasets/
            gather_real_world_data_and_fit_parameters.ipynb".
        start_date (string): Date (YYYY-MM-DD) to start the simulation.
        pop_between_age_18_65 (float): Fraction of the population between ages 18-65.
            This is the subset of the population whose employment/unemployment affects
            economic productivity.
            Range: 0 <= pop_between_age_18_65 <= 1.
        infection_too_sick_to_work_rate (float): Fraction of people infected with
            COVID-19. Infected people don't work.
            Range: 0 <= infection_too_sick_to_work_rate <= 1
        risk_free_interest_rate (float): Percentage of interest paid by the federal
            government to borrow money from the federal reserve for COVID-19 relief
            (direct payments). Higher interest rates mean that direct payments
            have a larger cost on the federal government's economic index.
            Range: 0 <= risk_free_interest_rate
        economic_reward_crra_eta (float): CRRA eta parameter for modeling the economic
            reward non-linearity.
            A useful reference: https://en.wikipedia.org/wiki/Isoelastic_utility
            Range: 0 <= economic_reward_crra_eta
        health_priority_scaling_agents (float): A factor indicating how much more the
            states prioritize health (roughly speaking, loss of lives due to
            opening up more) over the economy (roughly speaking, a loss in GDP
            due to shutting down resulting in more unemployment) compared to the
            real-world.
            For example, a value of 1 corresponds to the real-world, while
            a value of 2 means that states cared twice as much about public health
            (preventing deaths), while a value of 0.5 means that states cared twice
            as much about the economy (preventing GDP drops).
            Range: 0 <= health_priority_scaling_agents
        health_priority_scaling_planner (float): same as above,
            but for the federal government.
            Range: 0 <= health_priority_scaling_planner
    """

    def __init__(
        self,
        *base_env_args,
        use_real_world_data=False,
        use_real_world_policies=False,
        path_to_data_and_fitted_params="",
        start_date="2020-03-22",
        pop_between_age_18_65=0.6,
        infection_too_sick_to_work_rate=0.1,
        risk_free_interest_rate=0.03,
        economic_reward_crra_eta=2,
        health_priority_scaling_agents=1,
        health_priority_scaling_planner=1,
        reward_normalization_factor=1,
        **base_env_kwargs,
    ):
        verify_activation_code()

        # Used for datatype checks
        self.np_float_dtype = np.float32
        self.np_int_dtype = np.int32

        # Flag to use real-world data or the fitted models instead
        self.use_real_world_data = use_real_world_data
        # Flag to use real-world policies (actions) or the supplied actions instead
        self.use_real_world_policies = use_real_world_policies

        # If we use real-world data, we also want to use the real-world policies
        if self.use_real_world_data:
            print(
                "Using real-world data to initialize as well as to "
                "step through the env."
            )
            # Note: under this setting, the real_world policies are also used.
            assert self.use_real_world_policies, (
                "Since the env. config. 'use_real_world_data' is True, please also "
                "set 'use_real_world_policies' to True."
            )
        else:
            print(
                "Using the real-world data to only initialize the env, "
                "and using