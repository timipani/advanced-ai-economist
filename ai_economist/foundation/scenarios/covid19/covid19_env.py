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
                "and using the fitted models to step through the env."
            )

        # Load real-world date
        if path_to_data_and_fitted_params == "":
            current_dir = os.path.dirname(__file__)
            self.path_to_data_and_fitted_params = os.path.join(
                current_dir, "../../../datasets/covid19_datasets/data_and_fitted_params"
            )
        else:
            self.path_to_data_and_fitted_params = path_to_data_and_fitted_params

        print(
            "Loading real-world data from {}".format(
                self.path_to_data_and_fitted_params
            )
        )
        real_world_data_npz = np.load(
            os.path.join(self.path_to_data_and_fitted_params, "real_world_data.npz")
        )
        self._real_world_data = {}
        for key in list(real_world_data_npz):
            self._real_world_data[key] = real_world_data_npz[key]

        # Load fitted parameters
        print(
            "Loading fit parameters from {}".format(self.path_to_data_and_fitted_params)
        )
        self.load_model_constants(self.path_to_data_and_fitted_params)
        self.load_fitted_params(self.path_to_data_and_fitted_params)

        try:
            self.start_date = datetime.strptime(start_date, self.date_format)
        except ValueError:
            print(f"Incorrect data format, should be {self.date_format}")

        # Start date should be beyond the date for which data is available
        assert self.start_date >= self.policy_start_date

        # Compute a start date index based on policy start date
        self.start_date_index = (self.start_date - self.policy_start_date).days
        assert 0 <= self.start_date_index < len(self._real_world_data["policy"])

        # For date logging (This will be overwritten in additional_reset_steps;
        # see below)
        self.current_date = None

        # When using real-world policy, limit the episode length
        # to the length of the available policy.
        if self.use_real_world_policies:
            real_world_policy_length = (
                len(self._real_world_data["policy"]) - self.start_date_index
            )
            print("Using real-world policies, ignoring external action inputs.")
            assert base_env_kwargs["episode_length"] <= real_world_policy_length, (
                f"The real-world policies are only available for "
                f"{real_world_policy_length} timesteps; so the 'episode_length' "
                f"in the environment configuration can only be at most "
                f"{real_world_policy_length}"
            )
        else:
            print("Using external action inputs.")

        # US states and populations
        self.num_us_states = len(self.us_state_population)

        assert (
            base_env_kwargs["n_agents"] == self.num_us_states
        ), "n_agents should be set to the number of US states, i.e., {}.".format(
            self.num_us_states
        )
        # Note: For a faster environment step time, we collate all the individual agents
        # into a single agent index "a" and we flatten the component action masks too.
        assert base_env_kwargs[
            "collate_agent_step_and_reset_data"
        ], "The env. config 'collate_agent_step_and_reset_data' should be set to True."
        super().__init__(*base_env_args, **base_env_kwargs)

        # Add attributes to self.world for use in components
        self.world.us_state_population = self.us_state_population
        self.world.us_population = self.us_population
        self.world.start_date = self.start_date
        self.world.n_stringency_levels = self.num_stringency_levels
        self.world.use_real_world_policies = self.use_real_world_policies
        if self.use_real_world_policies:
            # Agent open/close stringency levels
            self.world.real_world_stringency_policy = self._real_world_data["policy"][
                self.start_date_index :
            ]
            # Planner subsidy levels
            self.world.real_world_subsidy = self._real_world_data["subsidy"][
                self.start_date_index :
            ]

        # Policy --> Unemployment
        #   For accurately modeling the state-wise unemployment, we convolve
        #   the current stringency policy with a family of exponential filters
        #   with separate means (lambdas).
        # This code sets up things we will use in `unemployment_step()`,
        #   which includes a detailed breakdown of how the unemployment model is
        #   implemented.
        self.stringency_level_history = None
        # Each filter captures a temporally extended response to a stringency change.
        self.num_filters = len(self.conv_lambdas)
        self.f_ts = np.tile(
            np.flip(np.arange(self.filter_len), (0,))[None, None],
            (1, self.num_filters, 1),
        ).astype(self.np_float_dtype)
        self.unemp_conv_filters = np.exp(-self.f_ts / self.conv_lambdas[None, :, None])
        # Each state weights these filters differently.
        self.repeated_conv_weights = np.repeat(
            self.grouped_convolutional_filter_weights.reshape(
                self.num_us_states, self.num_filters
            )[:, :, np.newaxis],
            self.filter_len,
            axis=-1,
        )

        # For manually modulating SIR/Unemployment parameters
        self._beta_intercepts_modulation = 1
        self._beta_slopes_modulation = 1
        self._unemployment_modulation = 1

        # Economy-related
        # Interest rate for borrowing money from the federal reserve
        self.risk_free_interest_rate = self.np_float_dtype(risk_free_interest_rate)

        # Compute each worker's daily productivity when at work (to match 2019 GDP)
        # We assume the open/close stringency policy level was always at it's lowest
        # value (i.e., 1) before the pandemic started.
        num_unemployed_at_stringency_level_1 = self.unemployment_step(
            np.ones(self.num_us_states)
        )
        workforce = (
            self.us_population * pop_between_age_18_65
            - np.sum(num_unemployed_at_stringency_level_1)
        ).astype(self.np_int_dtype)
        workers_per_capita = (workforce / self.us_population).astype(
            self.np_float_dtype
        )
        gdp_per_worker = (self.gdp_per_capita / workers_per_capita).astype(
            self.np_float_dtype
        )
        self.num_days_in_an_year = 365
        self.daily_production_per_worker = (
            gdp_per_worker / self.num_days_in_an_year
        ).astype(self.np_float_dtype)

        self.infection_too_sick_to_work_rate = self.np_float_dtype(
            infection_too_sick_to_work_rate
        )
        assert 0 <= self.infection_too_sick_to_work_rate <= 1

        self.pop_between_age_18_65 = self.np_float_dtype(pop_between_age_18_65)
        assert 0 <= self.pop_between_age_18_65 <= 1

        # Compute max possible productivity values (used for agent reward normalization)
        max_productivity_t = self.economy_step(
            self.us_state_population,
            np.zeros((self.num_us_states), dtype=self.np_int_dtype),
            np.zeros((self.num_us_states), dtype=self.np_int_dtype),
            num_unemployed_at_stringency_level_1,
            infection_too_sick_to_work_rate=self.infection_too_sick_to_work_rate,
            population_between_age_18_65=self.pop_between_age_18_65,
        )
        self.maximum_productivity_t = max_productivity_t

        # Economic reward non-linearity
        self.economic_reward_crra_eta = self.np_float_dtype(economic_reward_crra_eta)
        assert 0.0 <= self.economic_reward_crra_eta < 20.0

        # Health indices are normalized by maximum annual GDP
        self.agents_health_norm = self.maximum_productivity_t * self.num_days_in_an_year
        self.planner_health_norm = np.sum(self.agents_health_norm)

        # Economic indices are normalized by maximum annual GDP
        self.agents_economic_norm = (
            self.maximum_productivity_t * self.num_days_in_an_year
        )
        self.planner_economic_norm = np.sum(self.agents_economic_norm)

        def scale_health_over_economic_index(health_priority_scaling, alphas):
            """
            Given starting alpha(s), compute new alphas so that the
            resulting alpha:1-alpha ratio is scaled by health_weightage
            """
            z = alphas / (1 - alphas)  # alphas = z / (1 + z)
            scaled_z = health_priority_scaling * z
            new_alphas = scaled_z / (1 + scaled_z)
            return new_alphas

        # Agents' health and economic index weightages
        # fmt: off
        self.weightage_on_marginal_agent_health_index = \
            scale_health_over_economic_index(
                health_priority_scaling_agents,
                self.inferred_weightage_on_agent_health_index,
            )
        # fmt: on
        assert (
            (self.weightage_on_marginal_agent_health_index >= 0)
            & (self.weightage_on_marginal_agent_health_index <= 1)
        ).all()
        self.weightage_on_marginal_agent_economic_index = (
            1 - self.weightage_on_marginal_agent_health_index
        )

        # Planner's health and economic index weightages
        # fmt: off
        self.weightage_on_marginal_planner_health_index = \
            scale_health_over_economic_index(
                health_priority_scaling_planner,
                self.inferred_weightage_on_planner_health_index,
            )
        # fmt: on
        assert 0 <= self.weightage_on_marginal_planner_health_index <= 1
        self.weightage_on_marginal_planner_economic_index = (
            1 - self.weightage_on_marginal_planner_health_index
        )

        # Normalization factor for the reward (often useful for RL training)
        self.reward_normalization_factor = reward_normalization_factor

        # CUDA-related attributes (for GPU simulations)
        # Note: these will be set / overwritten via the env_wrapper
        # use_cuda will be set to True (by the env_wrapper), if needed
        # to be simulated on the GPU
        self.use_cuda = False
        self.cuda_data_manager = None
        self.cuda_function_manager = None
        self.cuda_step = lambda *args, **kwargs: None
        self.cuda_compute_reward = lambda *args, **kwargs: None

        # Adding use_cuda to self.world for use in components
        self.world.use_cuda = self.use_cuda
        self.world.cuda_data_manager = self.cuda_data_manager
        self.world.cuda_function_manager = self.cuda_function_manager

    name = "CovidAndEconomySimulation"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    required_entities = []

    def reset_starting_layout(self):
        pass

    def reset_agent_states(self):
        self.world.clear_agent_locs()

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the GPU (device).
        """
        data_dict = DataFeed()
        # Global States
        data_dict.add_data(
            name="susceptible",
            data=self.world.global_state["Susceptible"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="infected",
            data=self.world.global_state["Infected"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="recovered",
            data=self.world.global_state["Recovered"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="deaths",
            data=self.world.global_state["Deaths"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="unemployed",
            data=self.world.global_state["Unemployed"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="vaccinated",
            data=self.world.global_state["Vaccinated"],
            save_copy_and_apply_at_reset=True,
        )
        # Actions
        data_dict.add_data(
            name="stringency_level",
            data=self.world.global_state["Stringency Level"].astype(self.np_int_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="subsidy_level",
            data=self.world.global_state["Subsidy Level"].astype(self.np_int_dtype),
            save_copy_and_apply_at_reset=True,
        )
        # Economy-related
        data_dict.add_data(
            name="subsidy",
            data=self.world.global_state["Subsidy"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="postsubsidy_productivity",
            data=self.world.global_state["Postsubsidy Productivity"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="productivity",
            data=np.zeros_like(
                self.world.global_state["Susceptible"], dtype=self.np_float_dtype
            ),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="incapacitated",
            data=np.zeros((self.num_us_states), dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="cant_work",
            data=np.zeros((self.num_us_states), dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="num_people_that_can_work",
            data=np.zeros((self.num_us_states), dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="us_state_population",
            data=self.us_state_population,
        )
        data_dict.add_data(
            name="infection_too_sick_to_work_rate",
            data=self.infection_too_sick_to_work_rate,
        )
        data_dict.add_data(
            name="population_between_age_18_65",
            data=self.pop_between_age_18_65,
        )
        data_dict.add_data(
            name="daily_production_per_worker",
            data=self.daily_production_per_worker,
        )
        data_dict.add_data(
            name="maximum_productivity",
            data=self.maximum_productivity_t,
        )
        # SIR-related
        data_dict.add_data(
            name="real_world_stringency_policy_history",
            data=(
                self._real_world_data["policy"][
                    self.start_date_index - self.beta_delay + 1 : self.start_date_index,
                    :,
                ]
            ).astype(self.np_int_dtype),
        )
        data_dict.add_data(
            name="beta_delay",
            data=self.beta_delay,
        )
        data_dict.add_data(
            name="beta_slopes",
            data=self.beta_slopes,
        )
        data_dict.add_data(
            name="beta_intercepts",
            data=self.beta_intercepts,
        )
        data_dict.add_data(
            name="beta",
            data=np.zeros((self.num_us_states), dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="gamma",
            data=self.gamma,
        )
        data_dict.add_data(
            name="death_rate",
            data=self.death_rate,
        )
        # Unemployment fit parameters
        data_dict.add_data(
            name="filter_len",
            data=self.filter_len,
        )
        data_dict.add_data(
            name="num_filters",
            data=self.num_filters,
        )
        data_dict.add_data(
            name="delta_stringency_level",
            data=(
                self.stringency_level_history[1:] - self.stringency_level_history[:-1]
            ).astype(self.np_int_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="grouped_convolutional_filter_weights",
            data=self.grouped_convolutional_filter_weights,
        )
        data_dict.add_data(
            name="unemp_conv_filters",
            data=self.unemp_conv_filters,
        )
        data_dict.add_data(
            name="unemployment_bias",
            data=self.unemployment_bias,
        )
        data_dict.add_data(
            name="signal",
            data=np.zeros(
                (self.n_agents, self.num_filters, self.filter_len),
                dtype=self.np_float_dtype,
            ),
            save_copy_and_apply_at_reset=True,
        )
        # Reward-related
        data_dict.add_data(
            name="min_marginal_agent_health_index",
            data=self.min_marginal_agent_health_index,
        )
        data_dict.add_data(
            name="max_marginal_agent_health_index",
            data=self.max_marginal_agent_health_index,
        )
        data_dict.add_data(
            name="min_marginal_agent_economic_index",
            data=self.min_marginal_agent_economic_index,
        )
        data_dict.add_data(
            name="max_marginal_agent_economic_index",
            data=self.max_marginal_agent_economic_index,
        )
        data_dict.add_data(
            name="min_marginal_planner_health_index",
            data=self.min_marginal_planner_health_index,
        )
        data_dict.add_data(
            name="max_marginal_planner_health_index",
            data=self.max_marginal_planner_health_index,
        )
        data_dict.add_data(
            name="min_marginal_planner_economic_index",
            data=self.min_marginal_planner_economic_index,
        )
        data_dict.add_data(
            name="max_marginal_planner_economic_index",
            data=self.max_marginal_planner_economic_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_agent_health_index",
            data=self.weightage_on_marginal_agent_health_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_agent_economic_index",
            data=self.weightage_on_marginal_agent_economic_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_planner_health_index",
            data=self.weightage_on_marginal_planner_health_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_planner_economic_index",
            data=self.weightage_on_marginal_planner_economic_index,
        )
        data_dict.add_data(
            name="value_of_life",
            data=self.value_of_life,
        )
        data_dict.add_data(
            name="economic_reward_crra_eta",
            data=self.economic_reward_crra_eta,
        )
        data_dict.add_data(
            name="num_days_in_an_year",
            data=self.num_days_in_an_year,
        )
        data_dict.add_data(
            name="risk_free_interest_rate",
            data=self.risk_free_interest_rate,
        )
        data_dict.add_data(
            name="agents_health_norm",
            data=self.agents_health_norm,
        )
        data_dict.add_data(
            name="agents_economic_norm",
            data=self.agents_economic_norm,
        )
        data_dict.add_data(
            name="planner_health_norm",
            data=self.planner_health_norm,
        )
        data_dict.add_data(
            name="planner_economic_norm",
            data=self.planner_economic_norm,
        )

        return data_dict

    def get_tensor_dictionary(self):
        """
        Create a dictionary of (Pytorch-accessible) data to push to the GPU (device).
        """
        tensor_dict = DataFeed()
        return tensor_dict

    def scenario_step(self):
        """
        Update the state of the USA based on the Covid-19 and Economy dynamics.
        This internally implements three steps
        - sir_step() - updates the susceptible, infected, recovered, deaths
        and vaccination numbers based on the SIR equations
        - unemployment_step() - uses the unemployment model to updates the unemployment
         based on the stringency levels
        - economy_step - computes the current producitivity numbers for the agents
        """
        if self.use_cuda:
            self.cuda_step(
                self.cuda_data_manager.device_data("susceptible"),
                self.cuda_data_manager.device_data("infected"),
                self.cuda_data_manager.device_data("recovered"),
                self.cuda_data_manager.device_data("deaths"),
                self.cuda_data_manager.device_data("vaccinated"),
                self.cuda_data_manager.device_data("unemployed"),
                self.cuda_data_manager.device_data("subsidy"),
                self.cuda_data_manager.device_data("productivity"),
                self.cuda_data_manager.device_data("stringency_level"),
                self.cuda_data_manager.device_data("num_stringency_levels"),
                self.cuda_data_manager.device_data("postsubsidy_productivity"),
                self.cuda_data_manager.device_data("num_vaccines_available_t"),
                self.cuda_data_manager.device_data(
                    "real_world_stringency_policy_history"
                ),
                self.cuda_data_manager.device_data("beta_delay"),
                self.cuda_data_manager.device_data("beta_slopes"),
                self.cuda_data_manager.device_data("beta_intercepts"),
                self.cuda_data_manager.device_data("beta"),
                self.cuda_data_manager.device_data("gamma"),
                self.cuda_data_manager.device_data("death_rate"),
                self.cuda_data_manager.device_data("incapacitated"),
                self.cuda_data_manager.device_data("cant_work"),
                self.cuda_data_manager.device_data("num_people_that_can_work"),
                self.cuda_data_manager.device_data("us_state_population"),
                self.cuda_data_manager.device_data("infection_too_sick_to_work_rate"),
                self.cuda_data_manager.device_data("population_between_age_18_65"),
                self.cuda_data_manager.device_data("filter_len"),
                self.cuda_data_manager.device_data("num_filters"),
                self.cuda_data_manager.device_data("delta_stringency_level"),
                self.cuda_data_manager.device_data(
                    "grouped_convolutional_filter_weights"
                ),
                self.cuda_data_manager.device_data("unemp_conv_filters"),
                self.cuda_data_manager.device_data("unemployment_bias"),
                self.cuda_data_manager.device_data("signal"),
                self.cuda_data_manager.device_data("daily_production_per_worker"),
                self.cuda_data_manager.device_data("maximum_productivity"),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_world-agent_state"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_world-agent_postsubsidy_productivity"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_world-lagged_stringency_level"
                ),
                self.cuda_data_manager.device_data(f"{_OBSERVATIONS}_a_time"),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_world-agent_state"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_world-agent_postsubsidy_productivity"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_world-lagged_stringency_