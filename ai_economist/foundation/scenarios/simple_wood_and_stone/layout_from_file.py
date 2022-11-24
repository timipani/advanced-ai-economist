
# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy import signal

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.scenarios.utils import rewards, social_metrics


@scenario_registry.add
class LayoutFromFile(BaseEnvironment):
    """
    World containing stone and wood with stochastic regeneration. Refers to a fixed
    layout file (see ./map_txt/ for examples) to determine the spatial arrangement of
    stone, wood, and water tiles.

    Args:
        planner_gets_spatial_obs (bool): Whether the planner agent receives spatial
            observations from the world.
        full_observability (bool): Whether the mobile agents' spatial observation
            includes the full world view or is instead an egocentric view.
        mobile_agent_observation_range (int): If not using full_observability,
            the spatial range (on each side of the agent) that is visible in the
            spatial observations.
        env_layout_file (str): Name of the layout file in ./map_txt/ to use.
            Note: The world dimensions of that layout must match the world dimensions
            argument used to construct the environment.
        resource_regen_prob (float): Probability that an empty source tile will
            regenerate a new resource unit.
        fixed_four_skill_and_loc (bool): Whether to use a fixed set of build skills and
            starting locations, with agents grouped into starting locations based on
            which skill quartile they are in. False, by default.
            True, for experiments in https://arxiv.org/abs/2004.13332.
            Note: Requires that the environment uses the "Build" component with
            skill_dist="pareto".
        starting_agent_coin (int, float): Amount of coin agents have at t=0. Defaults
            to zero coin.
        isoelastic_eta (float): Parameter controlling the shape of agent utility
            wrt coin endowment.
        energy_cost (float): Coefficient for converting labor to negative utility.
        energy_warmup_constant (float): Decay constant that controls the rate at which
            the effective energy cost is annealed from 0 to energy_cost. Set to 0
            (default) to disable annealing, meaning that the effective energy cost is
            always energy_cost. The units of the decay constant depend on the choice of
            energy_warmup_method.
        energy_warmup_method (str): How to schedule energy annealing (warmup). If
            "decay" (default), use the number of completed episodes. If "auto",
            use the number of timesteps where the average agent reward was positive.
        planner_reward_type (str): The type of reward used for the planner. Options
            are "coin_eq_times_productivity" (default),
            "inv_income_weighted_coin_endowment", and "inv_income_weighted_utility".
        mixing_weight_gini_vs_coin (float): Degree to which equality is ignored w/
            "coin_eq_times_productivity". Default is 0, which weights equality and
            productivity equally. If set to 1, only productivity is rewarded.
    """

    name = "layout_from_file/simple_wood_and_stone"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = ["Wood", "Stone", "Water"]

    def __init__(
        self,
        *base_env_args,
        planner_gets_spatial_info=True,
        full_observability=False,
        mobile_agent_observation_range=5,
        env_layout_file="quadrant_25x25_20each_30clump.txt",
        resource_regen_prob=0.01,
        fixed_four_skill_and_loc=False,
        starting_agent_coin=0,
        isoelastic_eta=0.23,
        energy_cost=0.21,
        energy_warmup_constant=0,
        energy_warmup_method="decay",
        planner_reward_type="coin_eq_times_productivity",
        mixing_weight_gini_vs_coin=0.0,
        **base_env_kwargs,
    ):
        super().__init__(*base_env_args, **base_env_kwargs)

        # Whether agents receive spatial information in their observation tensor
        self._planner_gets_spatial_info = bool(planner_gets_spatial_info)

        # Whether the (non-planner) agents can see the whole world map
        self._full_observability = bool(full_observability)

        self._mobile_agent_observation_range = int(mobile_agent_observation_range)

        # Load in the layout
        path_to_layout_file = Path(f"{Path(__file__).parent}/map_txt/{env_layout_file}")

        with open(path_to_layout_file, "r") as f:
            self.env_layout_string = f.read()
            self.env_layout = self.env_layout_string.split(";")

        # Convert the layout to landmark maps
        landmark_lookup = {"W": "Wood", "S": "Stone", "@": "Water"}
        self._source_maps = {
            r: np.zeros(self.world_size) for r in landmark_lookup.values()
        }
        for r, symbol_row in enumerate(self.env_layout):
            for c, symbol in enumerate(symbol_row):
                landmark = landmark_lookup.get(symbol, None)
                if landmark:
                    self._source_maps[landmark][r, c] = 1

        # For controlling how resource regeneration behavior
        self.layout_specs = dict(
            Wood={
                "regen_weight": float(resource_regen_prob),
                "regen_halfwidth": 0,
                "max_health": 1,
            },
            Stone={
                "regen_weight": float(resource_regen_prob),
                "regen_halfwidth": 0,
                "max_health": 1,
            },
        )
        assert 0 <= self.layout_specs["Wood"]["regen_weight"] <= 1
        assert 0 <= self.layout_specs["Stone"]["regen_weight"] <= 1

        # How much coin do agents begin with at upon reset
        self.starting_agent_coin = float(starting_agent_coin)
        assert self.starting_agent_coin >= 0.0

        # Controls the diminishing marginal utility of coin.
        # isoelastic_eta=0 means no diminishing utility.
        self.isoelastic_eta = float(isoelastic_eta)
        assert 0.0 <= self.isoelastic_eta <= 1.0

        # The amount that labor is weighted in utility computation
        # (once annealing is finished)
        self.energy_cost = float(energy_cost)
        assert self.energy_cost >= 0

        # Which method to use for calculating the progress of energy annealing
        # If method = 'decay': #completed episodes
        # If method = 'auto' : #timesteps where avg. agent reward > 0
        self.energy_warmup_method = energy_warmup_method.lower()
        assert self.energy_warmup_method in ["decay", "auto"]
        # Decay constant for annealing to full energy cost
        # (if energy_warmup_constant == 0, there is no annealing)
        self.energy_warmup_constant = float(energy_warmup_constant)
        assert self.energy_warmup_constant >= 0
        self._auto_warmup_integrator = 0

        # Which social welfare function to use
        self.planner_reward_type = str(planner_reward_type).lower()

        # How much to weight equality if using SWF=eq*prod:
        # 0 -> SWF=eq * prod
        # 1 -> SWF=prod
        self.mixing_weight_gini_vs_coin = float(mixing_weight_gini_vs_coin)
        assert 0 <= self.mixing_weight_gini_vs_coin <= 1.0

        # Use this to calculate marginal changes and deliver that as reward
        self.init_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.prev_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.curr_optimization_metric = {agent.idx: 0 for agent in self.all_agents}

        """
        Fixed Four Skill and Loc
        ------------------------
        """
        self.agent_starting_pos = {agent.idx: [] for agent in self.world.agents}

        self.fixed_four_skill_and_loc = bool(fixed_four_skill_and_loc)
        if self.fixed_four_skill_and_loc:
            bm = self.get_component("Build")
            assert bm.skill_dist == "pareto"
            pmsm = bm.payment_max_skill_multiplier

            # Temporarily switch to a fixed seed for controlling randomness
            seed_state = np.random.get_state()
            np.random.seed(seed=1)

            # Generate a batch (100000) of num_agents (sorted/clipped) Pareto samples.
            pareto_samples = np.random.pareto(4, size=(100000, self.n_agents))
            clipped_skills = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
            sorted_clipped_skills = np.sort(clipped_skills, axis=1)
            # The skill level of the i-th skill-ranked agent is the average of the
            # i-th ranked samples throughout the batch.
            average_ranked_skills = sorted_clipped_skills.mean(axis=0)
            self._avg_ranked_skill = average_ranked_skills * bm.payment

            np.random.set_state(seed_state)

            # Fill in the starting location associated with each skill rank
            starting_ranked_locs = [
                # Worst group of agents goes in top right
                (0, self.world_size[1] - 1),
                # Second-worst group of agents goes in bottom left
                (self.world_size[0] - 1, 0),
                # Second-best group of agents goes in top left
                (0, 0),
                # Best group of agents goes in bottom right
                (self.world_size[1] - 1, self.world_size[1] - 1),
            ]
            self._ranked_locs = []

            # Based on skill, assign each agent to one of the location groups
            skill_groups = np.floor(
                np.arange(self.n_agents) * (4 / self.n_agents),
            ).astype(np.int)
            n_in_group = np.zeros(4, dtype=np.int)
            for g in skill_groups:
                # The position within the group is given by the number of agents
                # counted in the group thus far.
                g_pos = n_in_group[g]

                # Top right
                if g == 0:
                    r = starting_ranked_locs[g][0] + (g_pos // 4)
                    c = starting_ranked_locs[g][1] - (g_pos % 4)
                    self._ranked_locs.append((r, c))

                # Bottom left
                elif g == 1:
                    r = starting_ranked_locs[g][0] - (g_pos // 4)
                    c = starting_ranked_locs[g][1] + (g_pos % 4)
                    self._ranked_locs.append((r, c))

                # Top left
                elif g == 2:
                    r = starting_ranked_locs[g][0] + (g_pos // 4)
                    c = starting_ranked_locs[g][1] + (g_pos % 4)
                    self._ranked_locs.append((r, c))

                # Bottom right
                elif g == 3:
                    r = starting_ranked_locs[g][0] - (g_pos // 4)
                    c = starting_ranked_locs[g][1] - (g_pos % 4)
                    self._ranked_locs.append((r, c))

                else:
                    raise ValueError

                # Count the agent we just placed.
                n_in_group[g] = n_in_group[g] + 1

    @property
    def energy_weight(self):
        """
        Energy annealing progress. Multiply with self.energy_cost to get the
        effective energy coefficient.
        """
        if self.energy_warmup_constant <= 0.0:
            return 1.0

        if self.energy_warmup_method == "decay":
            return float(1.0 - np.exp(-self._completions / self.energy_warmup_constant))

        if self.energy_warmup_method == "auto":
            return float(
                1.0
                - np.exp(-self._auto_warmup_integrator / self.energy_warmup_constant)
            )

        raise NotImplementedError

    def get_current_optimization_metrics(self):