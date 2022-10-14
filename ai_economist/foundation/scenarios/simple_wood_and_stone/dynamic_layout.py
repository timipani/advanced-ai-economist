# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy

import numpy as np
from scipy import signal

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.scenarios.utils import rewards, social_metrics


@scenario_registry.add
class Uniform(BaseEnvironment):
    """
    World containing spatially-segregated stone and wood with stochastic regeneration.

    For controlling how resource regeneration behavior...
        Coverage: if fraction, target fraction of total tiles; if integer, target number
            of tiles
        Regen Halfwidth: width of regen kernel = 1 + (2 * halfwidth); set >0 to create
            a spatial social dilemma
        Regen Weight: regen probability per tile counted by the regen kernel
        Max Health: how many resource units can populate a source block
        Clumpiness: degree to which resources are spatially clustered
        Gradient Steepness: degree to which stone/wood are restricted to the top/bottom
            of the map

    Args:
        planner_gets_spatial_obs (bool): Whether the planner agent receives spatial
            observations from the world.
        full_observability (bool): Whether the mobile agents' spatial observation
            includes the full world view or is instead an egocentric view.
        mobile_agent_observation_range (int): If not using full_observability,
            the spatial range (on each side of the agent) that is visible in the
            spatial observations.
        starting_wood_coverage (int, float): Target coverage of wood at t=0.
        wood_regen_halfwidth (int): Regen halfwidth for wood.
        wood_regen_weight (float): Regen weight for wood.
        wood_max_health (int): Max wood units per wood source tile.
        wood_clumpiness (float): Degree of wood clumping.
        starting_stone_coverage (int, float): Target coverage of stone at t=0.
        stone_regen_halfwidth (int): Regen halfwidth for stone.
        stone_regen_weight (float): Regen weight for stone.
        stone_max_health (int): Max stone units per stone source tile.
        stone_clumpiness (float): Degree of stone clumping.
        gradient_steepness (int, float): How steeply source tile probability falls
            off from the top/bottom of the map.
        checker_source_blocks (bool): Whether to space source tiles in a "checker"
            formation.
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
            "decay" (d