# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from numpy.random import rand

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class Gather(BaseComponent):
    """
    Allows mobile agents to move around the world and collect resources and prevents
    agents from moving to invalid locations.

    Can be configured to include collection skill, where agents have heterogeneous
    probabilities of collecting bonus resources without additional labor cost.

    Args:
        move_labor (float): Labor cost associated with movement. Must be >= 0.
            Default is 1.0.
        collect_labor (float): Labor cost associated with collecting resources. This
            cost is added (in addition to any movement cost) when the agent lands on
            a tile that is populated with resources (triggering collection).
            Must be >= 0. Default is 1.0.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a bonus prob of 0. "pareto" and
            "lognormal" sample skills from the associated distributions.
    """

    name = "Gather"
    required_entities = ["Coin", "House", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        move_labor=1.0,
        collect_labor=1.0,
        skill_dist="none",
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.move_labor = float(move_labor)
        assert self.move_labor >= 0

        self.collect_labor = float(collect_labor)
        assert self.collect_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pa