# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class Build(BaseComponent):
    """
    Allows mobile agents to build house landmarks in the world using stone and wood,
    earning income.

    Can be configured to include heterogeneous building skill where agents earn
    different levels of income when building.

    Args:
        payment (int): Default amount of coin agents earn from building.
            Must be >= 0. Default is 10.
        payment_max_skill_multiplier (int): Maximum skill multiplier that an agent
            can sample. Must be >= 1. Default is 1.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a multiplier of 1. "pareto" and
            "lognormal" sample skills from the associated distributions.
        build_labor (float): Labor cost associated with building a house.
            Must be >= 0. Default is 10.
    """

    name = "Build"
    component_type = "Build"
    required_entities = ["Wood", "Stone", "Coin", "House", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        payment=10,
        payment_max_skill_multiplier=1,
        skill_dist="none",
        build_labor=10.0,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment = int(payment)
        assert self.payment >= 0

        self.payment_max_skill_multiplier = int(payment_max_skill_multiplier)
        assert self.payment_max_skill_multiplier >= 1

        self.resource_cost = {"Wood": 1, "Stone": 1}

        self.build_labor = float(build_labor)
        assert self.build_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.sampled_skills = {}

        self.builds = []

    def agent_can_build(self, agent):
        """Return True if agent can actually build in its current location."""
        # See if the agent has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # Do nothing if this spot is already occupied by a landmark or resource
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        # If we made it here, the agent can build.
        return True

    # Required methods for implementing components
    # ------------------------------------