
# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.agents import agent_registry
from ai_economist.foundation.entities import landmark_registry, resource_registry


class Maps:
    """Manages the spatial configuration of the world as a set of entity maps.

    A maps object is built during world construction, which is a part of environment
    construction. The maps object is accessible through the world object. The maps
    object maintains a map state for each of the spatial entities that are involved
    in the constructed environment (which are determined by the "required_entities"
    attributes of the Scenario and Component classes used to build the environment).

    The Maps class also implements some of the basic spatial logic of the game,
    such as which locations agents can occupy based on other agent locations and
    locations of various landmarks.

    Args:
        size (list): A length-2 list specifying the dimensions of the 2D world.
            Interpreted as [height, width].
        n_agents (int): The number of mobile agents (does not include planner).
        world_resources (list): The resources registered during environment
            construction.
        world_landmarks (list): The landmarks registered during environment
            construction.
    """

    def __init__(self, size, n_agents, world_resources, world_landmarks):
        self.size = size
        self.sz_h, self.sz_w = size

        self.n_agents = n_agents

        self.resources = world_resources
        self.landmarks = world_landmarks
        self.entities = world_resources + world_landmarks

        self._maps = {}  # All maps
        self._blocked = []  # Solid objects that no agent can move through
        self._private = []  # Solid objects that only permit movement for parent agents
        self._public = []  # Non-solid objects that agents can move on top of
        self._resources = []  # Non-solid objects that can be collected

        self._private_landmark_types = []
        self._resource_source_blocks = []

        self._map_keys = []

        self._accessibility_lookup = {}

        for resource in self.resources:
            resource_cls = resource_registry.get(resource)
            if resource_cls.collectible:
                self._maps[resource] = np.zeros(shape=self.size)
                self._resources.append(resource)
                self._map_keys.append(resource)

                self.landmarks.append("{}SourceBlock".format(resource))

        for landmark in self.landmarks:
            dummy_landmark = landmark_registry.get(landmark)()

            if dummy_landmark.public:
                self._maps[landmark] = np.zeros(shape=self.size)
                self._public.append(landmark)
                self._map_keys.append(landmark)

            elif dummy_landmark.blocking:
                self._maps[landmark] = np.zeros(shape=self.size)
                self._blocked.append(landmark)
                self._map_keys.append(landmark)
                self._accessibility_lookup[landmark] = len(self._accessibility_lookup)

            elif dummy_landmark.private:
                self._private_landmark_types.append(landmark)
                self._maps[landmark] = dict(
                    owner=-np.ones(shape=self.size, dtype=np.int16),
                    health=np.zeros(shape=self.size),
                )
                self._private.append(landmark)
                self._map_keys.append(landmark)
                self._accessibility_lookup[landmark] = len(self._accessibility_lookup)

            else: