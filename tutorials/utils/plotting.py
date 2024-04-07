# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from ai_economist.foundation import landmarks, resources


def plot_map(maps, locs, ax=None, cmap_order=None):
    world_size = np.array(maps.get("Wood")).shape
    max_health = {"Wood": 1, "Stone": 1, "House": 1}
    n_agents = len(locs)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        ax.cla()
    tmp = np.zeros((3, world_size[0], world_size[1]))
    cmap = plt.get_cmap("jet", n_agents)

    if cmap_order is None:
        cmap_order = list(range(n_agents))
    else:
        cmap_order = list(cmap_order)
        assert len(cmap_order) == n_agents

    scenario_entities = [k for k in maps.keys() if "source" not in k.lower()]
    for entity in scenario_entities:
        if entity == "House":
            continue
        elif resources.has(entity):
            if resources.get(entity).collectible:
                map_ = (
                    resources.get(entity).color[:, None, None]
                    * np.array(maps.get(entity))[None]
                )
                map_ /= max_health[entity]
                tmp += map_
        elif landmarks.has(entity):
            map_ = (
                landmarks.get(entity).color[:, None, None]
                * np.array(maps.get(entity))[None]
            )
            tmp += map_
        else:
            continue

    if isinstance(maps, dict):
        house_idx = np.array(maps.get("House")["owner"])
        house_health = np.array(maps.get("House")["health"])
    else:
        house_idx = maps.get("House", owner=True)
        house_health = maps.get("House")
    for i in range(n_agents):
        houses = house_health * (house_idx == cmap_order[i])
        agent = np.zeros_like(houses)
        agent += houses
        col = np.array(cmap(i)[:3])
        map_ = col[:, None, None] * agent[None]
        tmp += map_

    tmp *= 0.7
    tmp += 0.3

    tmp = np.transpose(tmp, [1, 2, 0])
    tmp 