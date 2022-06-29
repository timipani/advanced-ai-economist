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
        collect_labor (float): Labo