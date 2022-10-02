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
        Clumpiness: degree to which resources 