# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import random

import numpy as np

from ai_economist.foundation.base.registrar import Registry


class BaseAgent:
    """Base class for Agent classes.

    Instances of Agent classes are created for each agent in the environment. Agent
    instances are stateful, capturing location, inventory, endogenous variables,
    and any additional state fields created by environment components during
    construction (see BaseComponent.get_additional_state_fields in base_component.py).

    They also provide a simple API for getting/setting actions for each of their
    registered action subspaces (whic