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
    registered action subspaces (which depend on the components used to build
    the environment).

    Args:
        idx (int or str): Index that uniquely identifies the agent object amongst the
            other agent objects registered in its environment.
        multi_action_mode (bool): Whether to allow the agent to take one action for
            each of its registered action subspaces each timestep (if True),
            or to limit the agent to take only one action each timestep (if False).
    """

    name = ""

    def __init__(self, idx=None, multi_action_mode=None):
        assert self.name

        if idx is None:
            idx = 0

        if multi_action_mode is None:
            multi_action_mode = False

        if isinstance(idx, str):
            self._idx = idx
        else:
            self._idx = int(idx)

        self.multi_action_mode = bool(multi_action_mode)
        self.single_action_map = (
            {}
        )  # Used to convert single-action-mode actions to the general format

        self.action = dict()
        self.action_dim = dict()
        self._action_names = []
        self