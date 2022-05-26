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
        self._multi_action_dict = {}
        self._unique_actions = 0
        self._total_actions = 0

        self.state = dict(loc=[0, 0], inventory={}, escrow={}, endogenous={})

        self._registered_inventory = False
        self._registered_endogenous = False
        self._registered_components = False
        self._noop_action_dict = dict()

        # Special flag to allow logic for multi-action-mode agents
        # that are not given any actions.
        self._passive_multi_action_agent = False

        # If this gets set to true, we can make masks faster
        self._one_component_single_action = False
        self._premask = None

    @property
    def idx(self):
        """Index used to identify this agent. Must be unique within the environment."""
        return self._idx

    def register_inventory(self, resources):
        """Used during environment construction to populate inventory/escrow fields."""
        assert not self._registered_inventory
        for entity_name in resources:
            self.inventory[entity_name] = 0
            self.escrow[entity_name] = 0
        self._registered_inventory = True

    def register_endogenous(self, endogenous):
        """Used during environment construction to populate endogenous state fields."""
        assert not self._registered_endogenous
        for entity_name in endogenous:
            self.endogenous[entity_name] = 0
        self._registered_endogenous = True

    def _incorporate_component(self, action_name, n):
        extra_n = (
            1 if self.multi_action_mode else 0
        )  # Each sub-action has a NO-OP in multi action mode)
        self.action[action_name] = 0
        self.action_dim[action_name] = n + extra_n
        self._action_names.append(action_name)
        self._multi_action_dict[action_name] = False
        self._unique_actions += 1
        if self.multi_action_mode:
            self._total_actions += n + extra_n
        else:
            for action_n in range(1, n + 1):
                self._total_actions += 1
                self.single_action_map[int(self._total_actions)] = [
                    action_name,
                    action_n,
      