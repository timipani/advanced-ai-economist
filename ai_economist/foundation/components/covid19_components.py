
# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime

import GPUtil
import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

try:
    num_gpus_available = len(GPUtil.getAvailable())
    print(f"Inside covid19_components.py: {num_gpus_available} GPUs are available.")
    if num_gpus_available == 0:
        print("No GPUs found! Running the simulation on a CPU.")
    else:
        from warp_drive.utils.constants import Constants
        from warp_drive.utils.data_feed import DataFeed

        _OBSERVATIONS = Constants.OBSERVATIONS
        _ACTIONS = Constants.ACTIONS
except ModuleNotFoundError:
    print(
        "Warning: The 'WarpDrive' package is not found and cannot be used! "
        "If you wish to use WarpDrive, please run "
        "'pip install rl-warp-drive' first."
    )
except ValueError:
    print("No GPUs found! Running the simulation on a CPU.")


@component_registry.add
class ControlUSStateOpenCloseStatus(BaseComponent):
    """
    Sets the open/close stringency levels for states.
    Args:
        n_stringency_levels (int): number of stringency levels the states can chose
            from. (Must match the number in the model constants dictionary referenced by
            the parent scenario.)
        action_cooldown_period (int): action cooldown period in days.
            Once a stringency level is set, the state(s) cannot switch to another level
            for a certain number of days (referred to as the "action_cooldown_period")
    """

    name = "ControlUSStateOpenCloseStatus"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        n_stringency_levels=10,
        action_cooldown_period=28,
        **base_component_kwargs,
    ):

        self.action_cooldown_period = action_cooldown_period
        super().__init__(*base_component_args, **base_component_kwargs)
        self.np_int_dtype = np.int32

        self.n_stringency_levels = int(n_stringency_levels)
        assert self.n_stringency_levels >= 2
        self._checked_n_stringency_levels = False

        self.masks = dict()
        self.default_agent_action_mask = [1 for _ in range(self.n_stringency_levels)]
        self.no_op_agent_action_mask = [0 for _ in range(self.n_stringency_levels)]
        self.masks["a"] = np.repeat(
            np.array(self.no_op_agent_action_mask)[:, np.newaxis],
            self.n_agents,
            axis=-1,
        )

        # (This will be overwritten during reset; see below)
        self.action_in_cooldown_until = None

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        # Store the times when the next set of actions can be taken.
        self.action_in_cooldown_until = np.array(
            [self.world.timestep for _ in range(self.n_agents)]
        )

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.n_stringency_levels
        return None

    def generate_masks(self, completions=0):
        for agent in self.world.agents:
            if self.world.use_real_world_policies:
                self.masks["a"][:, agent.idx] = self.default_agent_action_mask
            else:
                if self.world.timestep < self.action_in_cooldown_until[agent.idx]:
                    # Keep masking the actions
                    self.masks["a"][:, agent.idx] = self.no_op_agent_action_mask
                else:  # self.world.timestep == self.action_in_cooldown_until[agent.idx]
                    # Cooldown period has ended; unmask the "subsequent" action
                    self.masks["a"][:, agent.idx] = self.default_agent_action_mask
        return self.masks

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the GPU (device).
        """