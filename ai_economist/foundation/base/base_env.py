
# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import random
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from ai_economist.foundation.agents import agent_registry
from ai_economist.foundation.base.registrar import Registry
from ai_economist.foundation.base.world import World
from ai_economist.foundation.components import component_registry
from ai_economist.foundation.entities import (
    endogenous_registry,
    landmark_registry,
    resource_registry,
)


class BaseEnvironment(ABC):
    """
    Base Environment class. Should be used as the parent class for Scenario classes.
    Instantiates world, agent, and component objects.

    Provides Gym-style API for resetting and stepping:
        obs                  <-- env.reset()
        obs, rew, done, info <-- env.step(actions)

    Also provides Gym-style API for controlling random behavior:
        env.seed(seed) # Sets numpy and built-in RNG seeds to seed

    Reference: OpenAI Gym [https://github.com/openai/gym]

    Environments in this framework are instances of Scenario classes (which are built
    as extensions of BaseEnvironment). A Scenario must implement the following
    abstract methods (method docstrings provide detailed explanations):
        reset_starting_layout
        reset_agent_states
        scenario_step
        generate_observations
        compute_reward

    Scenario classes define their own passive dynamics--that is, dynamics that do not
    depend on agent actions--and supply observations. It is up to the Scenario class
    to handle reward.

    Interactions with the environment are handled through components, which define
    actions that agents can perform. Components are defined through distinct
    Component classes (which extend BaseComponent [see base_component.py]) and must
    be included in the components_registry in order to be used (see below).
    Components influence the environment dynamics through effects they have on
    agent/world states. They also (optionally) supply observations.

    The actions available to the agents, observations they receive, the dynamics of
    the environment, and the rewards depend of the choice of which Scenario class and
    Component class(es) to use.

    In multi_action_mode, an agent may choose an action for each of the action
    subspaces defined by the included Component classes. A Component can define 0, 1,
    or several action subspaces for a given agent type. If not using
    multi_action_mode, these action subspaces are combined into a single action space
    and the agent may select one action within this aggregated space.

    For additional detail regarding actions and action subspaces, see the
    BaseComponent class in base_component.py.

    There are 2 types of agents: mobile agents and the planner agent. There can be
    two or more mobile agents and a single planner agent. Conceptually, mobile agents
    represent the individual actors in the economic simulation while the planner
    agent represents a social planner that sets macroeconomic policy.

    This environment framework makes extensive use of Python classes. Scenarios,
    Components, Agents, and environment entities such as Resources, Landmarks,
    and Endogenous variables are all implemented as classes. These classes are
    accessed via registries. See top example.

    Example:
        from ai_economist import foundation
        # foundation.scenarios  <-- Scenario class registry
        # foundation.components <-- Component class registry
        # foundation.agents     <-- Agent class registry
        # foundation.resources  <-- Resource class registry
        # foundation.landmarks  <-- Landmark class registry
        # foundation.endogenous <-- Endogenous class registry

        # see ../scenarios/simple_wood_and_stone/dynamic_layout.py
        UniScenarioClass = foundation.scenarios.get("uniform/simple_wood_and_stone")

        # see ../components/build.py and ../components/move.py
        BuildComponentClass  = foundation.components.get("Build")
        GatherComponentClass = foundation.components.get("Gather")

    Example:
        from ai_economist import foundation
        from ai_economist.foundation.base.base_env import BaseEnvironment

        ScenarioClass = foundation.scenarios.get(...)
        assert issubclass(ScenarioClass, BaseEnvironment)

        env = ScenarioClass(
            components=[
                ("Build", {"payment": 20}),
                ("Gather", {"move_labor": 1.0, "collect_labor": 2.0}),
            ],
            n_agents=20,
            world_size=[25, 25],
        )

        obs = env.reset()

        actions = {agent.idx: ... for agent in env.all_agents}
        obs, rew, done, info = env.step(actions)

    Args:
        components (list): A list of tuples ("Component Name", {Component kwargs}) or
            list of dicts {"Component Name": {Component kwargs}} specifying the
            components that the instantiated environment will include.
            "Component Name" must be a string matching the name of a registered
            Component class.
            {Component kwargs} must be a dictionary of kwargs that can be passed as
            arguments to the Component class with name "Component Name".
            Resetting, stepping, and observation generation will be carried out in
            the order in which components are listed. This should be considered,
            as re-ordering the components list may impact the dynamics of the
            environment.
        n_agents (int): The number of mobile agents (does not include planner).
            Number of agents must be > 1.
        world_size (list): A length-2 list specifying the dimensions of the 2D world.
            Interpreted as [height, width].
        episode_length (int): Number of timesteps in a single episode.
        multi_action_mode_agents (bool): Whether mobile agents use multi_action_mode.
        multi_action_mode_planner (bool): Whether the planner uses multi_action_mode.
        flatten_observations (bool): Whether to preprocess observations by
            concatenating all scalar/vector observation subfields into a single
            "flat" observation field. If not, return observations as minimally
            processed dictionaries.
        flatten_masks (bool): Whether to flatten action masks into a single array or
            to keep as a {"action_subspace_name": action_subspace_mask} dictionary.
            For integration with deep RL, it is helpful to set this to True, for the
            purpose of action masking: flattened masks have the same semantics as
            policy logits.
        allow_observation_scaling (bool): Whether to enable certain observation
            fields to be scaled to a range better suited for deep RL.
        dense_log_frequency (int): [optional] How often (in completed episodes) to
            create a dense log while playing an episode. By default, dense logging is
            turned off (dense_log_frequency=None). If dense_log_frequency=20,
            a dense log will be created when the total episode count is a multiple of
            20.
            Dense logs provide a log of agent states, actions, and rewards at each
            timestep of an episode. They also log world states at a coarser timescale
            (see below). Component classes optionally contribute additional
            information to the dense log.
            Note: dense logging is time consuming (especially with many agents).
        world_dense_log_frequency (int): When dense logging, how often (in timesteps) to
            log a snapshot of the world state. If world_dense_log_frequency=50
            (the default), the world state will be included in the dense log for
            timesteps where t is a multiple of 50.
            Note: More frequent world snapshots increase the dense log memory footprint.
        seed (int, optional): If provided, sets the numpy and built-in random number
            generator seeds to seed. You can control the seed after env construction
            using the 'seed' method.
    """

    # The name associated with this Scenario class (must be unique)
    # Note: This is what will identify the Scenario class in the scenario registry.
    name = ""

    # The (sub)classes of agents that this scenario applies to
    agent_subclasses = []

    # The (non-agent) game entities that are expected to be in play
    required_entities = None  # Replace with list or tuple (can be empty)

    def __init__(
        self,
        components=None,
        n_agents=None,
        world_size=None,
        episode_length=1000,
        multi_action_mode_agents=False,
        multi_action_mode_planner=True,
        flatten_observations=True,
        flatten_masks=True,
        allow_observation_scaling=True,
        dense_log_frequency=None,
        world_dense_log_frequency=50,
        collate_agent_step_and_reset_data=False,
        seed=None,
    ):

        # Make sure a name was declared by child class
        assert self.name

        # Make sure the agent_subclasses was declared by child class
        # and does not create potential conflicts
        assert isinstance(self.agent_subclasses, (tuple, list))