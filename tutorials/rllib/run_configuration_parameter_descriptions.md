Copyright (c) 2021, salesforce.com, inc.  
All rights reserved.  
SPDX-License-Identifier: BSD-3-Clause  
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# Introduction
This document describes the run configuration parameters used to implement two-level curriculum learning in the [paper](https://arxiv.org/abs/2108.02755): "The AI Economist: Optimal Economic Policy Design via Two-level Deep Reinforcement Learning".

The run configurations include the `environment`, `general`, `trainer` and the agent and planner `policy`-related parameters. 
It is helpful to first go through our [tutorial](../two_level_curriculum_learning_with_rllib.md) on two-level curriculum learning to understand how we used these configurations in conjunction with with a [training script](training_script.py) to perform training. For example configurations that we used in our two-phased training approach, please see the [phase one](phase1/config.yaml) and [phase two](phase2/config.yaml) configurations.


# Descriptions of the Run Configuration Parameters

## Environment

### Base Environment
- `allow_observation_scaling` (bool): Whether to enable certain observation fields to be scaled to a range better suited for deep RL. Defaults to True.
- `components` (list): A list of tuples ("Component Name", {Component kwargs}) or
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
- `dense_log_frequency` (int): [optional] How often (in completed episodes) to
    create a dense log while playing an episode. By default, dense logging is
    turned off (dense_log_frequency=None). If dense_log_frequency=20,
    a dense log will be created when the total episode count is a multiple of
    20.
    Dense logs provide a log of agent states, actions, and rewards at each
    timestep of an episode. They also log world states at a coarser timescale
    (see below). Component classes optionally contribute additional
    information to the dense log.
    Note: dense logging is time consuming (especially with many agents).    
- `episode_length` (int): Number of timesteps in a single episode.
- `flatten_masks` (bool): Whether to flatten action masks into a single array or
    to keep as a {"action_subspace_name": action_subspace_mask} dictionary.
    For integration with deep RL, it is helpful to set this to True, for the
    purpose of action masking: flattened masks have the same semantics as
    policy logits.
- `flatten_observations` (bool): Whether to preprocess observations by
    concatenating all scalar/vector observation subfields into a single
    "flat" observation field. If not, return observations as minimally
    processed dictionaries.
- `multi_action_mode_agents` (bool): Whether mobile agents use multi_action_mode.
- `multi_action_mode_planner` (bool): Whether the planner uses multi_action_mode.
- `n_agents` (int): The number of mobile agents (does not include planner).
    Number of agents must be > 1.
- `world_dense_log_frequency` (int): When dense logging, how often (in timesteps) to log a snapshot of the world state. If world_dense_log_frequency=50 (the default), the world state will be included in the dense log for timesteps where t is a multiple of 50. Note: More frequent world snapshots increase the dense log memory footprint.
- `world_size` (list): A length-2 list specifying the dimensions of the 2D world.
    Interpreted as [height, width].

### Build Component
- `build_labor` (float): Labor cost associated with building a house.
    Must be >= 0. Default is 10.
- `payment` (int): Default amount of coin agents earn from building.
    Must be >= 0. Default is 10.
- `payment_max_skill_multiplier` (int): Maximum skill multiplier that an agent
    can sample. Must be >= 1. Default is 1.
- `skill_dist` (str): Distribution type for sampling skills. Default ("none")
    gives all agents identical skill equal to a multiplier of 1. "pareto" and
    "lognormal" sample skills from the associated distributions.

### ContinuousDoubleAuction Component
- `max_bid_ask` (int): Maximum amount of coin that an agent can bid or ask for.
    Must be >= 1. Default is 10 coin.
- `max_num_orders` (int, optional): Maximum number of bids + asks that an agent can have open for a given resource. Must be >= 1. Default is no limit to number of orders.
- `order_labor` (float): Amount of labor incurred when an agent creates an order.
    Must be >= 0. Default is 0.25.
- `order_duration` (int): Number of environment timesteps before an unfilled
    bid/ask expires. Must be >= 1. Default is 50 timesteps.

### Gather Component
- `collect_labor` (float): Labor cost associated with collecting resources. This
    c