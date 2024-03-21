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
    as re-ordering the component