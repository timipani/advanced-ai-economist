# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import random
from pathlib import Path

import numpy as np
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda_driver
import scipy
import scipy.stats
import torch
from pycuda.compiler import SourceModule
from torch.distributions import Categorical
from tqdm import tqdm

from .constants import (
    consumer_state_scaling_factors,
    firm_state_scaling_factors,
    govt_state_scaling_factors,
)
from .networks import DeterministicPolicy, IndependentPolicyNet, PolicyNet
from .util import expand_to_digit_form, size_after_digit_expansion

_NP_DTYPE = np.float32

# the below line is 'strangely' necessary to make PyTorch work with PyCUDA
pytorch_cuda_init_success = torch.cuda.FloatTensor(8)


# for opening source files within module
module_path = Path(__file__).parent


def interval_list_contains(interval_list, step):
    for (lower, upper_non_inclusive) in interval_list:
        if lower <= step < upper_non_inclusive:
            return True
    return False


class NoOpOptimizer:
    """
    Dummy Optimizer.
    """

    def __init__(self):
        pass

    def step(self):
        pass


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reverse_cumsum(x):
    # assumes summing along episode iteration dim
    return x + torch.sum(x, dim=-2, keepdims=True) - torch.cumsum(x, dim=-2)


def discounted_returns(rewards, gamma):
    maxt = rewards.shape[-2]
    cumulative_rewards = 0
    returns = torch.zeros_like(rewards)
    for t in reversed(range(maxt)):
        returns[:, t, :] = rewards[:, t, :] + gamma * cumulative_rewards
        cumulative_rewards = rewards[:, t, :] + cumulative_rewards
    return returns


def compute_theta_coef(hparams_dict, episode):
    anneal_dict = hparams_dict["agents"]["consumer_anneal_theta"]
    if anneal_dict["anneal_on"]:
        exp_decay_length_in_steps = anneal_dict["exp_decay_length_in_steps"]
        theta_coef = np.float32(1.0 - (np.exp(-episode / exp_decay_length_in_steps)))
    else:
        return np.float32(1.0)
    return theta_coef


def government_action_mask(hparams_dict, step):
    government_actions_array = hparams_dict["agents"]["government_actions_array"]
    tax_annealing_params = hparams_dict["agents"]["government_anneal_taxes"]

    income_tax = torch.tensor(government_actions_array[:, 0]).cuda()
    corporate_tax = torch.tensor(government_actions_array[:, 1]).cuda()
    mask = torch.zeros(income_tax.shape[0]).cuda()

    if not tax_annealing_params["anneal_on"]:
        return None
    a0 = tax_annealing_params["start"]
    max_tax = tax_annealing_params["increase_const"] * step + a0
    mask[(income_tax > max_tax) | (corporate_tax > max_tax)] -= 1000.0

    return mask


def firm_action_mask(hparams_dict, step):
    # pick out all firm actions where wage is the wrong height,
    # and assign -1000.0 to those
    firm_actions_array = hparams_dict["agents"]["firm_actions_array"]
    wage_annealing_params = hparams_dict["agents"]["firm_anneal_wages"]
    price_annealing_params = hparams_dict["agents"]["firm_anneal_prices"]
    wages = torch.tensor(firm_actions_array[:, 1]).cuda()
    prices = torch.tensor(firm_actions_array[:, 0]).cuda()
    mask = torch.zeros(wages.shape[0]).cuda()

    if not (wage_annealing_params["anneal_on"] or price_annealing_params["anneal_on"]):
        return None

    if wage_annealing_params["anneal_on"]:
        a0 = wage_annealing_params["start"]
        max_wage = wage_annealing_params["increase_const"] * step + a0
        min_wage = -wage_annealing_params["decrease_const"] * step + a0
        mask[(wages < min_wage) | (wages > max_wage)] -= 1000.0
    if price_annealing_params["anneal_on"]:
        a0 = price_annealing_params["start"]
        max_price = price_annealing_params["increase_const"] * step + a0
        min_price = -price_annealing_params["decrease_const"] * step + a0
        mask[(prices < min_price) | (prices > max_price)] -= 1000.0

    return mask


def get_cuda_code(rel_path_to_cu_file, **preprocessor_vars_to_replace):
    with open(module_path / rel_path_to_cu_file) as cudasource:
        code_string = cudasource.read()

    # format for preprocessor macros in firm_rbc.cu is M_VARNAME.
    # Specify all these as args to nvcc.
    options_list = [
        f"-D M_{k.upper()}={v}" for k, v in preprocessor_vars_to_replace.items()
    ]

    return code_string, options_list


def add_penalty_for_no_ponzi(
    states, rewards, budget_offset, penalty_coef=20.0, penalty_scale=100.0
):
    budget_violations = -torch.clamp_max(states[..., budget_offset], 0.0)
    rewards[:, -1, :] -= penalty_coef * budget_violations / penalty_scale


def update_government_rewards(
    government_rewards, consumer_rewards, firm_rewards, cfg_dict
):
    assert (
        government_rewards == 0.0
    ).all()  # govt should have been assigned exactly 0 in cuda step function
    total_rewards = consumer_rewards.sum(dim=-1)
    if cfg_dict["agents"]["government_counts_firm_reward"] == 1:
        total_rewards = total_rewards + cfg_dict["agents"].get(
            "firm_reward_for_government_factor", 1.0
        ) * firm_rewards.sum(dim=-1)

    government_rewards[..., 0] = total_rewards[:]  # one govt for now


def update_penalty_coef(
    states,
    budget_offset,
    prev_penalty_coef,
    penalty_step_size=0.01,
    penalty_scale=100.0,
):
    budget_violations = -torch.clamp_max(states[..., budget_offset], 0.0)
    new_penalty_coef = (
        prev_penalty_coef
        + penalty_step_size * (budget_violations / penalty_scale).mean().item()
    )
    return new_penalty_coef


def get_actions_from_inds(action_inds, agents_dict):

    _action_inds = action_inds.cpu().to(torch.long)

    consumption_action_tensor = torch.tensor(
        agents_dict["consumer_consumption_actions_array"]
    )

    work_action_tensor = torch.tensor(agents_dict["consumer_work_actions_array"])
    num_firms = agents_dict["num_firms"]
    out_shape = _action_inds.shape[:-1] + (agents_dict["consumer_action_dim"],)
    consumer_actions_out = torch.zeros(out_shape)
    idx_hours_worked = num_firms
    idx_which_firm = num_firms + 1

    for i in range(num_firms):
        consumer_actions_out[..., i] = consumption_action_tensor[
            _action_inds[..., i], :
        ].squeeze(dim=-1)

    consumer_actions_out[..., num_firms] = work_action_tensor[
        _action_inds[..., idx_hours_worked], :
    ].squeeze(dim=-1)

    consumer_actions_out[..., (num_firms + 1)] = _action_inds[..., idx_which_firm]

    return consumer_actions_out


def anneal_entropy_coef(entropy_dict, step):
    if entropy_dict is None:
        return 1.0

    if entropy_dict["anneal_on"]:
        coef_floor = entropy_dict.get("coef_floor", 0.0)
        return max(
            np.exp(-step / entropy_dict["exp_decay_length_in_steps"]), coef_floor
        )
    return 1.0


def get_grad_norm(policy):
    grad_norm = 0.0
    for p in list(filter(lambda p: p.grad is not None, policy.parameters())):
        grad_norm += (p.grad.data.norm(2).item()) ** 2
    return grad_norm


def get_ev(adv, returns, cutoff=-1.0):
    return max(cutoff, (1 - (adv.detach().var() / returns.detach().var())).item())


def consumer_ppo_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    ppo_num_updates=3,
    reward_scale=1.0,
    clip_grad_norm=None,
    clip_param=0.1,
):
    # Get initial policy predictions
    multi_action_probs, old_value_preds = policy(states)

    old_value_preds = old_value_preds.detach()
    # Get returns
    rescaled_rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rescaled_rewards, gamma_const)

    # Value function loss
