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

    sum_old_log_probs = 0.0
    for action_ind, probs in enumerate(multi_action_probs):
        _CategoricalDist = Categorical(probs)
        sum_old_log_probs += -1.0 * _CategoricalDist.log_prob(actions[..., action_ind])
    sum_old_log_probs = sum_old_log_probs.detach()

    assert not G_discounted_returns.requires_grad
    assert not sum_old_log_probs.requires_grad
    assert not old_value_preds.requires_grad

    # Compute ppo loss
    for _ in range(ppo_num_updates):
        multi_action_probs, value_preds = policy(states)
        get_huber_loss = torch.nn.SmoothL1Loss()
        value_pred_clipped = old_value_preds + (value_preds - old_value_preds).clamp(
            -clip_param, clip_param
        )
        value_loss_new = get_huber_loss(
            value_preds.squeeze(dim=-1), G_discounted_returns
        )  # can use huber loss instead
        value_loss_clipped = get_huber_loss(
            value_pred_clipped.squeeze(dim=-1), G_discounted_returns
        )

        value_loss = torch.max(value_loss_new, value_loss_clipped).mean()

        # Policy loss with value function baseline.
        advantages = G_discounted_returns - value_preds.detach().squeeze(dim=-1)
        # Don't propagate through to VF network.
        assert not advantages.requires_grad

        # Trick: standardize advantages
        standardized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-6
        )
        sum_mean_entropy = 0.0  # mean over batch and agents
        sum_neg_log_probs = 0.0

        for action_ind, probs in enumerate(multi_action_probs):
            _CategoricalDist = Categorical(probs)
            sum_neg_log_probs += -1.0 * _CategoricalDist.log_prob(
                actions[..., action_ind]
            )
            sum_mean_entropy += _CategoricalDist.entropy().mean()

        assert sum_neg_log_probs.requires_grad
        # note: log probs are negative, so negate again here
        ratio = torch.exp(-sum_neg_log_probs + sum_old_log_probs)
        surr1 = ratio * standardized_advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            * standardized_advantages
        )

        ppo_loss = -torch.min(surr1, surr2).mean()

        loss = (
            ppo_loss - entropy_val * sum_mean_entropy + value_loss_weight * value_loss
        )

        # Apply gradients
        optimizer.zero_grad()
        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

        optimizer.step()


def ppo_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    ppo_num_updates=3,
    actions_mask=None,
    reward_scale=1.0,
    clip_grad_norm=None,
    clip_param=0.1,
):
    # Get initial policy predictions
    probs, old_value_preds = policy(states, actions_mask=actions_mask)
    old_value_preds = old_value_preds.detach()

    # Get returns
    rescaled_rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rescaled_rewards, gamma_const)

    # Value function loss

    _CategoricalDist = Categorical(probs)
    old_log_probs = -1.0 * _CategoricalDist.log_prob(actions).detach()

    assert not G_discounted_returns.requires_grad
    assert not old_log_probs.requires_grad
    assert not old_value_preds.requires_grad

    # Compute ppo loss
    for _ in range(ppo_num_updates):
        probs, value_preds = policy(states, actions_mask=actions_mask)
        get_huber_loss = torch.nn.SmoothL1Loss()
        value_pred_clipped = old_value_preds + (value_preds - old_value_preds).clamp(
            -clip_param, clip_param
        )
        value_loss_new = get_huber_loss(
            value_preds.squeeze(dim=-1), G_discounted_returns
        )  # can use huber loss instead
        value_loss_clipped = get_huber_loss(
            value_pred_clipped.squeeze(dim=-1), G_discounted_returns
        )

        value_loss = torch.max(value_loss_new, value_loss_clipped).mean()

        # Policy loss with value function baseline.
        advantages = G_discounted_returns - value_preds.detach().squeeze(dim=-1)
        # Don't propagate through to VF network.
        assert not advantages.requires_grad

        # Trick: standardize advantages
        standardized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-6
        )

        _CategoricalDist = Categorical(probs)
        neg_log_probs = -1.0 * _CategoricalDist.log_prob(actions)
        mean_entropy = _CategoricalDist.entropy().mean()

        assert neg_log_probs.requires_grad
        # note: log probs are negative, so negate again here
        ratio = torch.exp(-neg_log_probs + old_log_probs)
        surr1 = ratio * standardized_advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            * standardized_advantages
        )

        ppo_loss = -torch.min(surr1, surr2).mean()

        loss = ppo_loss - entropy_val * mean_entropy + value_loss_weight * value_loss

        # Apply gradients
        optimizer.zero_grad()
        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

        optimizer.step()


def consumer_policy_gradient_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    reward_scale=1.0,
    clip_grad_norm=None,
):
    # Get policy and value predictions
    multi_action_probs, value_preds = policy(states)

    # Get returns
    rescaled_rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rescaled_rewards, gamma_const)

    # Value function loss
    get_huber_loss = torch.nn.SmoothL1Loss()
    value_loss = get_huber_loss(
        value_preds.squeeze(dim=-1), G_discounted_returns
    ).mean()  # can use huber loss instead

    # Policy loss with value function baseline.
    advantages = G_discounted_returns - value_preds.detach().squeeze(dim=-1)
    # Don't propagate through to VF network.
    assert not advantages.requires_grad

    # Trick: standardize advantages
    standardized_advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-6
    )

    # Compute policy loss
    sum_mean_entropy = 0.0  # mean over batch and agents
    sum_neg_log_probs = 0.0

    for action_ind, probs in enumerate(multi_action_probs):
        _CategoricalDist = Categorical(probs)
        sum_neg_log_probs += -1.0 * _CategoricalDist.log_prob(actions[..., action_ind])
        sum_mean_entropy += _CategoricalDist.entropy().mean()

    pg_loss = (sum_neg_log_probs * standardized_advantages).mean()
    assert sum_neg_log_probs.requires_grad

    loss = pg_loss - entropy_val * sum_mean_entropy + value_loss_weight * value_loss

    # Apply gradients
    optimizer.zero_grad()
    loss.backward()

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

    optimizer.step()


def policy_gradient_step(
    policy,
    states,
    actions,
    rewards,
    optimizer,
    gamma_const,
    entropy_val=0.0,
    value_loss_weight=1.0,
    actions_mask=None,
    reward_scale=1.0,
    clip_grad_norm=None,
):

    # here, we must perform digit scaling
    optimizer.zero_grad()
    probs, value_preds = policy(states, actions_mask=actions_mask)
    rewards = rewards / reward_scale
    G_discounted_returns = discounted_returns(rewards, gamma_const)
    get_huber_loss = torch.nn.SmoothL1Loss()
    value_loss = get_huber_loss(
        value_preds.squeeze(dim=-1), G_discounted_returns
    ).mean()  # can use huber loss instead
    advantages = G_discounted_returns - value_preds.detach().squeeze(
        dim=-1
    )  # compute advantages (don't propagate through to VF network)
    assert not advantages.requires_grad
    # mean and standardize advantages
    standardized_advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-6
    )
    assert not standardized_advantages.requires_grad
    m = Categorical(probs)
    pg_loss = (-m.log_prob(actions) * standardized_advantages).mean()
    assert pg_loss.requires_grad
    entropy_regularize = entropy_val * m.entropy().mean()
    loss = pg_loss - entropy_regularize + value_loss_weight * value_loss
    loss.backward()

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)

    optimizer.step()


def save_dense_log(
    save_dir,
    epi,
    agent_type_arrays,
    agent_action_arrays,
    agent_aux_arrays,
):
    print(f"Saving dense log at episode {epi}")
    for agent_type in ["consumer", "firm", "government"]:
        states_batch, actions_batch, rewards_batch = agent_type_arrays[agent_type]
        aux_array = agent_aux_arrays[agent_type]
        if aux_array is not None:
            aux_array = aux_array.cpu().numpy()
        np.savez(
            str(Path(save_dir) / Path(f"episode_{epi}_{agent_type}.npz")),
            states=states_batch.cpu().numpy(),
            actions=actions_batch.cpu().numpy(),
            rewards=rewards_batch.cpu().numpy(),
            action_array=agent_action_arrays[agent_type],
            aux_array=aux_array,
        )


def save_policy_parameters(
    save_dir,
    epi,
    consumer_policy,
    firm_policy,
    government_policy,
    freeze_firms,
    freeze_govt,
):
    print(f"saving model parameters at episode {epi}")
    consumer_path = (
        Path(save_dir) / Path("saved_models") / Path(f"consumer_policy_{epi}.pt")
    )

    # always save the latest, to be overwritten later
    consumer_path_latest = (
        Path(save_dir) / Path("saved_models") / Path("consumer_policy_latest.pt")
    )
    os.makedirs(consumer_path.parent, exist_ok=True)
    torch.save(consumer_policy.state_dict(), consumer_path)
    torch.save(consumer_policy.state_dict(), consumer_path_latest)

    if freeze_firms is None:
        firm_path = (
            Path(save_dir) / Path("saved_models") / Path(f"firm_policy_{epi}.pt")
        )
        firm_path_latest = (
            Path(save_dir) / Path("saved_models") / Path("firm_policy_latest.pt")
        )

        os.makedirs(firm_path.parent, exist_ok=True)
        torch.save(firm_policy.state_dict(), firm_path)
        torch.save(firm_policy.state_dict(), firm_path_latest)
    if freeze_govt is None:
        government_path = (
            Path(save_dir) / Path("saved_models") / Path(f"government_policy_{epi}.pt")
        )
        government_path_latest = (
            Path(save_dir) / Path("saved_models") / Path("government_policy_latest.pt")
        )

        os.makedirs(government_path.parent, exist_ok=True)
        torch.save(government_policy.state_dict(), government_path)
        torch.save(government_policy.state_dict(), government_path_latest)


class ConsumerFirmRunManagerBatchParallel:
    """
    The Real Business Cycle Experiment Management Class.
    """

    def __init__(self, cfg_dict, freeze_firms=None, freeze_govt=None):
        self.cfg_dict = cfg_dict
        self.train_dict = cfg_dict["train"]
        self.agents_dict = cfg_dict["agents"]
        self.world_dict = cfg_dict["world"]
        self.save_dense_every = self.train_dict["save_dense_every"]
        self.save_dir = self.train_dict["save_dir"]

        self.freeze_firms = freeze_firms
        self.freeze_govt = freeze_govt

        self.__init_cuda_functions()
        self.__init_cuda_data_structs()
        self.__init_torch_data()

    def __init_cuda_data_structs(self):
        __td = self.train_dict
        __ad = self.agents_dict
        __wd = self.world_dict
        batch_size = __td["batch_size"]
        num_consumers = __ad["num_consumers"]
        num_firms = __ad["num_firms"]
        num_governments = __ad["num_governments"]
        firm_action_dim = __ad["firm_action_dim"]
        government_action_dim = __ad["government_action_dim"]
        consumer_state_dim = __ad["consumer_state_dim"]
        firm_state_dim = __ad["firm_state_dim"]
        government_state_dim = __ad["government_state_dim"]
        global_state_dim = __ad["global_state_dim"]
        consumer_endowment = __wd["initial_consumer_endowment"]
        firm_endowment = __wd["initial_firm_endowment"]
        initial_stocks = __wd["initial_stocks"]
        initial_wages = __wd["initial_wages"]
        initial_prices = __wd["initial_prices"]
        consumer_theta = __wd["consumer_theta"]

        consumer_rewards = np.zeros((batch_size, num_consumers), dtype=_NP_DTYPE)
        consumer_states = np.zeros(
            (batch_size, num_consumers, consumer_state_dim), dtype=_NP_DTYPE
        )

        firm_action_indices = np.zeros((batch_size, num_firms), dtype=np.int32)
        firm_actions = np.zeros(
            (batch_size, num_firms, firm_action_dim), dtype=_NP_DTYPE
        )
        firm_rewards = np.zeros((batch_size, num_firms), dtype=_NP_DTYPE)
        firm_states = np.zeros((batch_size, num_firms, firm_state_dim), dtype=_NP_DTYPE)

        government_action_indices = np.zeros(
            (batch_size, num_governments), dtype=np.int32
        )
        government_actions = np.zeros(
            (batch_size, num_governments, government_action_dim), dtype=_NP_DTYPE
        )
        government_rewards = np.zeros((batch_size, num_governments), dtype=_NP_DTYPE)
        government_states = np.zeros(
            (batch_size, num_governments, government_state_dim), dtype=_NP_DTYPE
        )

        # initialize states to right values here

        # global state init
        # for consumers, firms, and governments
        for state_arr in [consumer_states, firm_states, government_states]:
            # set prices to 1.0
            state_arr[:, :, 0:num_firms] = initial_prices
            # set wages to 0.0
            state_arr[:, :, num_firms : (2 * num_firms)] = initial_wages
            # set stocks to 0.0
            state_arr[:, :, (2 * num_firms) : (3 * num_firms)] = initial_stocks
            # set goods overdemanded to 0.0
            state_arr[:, :, (3 * num_firms) : (4 * num_firms)] = 0.0
            # set taxes to 0.0
            state_arr[:, :, (4 * num_firms)] = 0.0
            state_arr[:, :, (4 * num_firms) + 1] = 0.0

        # consumer states, set theta and initial budget
        if "paretoscaletheta" in __wd:
            pareto_vals = np.expand_dims(
                scipy.stats.pareto.ppf(
                    (np.arange(num_consumers) / num_consumers), __wd["paretoscaletheta"]
                ),
                axis=0,
            )
            consumer_states[:, :, consumer_state_dim - 1] = consumer_theta * (
                1.0 / pareto_vals
            )
        else:
            consumer_states[:, :, consumer_state_dim - 1] = consumer_theta
        consumer_states[:, :, global_state_dim] = consumer_endowment

        # firm states
        # capital
        if __wd.get("initial_capital", None) == "proportional":
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 1] = ((i + 1) / 10.0) * 2.0
        elif __wd.get("initial_capital", None) == "twolevel":
            for i in range(num_firms):
                if i < (num_firms // 2):
                    firm_states[:, i, global_state_dim + 1] = 5000
                else:
                    firm_states[:, i, global_state_dim + 1] = 10000
        else:
            firm_states[:, :, global_state_dim + 1] = 1.0

        # production alpha
        if __wd["production_alpha"] == "proportional":
            half_firms = num_firms // 2
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 2] = ((i % half_firms) + 1) * 0.2
        elif __wd["production_alpha"] == "fixed_array":
            alpha_arr = [0.2, 0.3, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.6, 0.8]
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 2] = alpha_arr[i]
        else:
            for i in range(num_firms):
                firm_states[:, i, global_state_dim + 2] = __wd["production_alpha"]

        # set one-hot fields correctly by index for each firm
        onehot_rows = np.eye(num_firms)
        firm_states[:, :, (global_state_dim + 3) :] = onehot_rows
        firm_states[:, :, global_state_dim] = firm_endowment

        # government states
        # for now, nothing beyond global state

        self.consumer_states_gpu_tensor = torch.from_numpy(consumer_states).cuda()
        # these are now tensors bc sampling for consumers via pytorch
        self.consumer_rewards_gpu_pycuda = cuda_driver.mem_alloc(
            consumer_rewards.nbytes
        )
        self.consumer_states_checkpoint_gpu_pycuda = cuda_driver.mem_alloc(
            consumer_states.nbytes
        )
        cuda_driver.memcpy_htod(self.consumer_rewards_gpu_pycuda, consumer_rewards)
        cuda_driver.memcpy_htod(
            self.consumer_states_checkpoint_gpu_pycuda, consumer_states
        )

        self.firm_states_gpu_tensor = torch.from_numpy(firm_states).cuda()
        self.firm_action_indices_gpu_pycuda = cuda_driver.mem_alloc(
            firm_action_indices.nbytes
        )
        self.firm_actions_gpu_pycuda = cuda_driver.mem_alloc(firm_actions.nbytes)
        self.firm_rewards_gpu_pycuda = cuda_driver.mem_alloc(firm_rewards.nbytes)
        self.firm_states_checkpoint_gpu_pycuda = cuda_driver.mem_alloc(
            firm_states.nbytes
        )
        cuda_driver.memcpy_htod(
            self.firm_action_indices_gpu_pycuda, firm_action_indices
        )
        cuda_driver.memcpy_htod(self.firm_actions_gpu_pycuda, firm_actions)
        cuda_driver.memcpy_htod(self.firm_rewards_gpu_pycuda, firm_rewards)
        cuda_driver.memcpy_htod(self.firm_states_checkpoint_gpu_pycuda, firm_states)

        self.government_states_gpu_tensor = tor